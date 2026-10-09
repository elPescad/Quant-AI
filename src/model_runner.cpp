// Offline model evaluation: runs the exact C++ inference path (feature pipeline +
// simulated quantum feature map + native C++ GRU, no libtorch) over a data split and
// scores the predictions against the CSV labels. No trading, no threads.
//
// Its accuracy should match what python/train_and_export.py reports for the same
// split; a mismatch means the C++ and Python feature pipelines have drifted apart.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "feature_pipeline.hpp"
#include "latency_stats.hpp"
#include "market_data.hpp"
#include "native_model.hpp"

int main(int argc, char** argv) {
    std::string data_path = "../data/market_ticks.csv";
    std::string model_path = "../models/ensemble_model.weights";
    std::string config_path;
    std::string period = "test";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--data") data_path = next();
        else if (a == "--model") model_path = next();
        else if (a == "--config") config_path = next();
        else if (a == "--period") period = next();
        else {
            std::cout << "Usage: model_runner [--data PATH] [--model PATH] [--config PATH] [--period test|val|train|all]\n";
            return a == "--help" ? 0 : 1;
        }
    }
    if (config_path.empty()) {
        const size_t slash = model_path.find_last_of('/');
        const size_t dot = model_path.find_last_of('.');
        const bool has_ext = dot != std::string::npos && (slash == std::string::npos || dot > slash);
        config_path = (has_ext ? model_path.substr(0, dot) : model_path) + "_config.txt";
    }

    FeatureConfig cfg;
    try {
        cfg = FeatureConfig::load(config_path);
    } catch (const std::exception& e) {
        std::cerr << "[-] " << e.what() << std::endl;
        return 1;
    }

    int64_t start = INT64_MIN, end = INT64_MAX;
    if (period == "test") start = cfg.test_start_ts;
    else if (period == "val") { start = cfg.val_start_ts; end = cfg.test_start_ts; }
    else if (period == "train") end = cfg.val_start_ts;
    else if (period != "all") { std::cerr << "[-] Unknown period " << period << std::endl; return 1; }

    NativeModel module;
    try {
        module = NativeModel::load(model_path);
    } catch (const std::exception& e) {
        std::cerr << "[-] Error loading model: " << e.what() << std::endl;
        return 1;
    }
    if (module.input_dim() != cfg.input_dim()) {
        std::cerr << "[-] Model reads " << module.input_dim() << " features but the config produces " << cfg.input_dim() << std::endl;
        return 1;
    }

    std::ifstream file(data_path);
    std::string line;
    if (!file.is_open() || !std::getline(file, line) || !is_tick_csv_header(line)) {
        std::cerr << "[-] Cannot read " << data_path << " (expected header: " << kTickCsvHeader << ")" << std::endl;
        return 1;
    }

    std::map<std::string, std::unique_ptr<TickerFeaturePipeline>> pipes;
    std::vector<float> window(static_cast<size_t>(cfg.seq_len) * cfg.input_dim(), 0.0f);

    long confusion[3][3] = {{0}};
    LatencyStats feature_us, model_us;
    std::vector<float> abs_edge;
    MarketTick tick;
    while (std::getline(file, line)) {
        if (!parse_tick_row(line, tick)) continue;
        auto& pipe = pipes[tick.ticker];
        if (!pipe) pipe = std::make_unique<TickerFeaturePipeline>(cfg);

        auto t0 = std::chrono::steady_clock::now();
        const bool ready = pipe->update(tick);
        auto t1 = std::chrono::steady_clock::now();
        feature_us.add(std::chrono::duration<double, std::micro>(t1 - t0).count());

        if (!ready || tick.target < 0 || tick.timestamp < start || tick.timestamp >= end) continue;

        pipe->copy_window(window.data());
        float probs[3];
        auto t2 = std::chrono::steady_clock::now();
        module.predict(window.data(), cfg.seq_len, cfg.input_dim(), probs);
        auto t3 = std::chrono::steady_clock::now();
        model_us.add(std::chrono::duration<double, std::micro>(t3 - t2).count());

        const int pred = static_cast<int>(std::max_element(probs, probs + 3) - probs);
        // The engine trades on p_buy - p_sell, so its spread decides how often costs are beaten
        abs_edge.push_back(std::abs(probs[2] - probs[0]));
        confusion[tick.target][pred]++;
    }

    long total = 0, correct = 0;
    for (int t = 0; t < 3; ++t)
        for (int p = 0; p < 3; ++p) { total += confusion[t][p]; if (t == p) correct += confusion[t][p]; }
    if (total == 0) {
        std::cerr << "[-] No labelled samples in period '" << period << "'" << std::endl;
        return 1;
    }

    double f1_sum = 0.0;
    const char* names[3] = {"SELL", "HOLD", "BUY"};
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "=== Model Evaluation (" << period << ", " << total << " samples) ===\n"
              << "Model: " << model_path << " | quantum_lift=" << cfg.quantum_lift << " | input_dim=" << cfg.input_dim() << "\n\n"
              << "Confusion matrix (rows = true, cols = predicted)\n"
              << "          SELL     HOLD      BUY\n";
    for (int t = 0; t < 3; ++t) {
        std::cout << std::left << std::setw(6) << names[t] << std::right;
        for (int p = 0; p < 3; ++p) std::cout << std::setw(9) << confusion[t][p];
        std::cout << "\n";
    }
    std::cout << "\n";
    for (int c = 0; c < 3; ++c) {
        long tp = confusion[c][c], pred_c = 0, true_c = 0;
        for (int k = 0; k < 3; ++k) { pred_c += confusion[k][c]; true_c += confusion[c][k]; }
        const double prec = pred_c ? static_cast<double>(tp) / pred_c : 0.0;
        const double rec = true_c ? static_cast<double>(tp) / true_c : 0.0;
        const double f1 = (prec + rec) > 0 ? 2 * prec * rec / (prec + rec) : 0.0;
        f1_sum += f1;
        std::cout << std::left << std::setw(5) << names[c] << std::right << " precision " << prec << " recall " << rec
                  << " f1 " << f1 << "\n";
    }

    auto mean = [](const std::vector<float>& v) { return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size(); };
    auto pct = [](std::vector<float> v, double q) {
        std::sort(v.begin(), v.end());
        return v.empty() ? 0.0 : static_cast<double>(v[std::min(v.size() - 1, static_cast<size_t>(q * v.size()))]);
    };
    std::cout << "\nAccuracy:   " << static_cast<double>(correct) / total << "\n"
              << "Macro F1:   " << f1_sum / 3.0 << "\n"
              << std::setprecision(2)
              << "Edge |p_buy - p_sell|: mean " << mean(abs_edge) << " | p50 " << pct(abs_edge, 0.50)
              << " | p90 " << pct(abs_edge, 0.90) << " | p99 " << pct(abs_edge, 0.99) << "\n"
              << "Latency:    features+quantum sim " << feature_us.mean() << " us | GRU forward " << model_us.mean()
              << " us (p99 " << model_us.quantile(0.99) << ")\n\n"
              << "AR(1) order-flow persistence (RLS, end of data):\n";
    for (const auto& [name, pipe] : pipes) {
        const auto& ar = pipe->ar();
        std::cout << "  " << std::left << std::setw(6) << name << std::right << std::setprecision(3) << " phi " << ar.phi()
                  << " +/- " << ar.phi_stderr() << "  intercept " << ar.intercept() << "\n";
    }
    return 0;
}
