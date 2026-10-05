// Offline model evaluation: runs the exact C++ inference path (feature pipeline +
// simulated quantum feature map + TorchScript GRU) over a data split and scores the
// predictions against the CSV labels. No trading, no threads.
//
// Its accuracy should match what python/train_and_export.py reports for the same
// split; a mismatch means the C++ and Python feature pipelines have drifted apart.

#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
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
#include "market_data.hpp"

int main(int argc, char** argv) {
    std::string data_path = "../data/market_ticks.csv";
    std::string model_path = "../models/quant_model.pt";
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
        std::string base = model_path;
        if (base.size() > 3 && base.substr(base.size() - 3) == ".pt") base = base.substr(0, base.size() - 3);
        config_path = base + "_config.txt";
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

    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path);
        module.eval();
    } catch (const c10::Error& e) {
        std::cerr << "[-] Error loading model: " << e.what() << std::endl;
        return 1;
    }
    at::set_num_threads(1);
    torch::InferenceMode guard;

    std::ifstream file(data_path);
    std::string line;
    if (!file.is_open() || !std::getline(file, line) || !is_tick_csv_header(line)) {
        std::cerr << "[-] Cannot read " << data_path << " (expected header: " << kTickCsvHeader << ")" << std::endl;
        return 1;
    }

    std::map<std::string, std::unique_ptr<TickerFeaturePipeline>> pipes;
    torch::Tensor input = torch::zeros({1, cfg.seq_len, cfg.input_dim()}, torch::kFloat32);

    long confusion[3][3] = {{0}};
    std::vector<double> feature_us, model_us;
    MarketTick tick;
    while (std::getline(file, line)) {
        if (!parse_tick_row(line, tick)) continue;
        auto& pipe = pipes[tick.ticker];
        if (!pipe) pipe = std::make_unique<TickerFeaturePipeline>(cfg);

        auto t0 = std::chrono::steady_clock::now();
        const bool ready = pipe->update(tick);
        auto t1 = std::chrono::steady_clock::now();
        feature_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());

        if (!ready || tick.target < 0 || tick.timestamp < start || tick.timestamp >= end) continue;

        pipe->copy_window(input.data_ptr<float>());
        auto t2 = std::chrono::steady_clock::now();
        torch::Tensor logits = module.forward({input}).toTensor();
        auto t3 = std::chrono::steady_clock::now();
        model_us.push_back(std::chrono::duration<double, std::micro>(t3 - t2).count());

        const int pred = static_cast<int>(logits.argmax(1).item<int64_t>());
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

    auto mean = [](const std::vector<double>& v) { return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size(); };
    std::cout << "\nAccuracy:   " << static_cast<double>(correct) / total << "\n"
              << "Macro F1:   " << f1_sum / 3.0 << "\n"
              << std::setprecision(2)
              << "Latency:    features+quantum sim " << mean(feature_us) << " us | GRU forward " << mean(model_us) << " us\n\n"
              << "AR(1) order-flow persistence (RLS, end of data):\n";
    for (const auto& [name, pipe] : pipes) {
        const auto& ar = pipe->ar();
        std::cout << "  " << std::left << std::setw(6) << name << std::right << std::setprecision(3) << " phi " << ar.phi()
                  << " +/- " << ar.phi_stderr() << "  intercept " << ar.intercept() << "\n";
    }
    return 0;
}
