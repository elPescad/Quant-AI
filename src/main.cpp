// Low-latency quant paper-trading engine.
//
//   producer thread : CSV file or stdin -> MarketTick -> SPSC ring buffer
//   consumer thread : per-ticker features (EWMA z-scores, RLS AR(1), simulated ZZ quantum
//                     feature map) -> native C++ GRU (no libtorch) -> per-bar allocation
//                     (greedy, or QUBO solved by simulated annealing) -> portfolio
//                     rebalance + per-tick risk exits
//
// By default only the held-out test period (from the model's config) is traded; the
// earlier data is streamed through to warm up the feature state.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#endif

#include "feature_pipeline.hpp"
#include "latency_stats.hpp"
#include "market_data.hpp"
#include "native_model.hpp"
#include "portfolio.hpp"
#include "qubo_allocator.hpp"
#include "ring_buffer.hpp"

namespace {

constexpr int COOLDOWN_BARS = 6; // No re-entry for this many bars after a risk exit

struct EngineOptions {
    std::string data_path = "../data/market_ticks.csv";
    std::string model_path = "../models/ensemble_model.weights";
    std::string config_path; // Defaults to <model>_config.txt
    std::string trades_path = "trades.csv";
    std::string allocator = "greedy"; // greedy | qubo  (greedy == QUBO at gamma 0, the walk-forward pick)
    std::string period = "test";    // test | val | all
    int64_t start_ts = INT64_MIN;
    int64_t end_ts = INT64_MAX;
    int max_positions = 4;
    double risk_aversion = 100.0;
    double fee_bps = 0.2; // Per fill
    bool verify_qubo = false;
    bool verbose = false;
};

void usage() {
    std::cout << "Usage: quant_engine [options]\n"
                 "  --data PATH            market ticks CSV, or - for stdin (default ../data/market_ticks.csv)\n"
                 "  --model PATH           .weights model (default ../models/ensemble_model.weights)\n"
                 "  --config PATH          feature config (default <model>_config.txt)\n"
                 "  --period test|val|all  which split to trade (default test = out-of-sample)\n"
                 "  --start-ts T --end-ts T  explicit trading window (unix seconds)\n"
                 "  --allocator greedy|qubo  position selection (default greedy = QUBO at gamma 0)\n"
                 "  --risk-aversion X      QUBO risk aversion gamma (default 100)\n"
                 "  --max-positions K      max simultaneous positions (default 4)\n"
                 "  --fee-bps X            fees per fill in basis points (default 0.2)\n"
                 "  --verify-qubo          check every SA solution against brute force\n"
                 "  --trades PATH          trade log (default trades.csv)\n"
                 "  --verbose              per-bar debug output\n";
}

bool parse_args(int argc, char** argv, EngineOptions& o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + a);
            return argv[++i];
        };
        if (a == "--data") o.data_path = next();
        else if (a == "--model") o.model_path = next();
        else if (a == "--config") o.config_path = next();
        else if (a == "--trades") o.trades_path = next();
        else if (a == "--allocator") o.allocator = next();
        else if (a == "--period") o.period = next();
        else if (a == "--start-ts") o.start_ts = std::stoll(next());
        else if (a == "--end-ts") o.end_ts = std::stoll(next());
        else if (a == "--max-positions") o.max_positions = std::stoi(next());
        else if (a == "--risk-aversion") o.risk_aversion = std::stod(next());
        else if (a == "--fee-bps") o.fee_bps = std::stod(next());
        else if (a == "--verify-qubo") o.verify_qubo = true;
        else if (a == "--verbose") o.verbose = true;
        else if (a == "--help" || a == "-h") { usage(); return false; }
        else throw std::runtime_error("Unknown option " + a);
    }
    if (o.config_path.empty()) {
        const size_t slash = o.model_path.find_last_of('/');
        const size_t dot = o.model_path.find_last_of('.');
        const bool has_ext = dot != std::string::npos && (slash == std::string::npos || dot > slash);
        o.config_path = (has_ext ? o.model_path.substr(0, dot) : o.model_path) + "_config.txt";
    }
    if (o.allocator != "qubo" && o.allocator != "greedy") throw std::runtime_error("--allocator must be qubo or greedy");
    return true;
}

LockFreeRingBuffer<MarketTick, 8192> event_queue;
std::atomic<bool> stream_finished(false);
std::atomic<bool> stop_requested(false);
std::atomic<long> rows_skipped(0);

void pin_thread_to_core(std::thread& th, int core_id) {
#ifdef __linux__
    if (core_id < 0 || core_id >= static_cast<int>(std::thread::hardware_concurrency())) return;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset);
#else
    (void)th;
    (void)core_id;
#endif
}

void file_stream_producer(const std::string& csv_file) {
    std::ifstream file;
    if (csv_file != "-") {
        file.open(csv_file);
        if (!file.is_open()) {
            std::cerr << "[-] Error opening market data file: " << csv_file << std::endl;
            stream_finished.store(true, std::memory_order_release);
            return;
        }
    }
    std::istream& input = csv_file == "-" ? std::cin : file;

    std::string line;
    std::getline(input, line);
    if (!is_tick_csv_header(line)) {
        std::cerr << "[-] Unexpected CSV header: " << line << "\n"
                  << "    Expected: " << kTickCsvHeader << "\n"
                  << "    Regenerate data with python/fetch_real_ticks.py or python/generate_ticks.py" << std::endl;
        stream_finished.store(true, std::memory_order_release);
        return;
    }

    int tick_id = 0;
    Backoff backoff;
    while (std::getline(input, line) && !stop_requested.load(std::memory_order_relaxed)) {
        MarketTick tick;
        if (!parse_tick_row(line, tick)) {
            rows_skipped.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        tick.id = tick_id++;
        backoff.reset();
        while (!event_queue.push(tick)) [[unlikely]] {
            if (stop_requested.load(std::memory_order_relaxed)) break;
            backoff.wait();
        }
    }
    stream_finished.store(true, std::memory_order_release);
}

struct TickerState {
    explicit TickerState(const FeatureConfig& cfg, int asset_idx)
        : pipeline(cfg), moves(cfg.label_horizon), asset(asset_idx) {}

    TickerFeaturePipeline pipeline;
    HorizonMoveTracker moves;
    int asset;
    double edge = 0.0;      // p_buy - p_sell from the latest model call
    long signal_bar = -1;   // Bar index of that call
    long cooldown_until = -1;
    double last_price = 0.0;
    double prev_bar_price = 0.0;
    bool seen_this_bar = false;
};

struct EngineStats {
    LatencyStats feature_us; // Feature pipeline incl. quantum circuit simulation
    LatencyStats model_us;   // GRU forward
    LatencyStats alloc_us;   // Allocation per bar
    long bars = 0;
    long trading_bars = 0;
    long ticks = 0;
    double cost_sum = 0.0; // One-way costs the allocator was given
    long cost_n = 0;
    long quoted_ticks = 0; // Ticks carrying a measured bid-ask spread
};

class Engine {
public:
    Engine(const EngineOptions& opt, const FeatureConfig& cfg, NativeModel& model)
        : opt_(opt), cfg_(cfg), model_(model),
          portfolio_(make_portfolio_params(opt)),
          risk_(0.99),
          allocator_(make_alloc_params(opt, cfg), risk_),
          window_(static_cast<size_t>(cfg.seq_len) * cfg.input_dim(), 0.0f) {}

    void run() {
        MarketTick tick;
        Backoff backoff;
        while (true) {
            // Read the flag BEFORE popping: if the producer had already finished and the
            // queue is still empty, every tick has been consumed.
            const bool producer_done = stream_finished.load(std::memory_order_acquire);
            if (event_queue.try_pop(tick)) [[likely]] {
                backoff.reset();
                if (!done_trading_) on_tick(tick);
            } else if (producer_done) {
                break;
            } else {
                backoff.wait();
            }
        }
        if (!done_trading_) {
            if (bar_ts_ != INT64_MIN) close_bar();
            portfolio_.liquidate_all(last_tick_id_);
        }
    }

    void report(double wall_sec) const {
        auto pct = [](const LatencyStats& v, double q) { return v.quantile(q); };
        auto mean = [](const LatencyStats& v) { return v.mean(); };

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\n==================================================\n"
                  << "          SYSTEM & HARDWARE BENCHMARKS\n"
                  << "==================================================\n"
                  << "Ticks Processed:       " << stats_.ticks << " (" << stats_.bars << " bars, "
                  << stats_.trading_bars << " traded)\n"
                  << "Feature+Quantum Sim:   avg " << mean(stats_.feature_us) << " us | p99 " << pct(stats_.feature_us, 0.99) << " us\n"
                  << "GRU Forward:           avg " << mean(stats_.model_us) << " us | p99 " << pct(stats_.model_us, 0.99) << " us\n"
                  << "Allocation/bar:        avg " << mean(stats_.alloc_us) << " us | p99 " << pct(stats_.alloc_us, 0.99) << " us\n"
                  << "Throughput:            " << static_cast<long>(stats_.ticks / std::max(wall_sec, 1e-9)) << " ticks/sec\n";

        if (opt_.allocator == "qubo") {
            const auto& s = allocator_.stats();
            std::cout << "QUBO Solves:           " << s.solves << " (avg "
                      << (s.solves ? static_cast<double>(s.variables_total) / s.solves : 0.0) << " binary vars)\n";
            if (s.verified > 0) {
                std::cout << "SA == Brute Force:     " << s.matched_optimum << "/" << s.verified << " ("
                          << 100.0 * s.matched_optimum / s.verified << "%), max energy gap "
                          << std::setprecision(6) << s.max_energy_gap << std::setprecision(2) << "\n";
            }
        }

        std::cout << "\n==================================================\n"
                  << "     AR(1) ORDER-FLOW PERSISTENCE (RLS estimate)\n"
                  << "==================================================\n";
        for (const auto& name : asset_names_) {
            const auto& ar = states_.at(name)->pipeline.ar();
            std::cout << std::left << std::setw(6) << name << std::right << " phi = " << std::setprecision(3) << ar.phi()
                      << " +/- " << ar.phi_stderr() << "  half-life " << std::setprecision(1) << ar.half_life() << " bars\n";
        }

        std::cout << std::setprecision(2)
                  << "\n==================================================\n"
                  << "        PAPER TRADING PORTFOLIO RESULTS\n"
                  << "==================================================\n"
                  << "Model:                 " << opt_.model_path << " (quantum_lift=" << cfg_.quantum_lift
                  << ", input_dim=" << cfg_.input_dim() << ")\n"
                  << "Allocator:             " << opt_.allocator;
        if (opt_.allocator == "qubo") std::cout << " (gamma=" << opt_.risk_aversion << ", K=" << opt_.max_positions << ")";
        std::cout << "\nPeriod:                " << opt_.period << " ["
                  << (trade_start_ == INT64_MIN ? std::string("start") : std::to_string(trade_start_)) << ", "
                  << (trade_end_ == INT64_MAX ? std::string("end") : std::to_string(trade_end_)) << ")\n"
                  << "Starting Equity:       $10000.00\n"
                  << "Ending Equity:         $" << portfolio_.get_total_equity() << "\n"
                  << "Total Net PnL:         $" << portfolio_.get_pnl() << " (" << portfolio_.get_return_pct() << "%)\n"
                  << "Fees Paid:             $" << portfolio_.get_fees_paid() << "\n"
                  << "Avg One-Way Cost:      " << (stats_.cost_n ? 1e4 * stats_.cost_sum / stats_.cost_n : 0.0)
                  << " bp (spread from " << (stats_.quoted_ticks * 2 > stats_.ticks ? "measured quotes" : "high-low proxy, capped")
                  << ", fees " << opt_.fee_bps << " bp/fill; opening a position budgets entry + exit)\n"
                  << "Total Round Trips:     " << portfolio_.get_total_trades() << " (" << portfolio_.get_risk_exits()
                  << " stop/take-profit exits)\n"
                  << "Win Rate:              " << portfolio_.get_win_rate() << " %\n"
                  << "Avg Gross Exposure:    " << portfolio_.get_avg_gross_exposure() << " % of equity\n"
                  << "Max Drawdown:          " << portfolio_.get_max_drawdown() << " %\n"
                  << "Sharpe Ratio (ann.):   " << portfolio_.calculate_sharpe_ratio() << "\n"
                  << "==================================================\n";
        if (rows_skipped.load() > 0) std::cout << "[!] Skipped " << rows_skipped.load() << " malformed CSV rows\n";
    }

    void set_window(int64_t start, int64_t end) {
        trade_start_ = start;
        trade_end_ = end;
    }

private:
    static PortfolioParams make_portfolio_params(const EngineOptions& o) {
        PortfolioParams p;
        p.trade_log = o.trades_path;
        p.fee_rate = o.fee_bps * 1e-4;
        p.position_weight = 1.0 / std::max(1, o.max_positions);
        return p;
    }

    static AllocatorParams make_alloc_params(const EngineOptions& o, const FeatureConfig& cfg) {
        AllocatorParams a;
        a.max_positions = o.max_positions;
        a.risk_aversion = o.risk_aversion;
        a.position_weight = 1.0 / std::max(1, o.max_positions);
        a.horizon = cfg.label_horizon;
        a.verify_with_brute_force = o.verify_qubo;
        return a;
    }

    bool trading_now() const { return bar_ts_ >= trade_start_ && bar_ts_ < trade_end_; }

    TickerState& state_for(const std::string& sym) {
        auto it = states_.find(sym);
        if (it != states_.end()) return *it->second;
        int idx = risk_.add_asset();
        asset_names_.push_back(sym);
        auto [ins, ok] = states_.emplace(sym, std::make_unique<TickerState>(cfg_, idx));
        return *ins->second;
    }

    void on_tick(const MarketTick& tick) {
        if (tick.timestamp != bar_ts_) {
            if (bar_ts_ != INT64_MIN) close_bar();
            if (tick.timestamp >= trade_end_) {
                // Past the trading window: flatten and stop the producer early
                portfolio_.liquidate_all(last_tick_id_);
                done_trading_ = true;
                stop_requested.store(true, std::memory_order_relaxed);
                return;
            }
            bar_ts_ = tick.timestamp;
            bar_index_++;
            stats_.bars++;
        }

        stats_.ticks++;
        last_tick_id_ = tick.id;
        const std::string sym(tick.ticker);
        TickerState& st = state_for(sym);

        auto t0 = std::chrono::steady_clock::now();
        const bool ready = st.pipeline.update(tick);
        auto t1 = std::chrono::steady_clock::now();
        stats_.feature_us.add(std::chrono::duration<double, std::micro>(t1 - t0).count());

        st.last_price = tick.raw_price;
        st.seen_this_bar = true;

        stats_.quoted_ticks += tick.quoted_spread > 0.0f;
        if (portfolio_.mark(tick.id, sym, tick.raw_price, tick.raw_spread, tick.quoted_spread)) {
            st.cooldown_until = bar_index_ + COOLDOWN_BARS;
        }

        if (ready && trading_now()) {
            st.pipeline.copy_window(window_.data());
            float p[3];
            auto t2 = std::chrono::steady_clock::now();
            model_.predict(window_.data(), cfg_.seq_len, cfg_.input_dim(), p);
            auto t3 = std::chrono::steady_clock::now();
            stats_.model_us.add(std::chrono::duration<double, std::micro>(t3 - t2).count());

            st.edge = static_cast<double>(p[2]) - static_cast<double>(p[0]);
            st.signal_bar = bar_index_;
        }
    }

    void close_bar() {
        // 1. Risk model: 1-bar returns of every ticker that printed in this bar
        const int n = risk_.size();
        std::vector<double> rets(n, 0.0);
        std::vector<uint8_t> valid(n, 0);
        for (const auto& name : asset_names_) {
            TickerState& st = *states_[name];
            if (!st.seen_this_bar) continue;
            if (st.prev_bar_price > 0.0 && st.last_price > 0.0) {
                rets[st.asset] = st.last_price / st.prev_bar_price - 1.0;
                valid[st.asset] = 1;
            }
            st.moves.on_bar_close(st.last_price);
        }
        risk_.update(rets, valid);

        // 2. Allocation over tickers with a fresh signal
        if (trading_now()) {
            stats_.trading_bars++;
            auto t0 = std::chrono::steady_clock::now();

            std::vector<AssetView> views;
            std::vector<std::string> names;
            int held_outside = 0;
            for (const auto& name : asset_names_) {
                TickerState& st = *states_[name];
                const int cur = portfolio_.direction(name);
                const bool eligible = st.signal_bar == bar_index_ && st.moves.ready() && risk_.warm() &&
                                      bar_index_ >= st.cooldown_until;
                if (!eligible) {
                    held_outside += cur != 0;
                    continue;
                }
                AssetView v;
                v.asset = st.asset;
                v.expected_return = st.edge * st.moves.typical_move();
                v.cost = portfolio_.one_way_cost(name);
                stats_.cost_sum += v.cost;
                stats_.cost_n++;
                v.current = cur;
                views.push_back(v);
                names.push_back(name);
            }

            const int capacity = opt_.max_positions - held_outside;
            std::vector<int> dirs = (opt_.allocator == "qubo") ? allocator_.allocate(views, capacity)
                                                               : QuboAllocator::greedy(views, capacity);
            std::vector<std::pair<std::string, int>> targets;
            for (size_t i = 0; i < names.size(); ++i) targets.push_back({names[i], dirs[i]});
            portfolio_.rebalance(last_tick_id_, targets);

            auto t1 = std::chrono::steady_clock::now();
            if (!views.empty()) stats_.alloc_us.add(std::chrono::duration<double, std::micro>(t1 - t0).count());
            portfolio_.record_equity();

            if (opt_.verbose && bar_index_ % 50 == 0) {
                std::cout << "[BAR " << bar_ts_ << "]";
                for (size_t i = 0; i < names.size(); ++i)
                    std::cout << " " << names[i] << ":" << dirs[i] << "(mu=" << views[i].expected_return * 1e4 << "bp)";
                std::cout << " | equity $" << portfolio_.get_total_equity() << "\n";
            }
        }

        for (const auto& name : asset_names_) {
            TickerState& st = *states_[name];
            if (st.seen_this_bar) st.prev_bar_price = st.last_price;
            st.seen_this_bar = false;
        }
    }

    const EngineOptions& opt_;
    const FeatureConfig& cfg_;
    NativeModel& model_;
    PortfolioManager portfolio_;
    RiskModel risk_;
    QuboAllocator allocator_;
    std::vector<float> window_;

    std::unordered_map<std::string, std::unique_ptr<TickerState>> states_;
    std::vector<std::string> asset_names_;
    int64_t bar_ts_ = INT64_MIN;
    long bar_index_ = -1;
    int last_tick_id_ = 0;
    int64_t trade_start_ = INT64_MIN;
    int64_t trade_end_ = INT64_MAX;
    bool done_trading_ = false;
    EngineStats stats_;
};

} // namespace

int main(int argc, char** argv) {
    std::cout << "=== Low-Latency Quantum-Inspired Quant Paper Trading Engine ===" << std::endl;

    EngineOptions opt;
    FeatureConfig cfg;
    try {
        if (!parse_args(argc, argv, opt)) return 0;
        cfg = FeatureConfig::load(opt.config_path);
    } catch (const std::exception& e) {
        std::cerr << "[-] " << e.what() << std::endl;
        return 1;
    }

    // Trading window: out-of-sample by default
    int64_t start = INT64_MIN, end = INT64_MAX;
    if (opt.period == "test") start = cfg.test_start_ts;
    else if (opt.period == "val") { start = cfg.val_start_ts; end = cfg.test_start_ts; }
    else if (opt.period != "all") { std::cerr << "[-] --period must be test, val or all" << std::endl; return 1; }
    if (opt.start_ts != INT64_MIN) start = opt.start_ts;
    if (opt.end_ts != INT64_MAX) end = opt.end_ts;
    if (opt.period == "all") std::cout << "[!] Trading the full file: this includes the training period (in-sample)." << std::endl;

    NativeModel module;
    try {
        module = NativeModel::load(opt.model_path);
    } catch (const std::exception& e) {
        std::cerr << "[-] Error loading model: " << e.what() << std::endl;
        return 1;
    }
    if (module.input_dim() != cfg.input_dim()) {
        std::cerr << "[-] Model reads " << module.input_dim() << " features but " << opt.config_path << " produces "
                  << cfg.input_dim() << "; the model and config must come from the same training run" << std::endl;
        return 1;
    }
    std::cout << "[+] Model " << opt.model_path << " | quantum_lift=" << cfg.quantum_lift << " (" << cfg.n_qubits
              << " qubits, " << cfg.reps << " reps, bandwidth " << cfg.bandwidth << ") | input_dim=" << cfg.input_dim()
              << " | seq_len=" << cfg.seq_len << std::endl;
    std::cout << "[+] Streaming " << opt.data_path << std::endl;

    Engine engine(opt, cfg, module);
    engine.set_window(start, end);

    auto wall_start = std::chrono::steady_clock::now();
    std::thread producer(file_stream_producer, opt.data_path);
    std::thread consumer([&engine] { engine.run(); });
    // Only the latency-critical consumer gets a core of its own (the last one)
    pin_thread_to_core(consumer, static_cast<int>(std::thread::hardware_concurrency()) - 1);
    producer.join();
    consumer.join();
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();

    engine.report(wall);
    return 0;
}
