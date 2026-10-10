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
#include <csignal>
#include <ctime>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#endif

#include "feature_pipeline.hpp"
#include "latency_stats.hpp"
#include "live_bars.hpp"
#include "market_data.hpp"
#include "native_model.hpp"
#include "portfolio.hpp"
#include "qubo_allocator.hpp"
#include "ring_buffer.hpp"
#ifdef QUANT_LIVE
#include "live_feed.hpp"
#include "paper_broker.hpp"
#endif

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
    double online_lr = 0.0;     // > 0: learn from each bar once its label is known
    double online_anchor = 0.1; // pull towards the trained weights
    std::string online_state;   // file that carries the learned adjustments across restarts
    bool verify_qubo = false;
    bool verbose = false;
    int status_every_s = -1; // periodic summary; -1 = 300 s live, off for backtests
    // Live mode (Alpaca)
    bool live = false;
    bool paper_orders = false; // mirror positions into the Alpaca paper account
    int live_check = 0; // seconds; > 0 = connectivity check only
    std::string feed = "iex";
    std::string symbols = "SPY,QQQ,AAPL,NVDA,MSFT,AMD";
    int warmup_days = 30;
    int bar_grace_s = 20;
    int close_grace_s = 90;
    std::string data_url, trading_url, stream_url;
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
                 "  --online-lr X          online learning rate for the model head (default 0 = off)\n"
                 "  --online-anchor X      pull of the learned head towards the trained one (default 0.1)\n"
                 "  --online-state PATH    save/restore what online learning learned (only for the same model)\n"
                 "  --verify-qubo          check every SA solution against brute force\n"
                 "  --trades PATH          trade log (default trades.csv; appended to in live mode)\n"
                 "  --status-every SECS    print a results summary this often (default 300 live, 0 = off)\n"
                 "  --verbose              per-bar debug output\n";
#ifdef QUANT_LIVE
    std::cout << "\n"
                 "Live paper trading on Alpaca market data (keys in APCA_API_KEY_ID / APCA_API_SECRET_KEY):\n"
                 "  --live                 stream live bars and trade them; sleeps while the market is closed\n"
                 "  --live-check [SECS]    test the clock, REST and stream connections, then exit (default 20 s)\n"
                 "  --paper-orders         with --live: place the positions in your Alpaca paper account too\n"
                 "                         (orders.csv / account.csv next to --trades)\n"
                 "  --feed iex|sip         Alpaca data feed (default iex = free plan; sip needs a paid plan)\n"
                 "  --symbols A,B,...      tickers (default SPY,QQQ,AAPL,NVDA,MSFT,AMD)\n"
                 "  --warmup-days N        history fetched at start-up to warm the features (default 30)\n"
                 "  --bar-grace S / --close-grace S   seconds to wait for late minute bars (default 20 / 90)\n"
                 "  --data-url / --trading-url / --stream-url URL   override Alpaca endpoints (testing)\n";
#endif
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
        else if (a == "--online-lr") o.online_lr = std::stod(next());
        else if (a == "--online-anchor") o.online_anchor = std::stod(next());
        else if (a == "--online-state") o.online_state = next();
        else if (a == "--verify-qubo") o.verify_qubo = true;
        else if (a == "--verbose") o.verbose = true;
        else if (a == "--status-every") o.status_every_s = std::stoi(next());
        else if (a == "--live") o.live = true;
        else if (a == "--paper-orders") o.paper_orders = true;
        else if (a == "--live-check") o.live_check = (i + 1 < argc && argv[i + 1][0] != '-') ? std::stoi(next()) : 20;
        else if (a == "--feed") o.feed = next();
        else if (a == "--symbols") o.symbols = next();
        else if (a == "--warmup-days") o.warmup_days = std::stoi(next());
        else if (a == "--bar-grace") o.bar_grace_s = std::stoi(next());
        else if (a == "--close-grace") o.close_grace_s = std::stoi(next());
        else if (a == "--data-url") o.data_url = next();
        else if (a == "--trading-url") o.trading_url = next();
        else if (a == "--stream-url") o.stream_url = next();
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
    if (o.status_every_s < 0) o.status_every_s = o.live ? 300 : 0;
    return true;
}

LockFreeRingBuffer<MarketTick, 8192> event_queue;
std::atomic<bool> stream_finished(false);
std::atomic<bool> stop_requested(false);
std::atomic<long> rows_skipped(0);
WakeSignal consumer_wake;
#ifdef QUANT_LIVE
LiveStatus live_status;
#endif

// "+$1,234.56" / "-$0.42"
std::string usd(double v, bool sign = false) {
    const long long cents = std::llround(std::abs(v) * 100.0);
    std::string whole = std::to_string(cents / 100);
    for (int i = static_cast<int>(whole.size()) - 3; i > 0; i -= 3) whole.insert(static_cast<size_t>(i), ",");
    char frac[8];
    std::snprintf(frac, sizeof(frac), ".%02lld", cents % 100);
    return std::string(cents && v < 0 ? "-" : (sign ? "+" : "")) + "$" + whole + frac;
}

std::string num(double v, int decimals, bool sign = false) {
    if (!std::isfinite(v)) return "n/a";
    char buf[64];
    std::snprintf(buf, sizeof(buf), sign ? "%+.*f" : "%.*f", decimals, v);
    return buf;
}

// New York wall time, e.g. ny_time(t, "%a %Y-%m-%d %H:%M")
std::string ny_time(int64_t t, const char* fmt) {
    const std::time_t local = static_cast<std::time_t>(t + live::new_york_offset(t));
    std::tm tm{};
    gmtime_r(&local, &tm);
    char buf[64];
    std::strftime(buf, sizeof(buf), fmt, &tm);
    return buf;
}

int64_t ny_day(int64_t t) {
    const int64_t local = t + live::new_york_offset(t);
    return local >= 0 ? local / 86400 : (local - 86399) / 86400;
}

int64_t wall_now() { return static_cast<int64_t>(std::time(nullptr)); }

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

void push_tick(const MarketTick& tick) {
    Backoff backoff;
    while (!event_queue.push(tick)) [[unlikely]] {
        if (stop_requested.load(std::memory_order_relaxed)) return;
        backoff.wait();
    }
    consumer_wake.notify();
}

void finish_stream() {
    stream_finished.store(true, std::memory_order_release);
    consumer_wake.notify();
}

void request_stop(int) { stop_requested.store(true); }

#ifdef QUANT_LIVE
// End-of-bar marker: an empty ticker. Lets the engine act on a bar as soon as it is
// complete instead of when the next bar's first tick arrives (5 minutes later live).
void push_bar_end(int64_t bar_start) {
    MarketTick marker;
    marker.timestamp = bar_start;
    marker.ticker[0] = '\0';
    push_tick(marker);
}

std::vector<std::string> split_symbols(const std::string& s) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size()) {
        const size_t comma = s.find(',', start);
        std::string sym = s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!sym.empty()) out.push_back(sym);
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return out;
}

LiveConfig make_live_config(const EngineOptions& o) {
    LiveConfig c;
    c.symbols = split_symbols(o.symbols);
    c.feed = o.feed;
    for (const char* k : {"APCA_API_KEY_ID", "ALPACA_API_KEY"}) if (!c.key.size() && std::getenv(k)) c.key = std::getenv(k);
    for (const char* k : {"APCA_API_SECRET_KEY", "ALPACA_SECRET_KEY"}) if (!c.secret.size() && std::getenv(k)) c.secret = std::getenv(k);
    if (!o.data_url.empty()) c.data_url = o.data_url;
    if (!o.trading_url.empty()) c.trading_url = o.trading_url;
    c.stream_url = o.stream_url;
    c.warmup_days = o.warmup_days;
    c.bar_grace_s = o.bar_grace_s;
    c.close_grace_s = o.close_grace_s;
    return c;
}

void live_producer(LiveConfig cfg) {
    int tick_id = 0;
    LiveFeed feed(std::move(cfg),
                  [&tick_id](const MarketTick& t) {
                      MarketTick copy = t;
                      copy.id = tick_id++;
                      push_tick(copy);
                  },
                  push_bar_end);
    feed.set_status(&live_status);
    feed.run(stop_requested);
    finish_stream();
}
#endif

void file_stream_producer(const std::string& csv_file) {
    std::ifstream file;
    if (csv_file != "-") {
        file.open(csv_file);
        if (!file.is_open()) {
            std::cerr << "[-] Error opening market data file: " << csv_file << std::endl;
            finish_stream();
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
        finish_stream();
        return;
    }

    int tick_id = 0;
    while (std::getline(input, line) && !stop_requested.load(std::memory_order_relaxed)) {
        MarketTick tick;
        if (!parse_tick_row(line, tick)) {
            rows_skipped.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        tick.id = tick_id++;
        push_tick(tick);
    }
    finish_stream();
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

    // Predictions of the last label_horizon bars, waiting for the price that settles them
    // (always scored; with online learning also learned from)
    struct Pending {
        float price = 0.0f;
        bool valid = false;
        int call = 0;         // the model's most likely class: +1 buy, -1 sell, 0 hold
        std::vector<float> z; // online learning: head inputs of every net at prediction time
    };
    std::deque<Pending> pending;
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
          window_(static_cast<size_t>(cfg.seq_len) * cfg.input_dim(), 0.0f),
          run_start_(wall_now()),
          next_status_(std::chrono::steady_clock::now() + std::chrono::seconds(std::max(opt.status_every_s, 1))) {}

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
            } else if (backoff.exhausted()) {
                // Idle (market closed, or between live bars): block instead of polling
                consumer_wake.wait([] { return !event_queue.empty() || stream_finished.load(std::memory_order_acquire); },
                                   std::chrono::milliseconds(100));
                maybe_status();
            } else {
                backoff.wait();
            }
        }
        if (!done_trading_) {
            if (bar_ts_ != INT64_MIN && !bar_closed_) close_bar();
            portfolio_.liquidate_all(last_tick_id_);
        }
        send_targets(); // flat
        save_online_state();
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

        if (online())
            std::cout << "Online Learning:       " << online_updates_ << " updates (lr " << std::defaultfloat << opt_.online_lr
                      << ", anchor " << opt_.online_anchor << std::fixed << ")" << (opt_.online_state.empty() ? "" : ", state in " + opt_.online_state) << "\n";

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
                  << "Gains / Losses:        " << usd(portfolio_.get_gains(), true) << " (" << portfolio_.get_winning_trades()
                  << " won) / " << usd(portfolio_.get_losses()) << " (" << portfolio_.get_losing_trades() << " lost)\n"
                  << "Buy/Sell Calls Right:  " << (dir_calls_ ? 100.0 * dir_right_ / dir_calls_ : 0.0) << " % of " << dir_calls_
                  << " (model's most likely class vs the price " << cfg_.label_horizon << " bars later)\n"
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

#ifdef QUANT_LIVE
    void set_broker(PaperBroker* broker) { broker_ = broker; }
#endif

private:
    static PortfolioParams make_portfolio_params(const EngineOptions& o) {
        PortfolioParams p;
        p.trade_log = o.trades_path;
        p.append_log = o.live;
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
        if (tick.ticker[0] == '\0') { // End-of-bar marker
            if (tick.timestamp == bar_ts_ && !bar_closed_) {
                close_bar();
                bar_closed_ = true;
            }
            return;
        }
        if (tick.timestamp != bar_ts_) {
            if (bar_ts_ != INT64_MIN && !bar_closed_) close_bar();
            bar_closed_ = false;
            if (tick.timestamp >= trade_end_) {
                // Past the trading window: flatten and stop the producer early
                portfolio_.liquidate_all(last_tick_id_);
                done_trading_ = true;
                stop_requested.store(true, std::memory_order_relaxed);
                return;
            }
            bar_ts_ = tick.timestamp;
            portfolio_.set_bar_time(bar_ts_);
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

        if (!trading_now()) return;
        settle_matured(st, tick.raw_price);
        TickerState::Pending entry;
        if (ready) {
            st.pipeline.copy_window(window_.data());
            if (online()) entry.z.resize(model_.z_size());
            float p[3];
            auto t2 = std::chrono::steady_clock::now();
            model_.predict(window_.data(), cfg_.seq_len, cfg_.input_dim(), p, online() ? entry.z.data() : nullptr);
            auto t3 = std::chrono::steady_clock::now();
            stats_.model_us.add(std::chrono::duration<double, std::micro>(t3 - t2).count());

            st.edge = static_cast<double>(p[2]) - static_cast<double>(p[0]);
            st.signal_bar = bar_index_;
            entry.valid = true;
            entry.call = (p[2] > p[1] && p[2] > p[0]) ? 1 : ((p[0] > p[1] && p[0] > p[2]) ? -1 : 0);
        }
        entry.price = tick.raw_price;
        st.pending.push_back(std::move(entry));
    }

    bool online() const { return opt_.online_lr > 0.0; }

    // Simulated positions as whole shares (toward zero) for the paper account
    void send_targets() {
#ifdef QUANT_LIVE
        if (!broker_) return;
        PaperBroker::Targets t;
        for (const auto& name : asset_names_) t[name] = static_cast<long>(std::trunc(portfolio_.units(name)));
        broker_->submit(std::move(t), bar_ts_);
#endif
    }

    // The prediction made label_horizon bars ago is settled by this price: score its buy/sell
    // call and, with online learning, learn from its label (same rule as the training data)
    // before predicting this bar
    void settle_matured(TickerState& st, float price) {
        if (static_cast<int>(st.pending.size()) < cfg_.label_horizon) return;
        const TickerState::Pending& old = st.pending.front();
        if (old.valid && old.price > 0.0f && price > 0.0f) {
            const double ret = static_cast<double>(price) / old.price - 1.0;
            if (old.call != 0 && ret != 0.0) {
                dir_calls_++;
                dir_right_ += (ret > 0.0) == (old.call > 0);
            }
            if (online()) {
                const int label = ret > cfg_.label_hurdle ? 2 : (ret < -cfg_.label_hurdle ? 0 : 1);
                model_.learn(old.z.data(), label, static_cast<float>(opt_.online_lr), static_cast<float>(opt_.online_anchor));
                online_updates_++;
            }
        }
        st.pending.pop_front();
    }

    bool market_closed() const {
#ifdef QUANT_LIVE
        return opt_.live && live_status.state.load() == LiveStatus::Closed;
#else
        return false;
#endif
    }

    // Called by the idle consumer (at least every 100 ms): a full summary every
    // --status-every seconds while the market is open, one at the close, and a one-line
    // heartbeat every hour while it is closed
    void maybe_status() {
        if (opt_.status_every_s <= 0) return;
        const auto now = std::chrono::steady_clock::now();
        const bool closed = market_closed();
        if (closed != closed_seen_) {
            closed_seen_ = closed;
            if (closed) {
                print_status();
                next_status_ = now + std::chrono::hours(1);
            } else {
                next_status_ = now + std::chrono::seconds(opt_.status_every_s);
            }
            return;
        }
        if (now < next_status_) return;
        if (closed) {
            print_heartbeat();
            next_status_ = now + std::chrono::hours(1);
        } else {
            print_status();
            next_status_ = now + std::chrono::seconds(opt_.status_every_s);
        }
    }

    std::string market_text() const {
#ifdef QUANT_LIVE
        if (opt_.live) {
            const int64_t open = live_status.next_open.load(), close = live_status.next_close.load();
            switch (live_status.state.load()) {
            case LiveStatus::Streaming: return "market open until " + ny_time(close, "%H:%M");
            case LiveStatus::Closed: {
                const double hours = std::max<int64_t>(0, open - wall_now()) / 3600.0;
                return "market closed, opens " + ny_time(open, "%a %Y-%m-%d %H:%M") + " (in " + num(hours, 1) + " h)";
            }
            case LiveStatus::Retrying: return "reconnecting to Alpaca";
            default: return "starting";
            }
        }
#endif
        return "backtest";
    }

    std::string allocator_text() const {
        return opt_.allocator == "qubo" ? "QUBO allocator, gamma " + num(opt_.risk_aversion, 1)
                                        : std::string("greedy allocator (= QUBO at gamma 0)");
    }

#ifdef QUANT_LIVE
    static std::string paper_line(const PaperBroker::Summary& a) {
        if (!a.days) return "no snapshots yet";
        return usd(a.equity) + ", total " + usd(a.equity - a.first_equity, true) + " (" +
               num(100.0 * (a.equity - a.first_equity) / kCapital, 2, true) + "% of $10k)";
    }
#endif

    void print_status() const {
        const PortfolioManager& pf = portfolio_;
        const double eq = pf.get_total_equity();
        const long n = pf.get_bars_recorded();
        const int trips = pf.get_total_trades();
        std::ostringstream o;
        o << "\n========== STATUS " << ny_time(wall_now(), "%Y-%m-%d %H:%M") << " New York | " << market_text() << " ==========\n"
          << "Engine (simulated $10k account, this run since " << ny_time(run_start_, "%m-%d %H:%M") << "; " << n << " bars traded"
          << (n ? ", last " + ny_time(bar_ts_, "%m-%d %H:%M") : std::string()) << ")\n"
          << "  Equity        " << usd(eq) << " | P&L " << usd(pf.get_pnl(), true) << " (" << num(pf.get_return_pct(), 2, true)
          << "%) | today " << usd(n ? eq - day_start_equity_ : 0.0, true) << "\n"
          << "  Round trips   " << trips << ": " << pf.get_winning_trades() << " won " << usd(pf.get_gains(), true) << ", "
          << pf.get_losing_trades() << " lost " << usd(pf.get_losses()) << " | win rate " << num(pf.get_win_rate(), 1) << "%";
        if (pf.get_winning_trades()) o << " | avg win " << usd(pf.get_gains() / pf.get_winning_trades(), true);
        if (pf.get_losing_trades()) o << ", avg loss " << usd(pf.get_losses() / pf.get_losing_trades());
        o << "\n";
        int open = 0;
        std::string list;
        for (const auto& name : asset_names_) {
            const Position* p = pf.position(name);
            if (!p) continue;
            open++;
            list += (list.empty() ? "" : "; ") + name + (p->units > 0 ? " long " : " short ") + num(std::abs(p->units), 1) + " @ " +
                    num(p->avg_price, 2) + " now " + num(p->current_mid_price, 2);
        }
        o << "  Open          " << open << " position(s), unrealised " << usd(pf.get_unrealized_pnl(), true)
          << (list.empty() ? "" : ": " + list) << "\n"
          << "  Risk, costs   max drawdown " << num(pf.get_max_drawdown(), 2) << "% | avg gross exposure "
          << num(pf.get_avg_gross_exposure(), 0) << "% | fees " << usd(pf.get_fees_paid()) << "\n";
        o << "  Sharpe        ";
        if (n > 1)
            o << num(pf.calculate_sharpe_ratio(), 2) << " annualised, +/- " << num(std::sqrt(kBarsPerYear / n), 2) << " after " << n
              << " bars (within 2x the +/- of 0 = indistinguishable from luck)\n";
        else
            o << "n/a until two bars have been traded\n";
        o << "  Strategy      " << allocator_text() << ", max " << opt_.max_positions << " positions, "
          << opt_.model_path.substr(opt_.model_path.find_last_of('/') + 1) << "\n"
          << "  Model         buy/sell calls right ";
        if (dir_calls_) o << num(100.0 * dir_right_ / dir_calls_, 1) << "% of " << dir_calls_ << " (vs the price " << cfg_.label_horizon * 5 << " min later)";
        else o << "n/a yet (each call is settled " << cfg_.label_horizon * 5 << " min later)";
        if (online()) o << " | online learning " << online_updates_ << " updates, lr " << num(opt_.online_lr, 4);
        o << "\n";
#ifdef QUANT_LIVE
        if (broker_) {
            const auto a = broker_->summary(kCapital);
            o << "Paper account (Alpaca, all runs in account.csv"
              << (a.days ? ", since " + ny_time(a.first_time, "%Y-%m-%d") + ", " + std::to_string(a.days) +
                               (a.days == 1 ? " trading day" : " trading days")
                         : std::string())
              << ")\n";
            if (a.days) {
                o << "  Equity        " << paper_line(a) << " | today " << usd(a.today_pnl, true) << "\n  Daily Sharpe  ";
                if (a.days < 3) o << "after 3 trading days";
                else o << num(a.daily_sharpe, 2) << ", +/- " << num(std::sqrt(252.0 / (a.days - 1)), 2) << " after " << a.days - 1 << " daily changes";
                o << " | max drawdown " << num(a.max_drawdown_pct, 2) << "% (daily closes)\n";
            }
            o << "  Orders        " << a.orders_ok << " accepted, " << a.orders_failed << " failed this run (orders.csv)\n";
        }
#endif
        o << "==========================================================================\n";
        std::cout << o.str() << std::flush;
    }

    void print_heartbeat() const {
        std::ostringstream o;
        o << "[status " << ny_time(wall_now(), "%Y-%m-%d %H:%M") << " NY] " << market_text() << " | engine " << usd(portfolio_.get_total_equity())
          << " (" << num(portfolio_.get_return_pct(), 2, true) << "%, " << portfolio_.get_total_trades() << " round trips)";
#ifdef QUANT_LIVE
        if (broker_) {
            const auto a = broker_->summary(kCapital);
            o << " | paper " << paper_line(a);
            if (a.days > 2) o << ", daily Sharpe " << num(a.daily_sharpe, 2) << " over " << a.days << " days";
        }
#endif
        std::cout << o.str() << std::endl;
    }

public:
    // Adopt a saved online-learning state if it was learned on this very model
    void resume_online_state() {
        if (opt_.online_state.empty()) return;
        std::ifstream meta(opt_.online_state + ".base");
        unsigned long long fp = 0;
        long updates = 0;
        if (!meta || !(meta >> std::hex >> fp >> std::dec >> updates)) {
            std::cout << "[+] Online learning starts fresh (no state in " << opt_.online_state << ")" << std::endl;
            return;
        }
        if (fp != model_.fingerprint() || !model_.adopt(NativeModel::load(opt_.online_state))) {
            std::cout << "[!] " << opt_.online_state << " was learned on a different model; starting fresh" << std::endl;
            return;
        }
        online_updates_ = updates;
        std::cout << "[+] Resumed online learning state (" << updates << " updates) from " << opt_.online_state << std::endl;
    }

    void save_online_state() const {
        if (!online() || opt_.online_state.empty()) return;
        try {
            model_.save(opt_.online_state);
            std::ofstream meta(opt_.online_state + ".base", std::ios::trunc);
            meta << std::hex << model_.fingerprint() << " " << std::dec << online_updates_ << "\n";
        } catch (const std::exception& e) {
            std::cerr << "[!] Could not save online state: " << e.what() << std::endl;
        }
    }

private:

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
            const int64_t day = ny_day(bar_ts_);
            if (day != day_) { // first bar of a New York day: today's P&L counts from here
                day_ = day;
                day_start_equity_ = last_equity_;
            }
            portfolio_.rebalance(last_tick_id_, targets);

            auto t1 = std::chrono::steady_clock::now();
            if (!views.empty()) stats_.alloc_us.add(std::chrono::duration<double, std::micro>(t1 - t0).count());
            portfolio_.record_equity();
            last_equity_ = portfolio_.get_total_equity();
            send_targets();
            if (opt_.live && online() && ++bars_since_save_ >= 12) { // hourly
                save_online_state();
                bars_since_save_ = 0;
            }
            if (opt_.live) {
                int open = 0;
                for (const auto& name : asset_names_) open += portfolio_.direction(name) != 0;
                char when[32];
                const std::time_t t = static_cast<std::time_t>(bar_ts_);
                std::strftime(when, sizeof(when), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
                std::cout << "[bar " << when << "] equity $" << std::fixed << std::setprecision(2)
                          << portfolio_.get_total_equity() << " (today " << usd(last_equity_ - day_start_equity_, true) << ") | "
                          << open << " open position(s) | " << portfolio_.get_total_trades() << " round trips" << std::endl;
            }

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
    bool bar_closed_ = false; // the current bar was already closed by an end-of-bar marker
    long online_updates_ = 0;
    int bars_since_save_ = 0;
    long dir_calls_ = 0, dir_right_ = 0; // settled buy/sell calls, and how many the price agreed with
    // Status summary
    static constexpr double kCapital = 10000.0; // PortfolioParams::starting_cash
    static constexpr double kBarsPerYear = 252.0 * 78.0;
    int64_t run_start_;
    std::chrono::steady_clock::time_point next_status_;
    bool closed_seen_ = false;
    int64_t day_ = INT64_MIN;
    double day_start_equity_ = kCapital;
    double last_equity_ = kCapital;
#ifdef QUANT_LIVE
    PaperBroker* broker_ = nullptr;
#endif
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

#ifndef QUANT_LIVE
    if (opt.live || opt.live_check) {
        std::cerr << "[-] This build has no live mode (needs libcurl with WebSocket support; see CMakeLists.txt)" << std::endl;
        return 1;
    }
#else
    if (opt.live || opt.live_check) {
        LiveConfig lc = make_live_config(opt);
        if (lc.key.empty() || lc.secret.empty()) {
            std::cerr << "[-] Set APCA_API_KEY_ID and APCA_API_SECRET_KEY for live mode" << std::endl;
            return 1;
        }
        if (opt.live_check) {
            LiveFeed feed(lc, [](const MarketTick&) {}, [](int64_t) {});
            return feed.check(opt.live_check, std::cout) ? 0 : 1;
        }
    }
#endif

    // Trading window: out-of-sample by default; live: bars from now on
    int64_t start = INT64_MIN, end = INT64_MAX;
    if (opt.live) {
        const int64_t now = static_cast<int64_t>(std::time(nullptr));
        start = now - now % 300;
    } else if (opt.period == "test") start = cfg.test_start_ts;
    else if (opt.period == "val") { start = cfg.val_start_ts; end = cfg.test_start_ts; }
    else if (opt.period != "all") { std::cerr << "[-] --period must be test, val or all" << std::endl; return 1; }
    if (opt.start_ts != INT64_MIN) start = opt.start_ts;
    if (opt.end_ts != INT64_MAX) end = opt.end_ts;
    if (opt.period == "all" && !opt.live) std::cout << "[!] Trading the full file: this includes the training period (in-sample)." << std::endl;

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
    std::cout << "[+] " << (opt.live ? "Live " + opt.feed + " feed for " + opt.symbols : "Streaming " + opt.data_path) << std::endl;

    Engine engine(opt, cfg, module);
    engine.set_window(start, end);
    if (opt.online_lr > 0.0) engine.resume_online_state();
#ifdef QUANT_LIVE
    std::unique_ptr<PaperBroker> broker;
    if (opt.paper_orders) {
        if (!opt.live) {
            std::cerr << "[-] --paper-orders needs --live" << std::endl;
            return 1;
        }
        const LiveConfig lc = make_live_config(opt);
        const size_t slash = opt.trades_path.find_last_of('/');
        try {
            broker = std::make_unique<PaperBroker>(lc.trading_url, lc.key, lc.secret, lc.symbols,
                                                   slash == std::string::npos ? "." : opt.trades_path.substr(0, slash));
        } catch (const std::exception& e) {
            std::cerr << "[-] " << e.what() << std::endl;
            return 1;
        }
        engine.set_broker(broker.get());
        std::cout << "[+] Paper orders on " << lc.trading_url << " for " << opt.symbols << std::endl;
    }
#endif

    auto wall_start = std::chrono::steady_clock::now();
    std::signal(SIGINT, request_stop);
    std::signal(SIGTERM, request_stop);
#ifdef QUANT_LIVE
    std::thread producer = opt.live ? std::thread(live_producer, make_live_config(opt)) : std::thread(file_stream_producer, opt.data_path);
#else
    std::thread producer(file_stream_producer, opt.data_path);
#endif
    std::thread consumer([&engine] { engine.run(); });
    // Only the latency-critical consumer gets a core of its own (the last one)
    pin_thread_to_core(consumer, static_cast<int>(std::thread::hardware_concurrency()) - 1);
    producer.join();
    consumer.join();
#ifdef QUANT_LIVE
    if (broker) broker->finish();
#endif
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();

    engine.report(wall);
    return 0;
}
