// Unit tests for the engine, including native GRU inference parity with PyTorch.
// Run from the repository root (or pass the fixture directory as argv[1]).

#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "ar_estimator.hpp"
#include "feature_pipeline.hpp"
#include "market_data.hpp"
#include "portfolio.hpp"
#include "quantum_sim.hpp"
#include "qubo.hpp"
#include "qubo_allocator.hpp"
#include "ring_buffer.hpp"
#include "native_model.hpp"

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, msg)                                                         \
    do {                                                                         \
        g_checks++;                                                              \
        if (!(cond)) {                                                           \
            g_failures++;                                                        \
            std::cerr << "  [FAIL] " << msg << "  (" << #cond << ", line " << __LINE__ << ")\n"; \
        }                                                                        \
    } while (0)

static bool near(double a, double b, double tol) { return std::abs(a - b) <= tol; }

// ------------------------------------------------------------------------------------
// Quantum simulator
// ------------------------------------------------------------------------------------
static void test_quantum_gates() {
    std::cout << "[quantum] gate algebra\n";
    using quantum::StateVector;

    StateVector sv(1);
    sv.h(0);
    CHECK(near(sv.amplitudes()[0].real(), 1 / std::sqrt(2.0), 1e-12) && near(sv.amplitudes()[1].real(), 1 / std::sqrt(2.0), 1e-12),
          "H|0> = |+>");
    sv.h(0);
    CHECK(near(std::norm(sv.amplitudes()[0]), 1.0, 1e-12), "HH = I");

    // Bell state (|00> + |11>)/sqrt2
    StateVector bell(2);
    bell.h(0);
    bell.cnot(0, 1);
    std::vector<double> p;
    bell.probabilities(p);
    CHECK(near(p[0], 0.5, 1e-12) && near(p[3], 0.5, 1e-12) && near(p[1] + p[2], 0.0, 1e-12), "H+CNOT makes a Bell state");

    // <Z0 Z1> = +1 for the Bell state, <Z0> = 0
    quantum::walsh_hadamard(p);
    CHECK(near(p[0b11], 1.0, 1e-12) && near(p[0b01], 0.0, 1e-12), "Bell state correlations <ZZ>=1, <Z>=0");

    // CNOT . RZ . CNOT == exp(-i theta/2 Z_a Z_b) (a diagonal phase)
    std::mt19937_64 rng(1);
    std::normal_distribution<double> g(0, 1);
    StateVector a(3), b(3);
    for (size_t i = 0; i < a.dim(); ++i) a.amplitudes()[i] = b.amplitudes()[i] = {g(rng), g(rng)};
    const double theta = 0.7;
    a.rzz(0, 2, theta);
    double max_err = 0.0;
    for (size_t i = 0; i < b.dim(); ++i) {
        const double z0 = (i & 1) ? -1 : 1, z2 = (i & 4) ? -1 : 1;
        const auto expected = b.amplitudes()[i] * std::polar(1.0, -0.5 * theta * z0 * z2);
        max_err = std::max(max_err, std::abs(expected - a.amplitudes()[i]));
    }
    CHECK(max_err < 1e-12, "RZZ gate decomposition equals exp(-i theta/2 ZZ)");

    // Unitarity: a long random circuit preserves the norm
    StateVector big(5);
    std::uniform_int_distribution<int> q(0, 4);
    for (int k = 0; k < 2000; ++k) {
        int i = q(rng), j = q(rng);
        big.h(i);
        big.rz(j, g(rng));
        big.ry(i, g(rng));
        if (i != j) big.cnot(i, j);
    }
    CHECK(near(big.norm_sq(), 1.0, 1e-9), "random 5-qubit circuit preserves norm");

    // Feature map outputs are expectation values, so they lie in [-1, 1]
    quantum::ZZFeatureMap fm(5, 2, 0.5);
    std::vector<double> out(fm.output_dim());
    bool in_range = true;
    for (int k = 0; k < 200; ++k) {
        double x[5];
        for (double& v : x) v = 3 * g(rng);
        fm.lift(x, out.data());
        for (double v : out) in_range &= (v >= -1 - 1e-9 && v <= 1 + 1e-9);
    }
    CHECK(fm.output_dim() == 62, "5-qubit lift is 62-dimensional");
    CHECK(in_range, "Pauli expectations within [-1, 1]");
}

static std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) out.push_back(cell);
    return out;
}

static void test_feature_parity(const std::string& fixture_dir) {
    std::cout << "[parity] C++ feature pipeline vs python/quantum_features.py\n";
    std::ifstream in(fixture_dir + "/feature_parity.csv");
    if (!in.is_open()) {
        std::cout << "  [SKIP] fixture not found in " << fixture_dir << " (run python/quantum_features.py)\n";
        return;
    }
    FeatureConfig cfg = FeatureConfig::load(fixture_dir + "/feature_parity_config.txt");
    std::string line;
    std::getline(in, line);
    CHECK(is_tick_csv_header(line), "fixture header");

    std::map<std::string, std::unique_ptr<TickerFeaturePipeline>> pipes;
    double max_err = 0.0;
    long rows = 0;
    while (std::getline(in, line)) {
        MarketTick t;
        if (!parse_tick_row(line, t)) { CHECK(false, "fixture row parses"); continue; }
        auto& pipe = pipes[t.ticker];
        if (!pipe) pipe = std::make_unique<TickerFeaturePipeline>(cfg);
        pipe->update(t);
        auto cells = split_csv(line);
        const auto& row = pipe->latest_row();
        CHECK(static_cast<int>(cells.size()) == 8 + cfg.input_dim(), "fixture width");
        for (int k = 0; k < cfg.input_dim(); ++k) max_err = std::max(max_err, std::abs(row[k] - std::stod(cells[8 + k])));
        rows++;
    }
    std::cout << "  rows " << rows << ", max |C++ - Python| = " << max_err << "\n";
    CHECK(rows > 0 && max_err < 1e-5, "C++ and Python features agree");
}

// ------------------------------------------------------------------------------------
// AR(1) estimation
// ------------------------------------------------------------------------------------

// The previous estimator (src/feature_extractor.hpp, removed), kept here for comparison
struct LegacyEwmaAr {
    double alpha = 0.001, mean = 0, var = 1, cov = 0, prev = 0;
    bool init = false;
    double update(double x) {
        if (!init) { prev = x; mean = x; init = true; return 0; }
        double diff = x - mean;
        mean += alpha * diff;
        var = (1 - alpha) * (var + alpha * diff * diff);
        double prev_diff = prev - mean;
        cov = (1 - alpha) * cov + alpha * (diff * prev_diff);
        prev = x;
        return std::clamp(var > 1e-6 ? cov / var : 0.0, -0.95, 0.95);
    }
};

static void test_ar_estimator() {
    std::cout << "[ar] RLS AR(1) estimator\n";
    std::mt19937_64 rng(42);
    std::normal_distribution<double> g(0, 1);

    // Stationary AR(1) with a non-zero mean: x = 2 + 0.85 x_{t-1} + e
    {
        ArRlsEstimator rls(0.999);
        LegacyEwmaAr legacy;
        double x = 2.0 / (1 - 0.85), legacy_phi = 0;
        double se_rls = 0, se_old = 0;
        long n = 0;
        for (int t = 0; t < 20000; ++t) {
            x = 2.0 + 0.85 * x + g(rng);
            rls.update(x);
            legacy_phi = legacy.update(x);
            if (t >= 500) {
                se_rls += std::pow(rls.phi() - 0.85, 2);
                se_old += std::pow(legacy_phi - 0.85, 2);
                n++;
            }
        }
        const double rmse_rls = std::sqrt(se_rls / n), rmse_old = std::sqrt(se_old / n);
        std::cout << "  stationary phi=0.85, mean 13.3: RLS phi " << rls.phi() << " +/- " << rls.phi_stderr()
                  << " | RMSE RLS " << rmse_rls << " vs legacy EWMA " << rmse_old << "\n";
        CHECK(std::abs(rls.phi() - 0.85) < 4 * rls.phi_stderr() + 1e-3, "RLS phi within 4 standard errors");
        CHECK(near(rls.intercept(), 2.0, 0.5), "RLS intercept recovered");
        CHECK(rmse_rls < rmse_old, "RLS more accurate than the legacy EWMA ratio");
        CHECK(rls.phi_stderr() > 0.0 && rls.phi_stderr() < 0.05, "standard error is sensible");
    }

    // Regime switch 0.9 -> 0.2: how fast does each estimator track?
    {
        ArRlsEstimator rls(0.995);
        LegacyEwmaAr legacy;
        double x = 0, se_rls = 0, se_old = 0;
        long n = 0;
        for (int t = 0; t < 6000; ++t) {
            const double phi = (t / 1500) % 2 == 0 ? 0.9 : 0.2;
            x = phi * x + g(rng);
            rls.update(x);
            const double lp = legacy.update(x);
            if (t % 1500 >= 300) {
                se_rls += std::pow(rls.phi() - phi, 2);
                se_old += std::pow(lp - phi, 2);
                n++;
            }
        }
        const double rmse_rls = std::sqrt(se_rls / n), rmse_old = std::sqrt(se_old / n);
        std::cout << "  regime switching 0.9 <-> 0.2: tracking RMSE RLS " << rmse_rls << " vs legacy EWMA " << rmse_old << "\n";
        CHECK(rmse_rls < 0.15, "RLS tracks regime switches");
        CHECK(rmse_rls < rmse_old, "RLS tracks regimes better than legacy");
    }

    // Degenerate input must not blow up
    {
        ArRlsEstimator rls;
        for (int t = 0; t < 5000; ++t) rls.update(1.0);
        CHECK(std::isfinite(rls.phi()) && std::isfinite(rls.phi_stderr()), "constant series stays finite");
    }
}

// ------------------------------------------------------------------------------------
// QUBO / simulated annealing
// ------------------------------------------------------------------------------------
static void test_qubo() {
    std::cout << "[qubo] simulated annealing vs exhaustive search\n";
    std::mt19937_64 rng(3);
    std::normal_distribution<double> g(0, 1);

    int matched = 0;
    const int trials = 40;
    for (int k = 0; k < trials; ++k) {
        const int n = 12 + k % 7; // 12..18 variables
        qubo::Qubo q(n);
        for (int i = 0; i < n; ++i) {
            q.add_linear(i, g(rng));
            for (int j = i + 1; j < n; ++j) q.add_quadratic(i, j, g(rng));
        }
        qubo::AnnealParams p;
        p.seed = 100 + k;
        qubo::SimulatedAnnealer sa(p);
        auto s = sa.solve(q);
        auto e = qubo::brute_force(q);
        CHECK(near(s.energy, q.energy(s.x), 1e-9), "SA reports the true energy of its solution");
        if (s.energy <= e.energy + 1e-9) matched++;
    }
    std::cout << "  random dense QUBOs (12-18 vars): SA found the global optimum in " << matched << "/" << trials << "\n";
    CHECK(matched >= trials * 0.95, "SA reaches the global optimum on >= 95% of instances");
}

static void test_allocator() {
    std::cout << "[allocator] portfolio QUBO constraints and greedy equivalence\n";
    std::mt19937_64 rng(11);
    std::normal_distribution<double> g(0, 1);
    std::uniform_int_distribution<int> cur(-1, 1);

    const int n_assets = 6;
    RiskModel risk(0.97);
    for (int i = 0; i < n_assets; ++i) risk.add_asset();
    // Two correlated clusters
    for (int t = 0; t < 500; ++t) {
        double m1 = g(rng), m2 = g(rng);
        std::vector<double> r(n_assets);
        for (int i = 0; i < n_assets; ++i) r[i] = 0.001 * ((i < 3 ? m1 : m2) * 0.9 + 0.3 * g(rng));
        risk.update(r, std::vector<uint8_t>(n_assets, 1));
    }

    int ok_constraints = 0, ok_optimal = 0, ok_greedy = 0, diversified = 0;
    const int trials = 60;
    for (int k = 0; k < trials; ++k) {
        std::vector<AssetView> views(n_assets);
        for (int i = 0; i < n_assets; ++i) {
            views[i].asset = i;
            views[i].expected_return = 0.002 * g(rng);
            views[i].cost = 0.00015;
            views[i].current = cur(rng);
        }
        const int K = 3;

        AllocatorParams ap;
        ap.max_positions = K;
        ap.risk_aversion = 400.0;
        ap.position_weight = 1.0 / K;
        ap.verify_with_brute_force = true;
        ap.anneal.seed = 900 + k;
        QuboAllocator alloc(ap, risk);
        auto d = alloc.allocate(views, K);

        int held = 0, cluster1 = 0, cluster2 = 0;
        for (int i = 0; i < n_assets; ++i) {
            held += d[i] != 0;
            (i < 3 ? cluster1 : cluster2) += std::abs(d[i]);
        }
        ok_constraints += held <= K;
        ok_optimal += alloc.stats().matched_optimum == 1;
        diversified += (cluster1 > 0 && cluster2 > 0) || held <= 1;

        // With no risk term the QUBO optimum is exactly the greedy top-K
        AllocatorParams ap0 = ap;
        ap0.risk_aversion = 0.0;
        QuboAllocator alloc0(ap0, risk);
        ok_greedy += alloc0.allocate(views, K) == QuboAllocator::greedy(views, K);
    }
    std::cout << "  constraints held " << ok_constraints << "/" << trials << ", SA optimal " << ok_optimal << "/" << trials
              << ", gamma=0 == greedy " << ok_greedy << "/" << trials << ", diversified across clusters " << diversified << "/"
              << trials << "\n";
    CHECK(ok_constraints == trials, "never more than K positions");
    CHECK(ok_optimal >= trials * 0.95, "SA matches brute force on portfolio QUBOs");
    CHECK(ok_greedy >= trials * 0.95, "QUBO with gamma=0 reproduces greedy");
}

// ------------------------------------------------------------------------------------
// Ring buffer
// ------------------------------------------------------------------------------------
static void test_ring_buffer() {
    std::cout << "[ring] SPSC stress test\n";
    static LockFreeRingBuffer<uint64_t, 64> q; // Small capacity forces wrap-around and full/empty races
    const uint64_t N = 5'000'000;
    std::atomic<bool> done{false};
    std::thread producer([&] {
        for (uint64_t i = 0; i < N; ++i) while (!q.push(i)) cpu_relax();
        done.store(true, std::memory_order_release);
    });
    uint64_t expected = 0, out_of_order = 0;
    uint64_t v;
    while (true) {
        const bool finished = done.load(std::memory_order_acquire);
        if (q.try_pop(v)) {
            if (v != expected) out_of_order++;
            expected = v + 1;
        } else if (finished) {
            break;
        }
    }
    producer.join();
    std::cout << "  received " << expected << "/" << N << " in order (" << out_of_order << " out of order)\n";
    CHECK(expected == N && out_of_order == 0, "every item delivered exactly once, in order");
}

// ------------------------------------------------------------------------------------
// Portfolio accounting
// ------------------------------------------------------------------------------------
static void test_portfolio() {
    std::cout << "[portfolio] accounting\n";
    PortfolioParams p;
    p.trade_log = "";
    p.fee_rate = 0.0001;
    p.position_weight = 0.25;
    PortfolioManager pm(p);

    pm.mark(0, "AAA", 100.0f, 0.02f);
    pm.mark(0, "BBB", 50.0f, 0.01f);
    pm.rebalance(0, {{"AAA", 1}, {"BBB", -1}});
    CHECK(pm.direction("AAA") == 1 && pm.direction("BBB") == -1, "opened long and short");
    CHECK(near(pm.get_gross_exposure(), 5000.0, 5.0), "each position is 25% of equity");
    // Round trip at unchanged prices loses only fees + spread
    pm.rebalance(1, {{"AAA", 0}, {"BBB", 0}});
    const double expected_cost = 2 * 2500 * (2 * 0.0001 + 0.0001 + 0.0001); // fee both ways + half spread both ways
    CHECK(near(pm.get_pnl(), -expected_cost, 0.6), "flat round trip costs fees + spread only");
    CHECK(near(pm.get_gross_exposure(), 0.0, 1e-9), "flat after closing");

    // Short proceeds are not buying power
    PortfolioParams p2 = p;
    p2.position_weight = 0.6;
    PortfolioManager pm2(p2);
    for (const char* t : {"A", "B", "C"}) pm2.mark(0, t, 10.0f, 0.0f);
    pm2.rebalance(0, {{"A", -1}, {"B", 1}, {"C", 1}});
    CHECK(pm2.get_gross_exposure() <= 1.0 * pm2.get_total_equity() + 1e-6, "gross leverage cap respected");

    CHECK(!pm.mark(2, "AAA", -5.0f, 0.01f) && !pm.mark(2, "AAA", NAN, 0.01f), "invalid prices rejected");

    // Stop loss
    PortfolioManager pm3(p);
    pm3.mark(0, "X", 100.0f, 0.0f);
    pm3.rebalance(0, {{"X", 1}});
    CHECK(pm3.mark(1, "X", 98.5f, 0.0f) && pm3.direction("X") == 0, "hard stop closes a losing long");

    // Costs: a measured quote overrides the capped high-low proxy
    PortfolioManager pm4(p);
    pm4.mark(0, "Q", 500.0f, 0.60f, 0.01f);  // high-low 12 bp, quoted 1 cent = 0.2 bp
    CHECK(near(pm4.one_way_cost("Q"), 0.5 * 0.01 / 500.0 + 0.0001, 1e-12), "quoted spread sets the cost");
    pm4.mark(1, "Q", 500.0f, 0.60f);         // no quote on this tick
    CHECK(near(pm4.one_way_cost("Q"), 0.5 * p.max_spread_cost_pct + 0.0001, 1e-12), "without a quote the capped proxy is used");
    pm4.mark(2, "Q", 500.0f, 0.60f, 10.0f);   // 2% quoted spread: bad data
    pm4.rebalance(2, {{"Q", 1}});
    CHECK(pm4.direction("Q") == 0, "no entry when the quoted spread is implausibly wide");
}

static void test_tick_parsing() {
    std::cout << "[market data] CSV rows with and without quoted_spread\n";
    MarketTick t;
    CHECK(parse_tick_row("1700000000,SPY,650.5,0.6,1.2,0.1,0.1,2", t) && t.quoted_spread < 0 && t.target == 2,
          "8-column row parses, spread unknown");
    CHECK(parse_tick_row("1700000000,SPY,650.5,0.6,1.2,0.1,0.1,2,0.01", t) && near(t.quoted_spread, 0.01, 1e-7) && t.target == 2,
          "9-column row carries the quoted spread");
    CHECK(parse_tick_row("1700000000,SPY,650.5,0.6,1.2,0.1,0.1,-1,-1", t) && t.quoted_spread < 0 && t.target == -1,
          "-1 quoted spread means unknown");
    CHECK(is_tick_csv_header("timestamp,ticker,raw_price,raw_spread,raw_ofi,raw_delta,raw_vol,target,quoted_spread"),
          "header with quoted_spread accepted");
}

// Native C++ GRU vs PyTorch on random-weight models in every combine mode
// (fixture from python/native_model.py --fixture tests/fixtures)
static void test_native_gru(const std::string& fixture_dir) {
    std::cout << "[native gru] parity with PyTorch\n";
    std::ifstream in(fixture_dir + "/gru_parity_inputs.bin", std::ios::binary);
    if (!in) {
        std::cout << "  [SKIP] fixture not found in " << fixture_dir << " (run python/native_model.py --fixture)\n";
        return;
    }
    int32_t dims[3];
    in.read(reinterpret_cast<char*>(dims), sizeof(dims));
    const int cases = dims[0], seq_len = dims[1], input_dim = dims[2];
    std::vector<float> x(static_cast<size_t>(cases) * seq_len * input_dim);
    in.read(reinterpret_cast<char*>(x.data()), static_cast<std::streamsize>(x.size() * sizeof(float)));
    CHECK(static_cast<bool>(in), "parity inputs readable");

    std::map<std::string, std::vector<std::array<double, 3>>> expected;
    std::ifstream csv(fixture_dir + "/gru_parity_expected.csv");
    std::string line;
    std::getline(csv, line);
    while (std::getline(csv, line)) {
        auto c = split_csv(line);
        expected[c[0]].push_back({std::stod(c[2]), std::stod(c[3]), std::stod(c[4])});
    }
    for (const char* name : {"single", "mean", "veto", "veto_short"}) {
        NativeModel m = NativeModel::load(fixture_dir + "/gru_parity_" + name + ".weights");
        CHECK(m.input_dim() == input_dim, std::string(name) + ": input_dim read from file");
        double max_err = 0.0;
        int vetoed = 0;
        for (int i = 0; i < cases; ++i) {
            float p[3];
            m.predict(x.data() + static_cast<size_t>(i) * seq_len * input_dim, seq_len, input_dim, p);
            for (int k = 0; k < 3; ++k) max_err = std::max(max_err, std::abs(p[k] - expected[name][i][k]));
            vetoed += p[1] == 1.0f;
        }
        CHECK(static_cast<int>(expected[name].size()) == cases && max_err < 1e-5,
              std::string(name) + ": C++ probabilities match PyTorch (max err " + std::to_string(max_err) + ")");
        std::cout << "    " << name << ": max |p_cpp - p_torch| " << max_err;
        if (std::string(name).rfind("veto", 0) == 0) std::cout << ", " << vetoed << "/" << cases << " vetoed";
        std::cout << "\n";
    }
    bool threw = false;
    try { NativeModel::load(fixture_dir + "/feature_parity_config.txt"); } catch (const std::exception&) { threw = true; }
    CHECK(threw, "non-model file is rejected");
}

int main(int argc, char** argv) {
    const std::string fixtures = argc > 1 ? argv[1] : "tests/fixtures";
    test_quantum_gates();
    test_feature_parity(fixtures);
    test_ar_estimator();
    test_qubo();
    test_allocator();
    test_ring_buffer();
    test_native_gru(fixtures);
    test_portfolio();
    test_tick_parsing();
    std::cout << "\n" << (g_checks - g_failures) << "/" << g_checks << " checks passed\n";
    return g_failures == 0 ? 0 : 1;
}
