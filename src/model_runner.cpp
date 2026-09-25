#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <thread>
#include <atomic>
#include <cmath>
#include <unordered_map>
#include <cstring>
#include <emmintrin.h>
#include <pthread.h>
#include "ring_buffer.hpp"
#include "portfolio.hpp"

constexpr int SEQ_LEN = 30;

struct MarketTick {
    int id;
    char ticker[8];
    float raw_price;
    float raw_spread;
    float raw_ofi;
    float raw_delta;
    float raw_vol;
};

class EWMStandardizer {
private:
    double alpha;
    double mean = 0.0;
    double variance = 0.0;
    bool initialized = false;
    uint64_t count = 0;

public:
    EWMStandardizer(int span = 500) {
        alpha = 2.0 / (span + 1.0); 
    }

    float normalize_and_update(double x) {
        count++;
        if (!initialized) {
            mean = x;
            variance = 0.0;
            initialized = true;
            return 0.0f;
        }

        double stddev = std::sqrt(variance);
        float z = 0.0f;
        if (stddev > 1e-6 && count >= 10) {
            z = static_cast<float>(std::clamp((x - mean) / stddev, -4.0, 4.0));
        }

        double delta = x - mean;
        mean += alpha * delta;
        variance = (1.0 - alpha) * (variance + alpha * delta * delta);

        return z;
    }

    uint64_t get_count() const { return count; }
};

struct RollingWindow {
    float data[SEQ_LEN * 4] = {0.0f};
    int head = 0;
    int count = 0;

    void push(float f1, float f2, float f3, float f4) {
        int base = head * 4;
        data[base]   = f1;
        data[base+1] = f2;
        data[base+2] = f3;
        data[base+3] = f4;
        
        head = (head + 1) % SEQ_LEN;
        if (count < SEQ_LEN) count++;
    }

    void fill_tensor(float* tensor_data) const {
        int idx = (count < SEQ_LEN) ? 0 : head;
        for (int i = 0; i < count; i++) {
            int base_src = idx * 4;
            int base_dst = i * 4;
            std::memcpy(tensor_data + base_dst, data + base_src, 4 * sizeof(float));
            idx = (idx + 1) % SEQ_LEN;
        }
    }
};

LockFreeRingBuffer<MarketTick, 8192> event_queue;
std::atomic<bool> stream_finished(false);

void pin_thread_to_core(std::thread& th, int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset);
}

void file_stream_producer(const std::string& csv_file) {
    std::ifstream file(csv_file);
    if (!file.is_open()) {
        std::cerr << "[-] Error opening market data file: " << csv_file << std::endl;
        stream_finished = true;
        return;
    }

    std::string line;
    std::getline(file, line);

    int tick_id = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string sym, p, s, o, d, v;

        if (std::getline(ss, sym, ',') &&
            std::getline(ss, p, ',') &&
            std::getline(ss, s, ',') &&
            std::getline(ss, o, ',') &&
            std::getline(ss, d, ',') &&
            std::getline(ss, v, ',')) {

            MarketTick tick;
            tick.id = tick_id++;
            std::strncpy(tick.ticker, sym.c_str(), sizeof(tick.ticker) - 1);
            tick.ticker[sizeof(tick.ticker) - 1] = '\0';
            tick.raw_price = std::stof(p);
            tick.raw_spread = std::stof(s);
            tick.raw_ofi = std::stof(o);
            tick.raw_delta = std::stof(d);
            tick.raw_vol = std::stof(v);

            while (!event_queue.push(tick)) [[unlikely]] {
                _mm_pause();
            }
        }
    }
    stream_finished = true;
}

void execution_consumer(torch::jit::script::Module& module, 
                        std::vector<double>& latencies_us,
                        PortfolioManager& portfolio) {
    torch::InferenceMode guard;
    latencies_us.reserve(500000);

    std::unordered_map<std::string, EWMStandardizer> acc_spread, acc_ofi, acc_delta, acc_vol;
    std::unordered_map<std::string, RollingWindow> ticker_memory;

    torch::Tensor input_tensor = torch::zeros({1, SEQ_LEN, 4}, torch::kFloat32);
    float* raw_data = input_tensor.data_ptr<float>();

    std::unordered_map<std::string, int> last_trade_tick;
    const int COOLDOWN_TICKS = 100;

    while (true) {
        auto popped = event_queue.pop();
        
        if (popped.has_value()) [[likely]] {
            auto start = std::chrono::high_resolution_clock::now();

            MarketTick tick = popped.value();
            std::string sym(tick.ticker);

            float n_spread = acc_spread[sym].normalize_and_update(tick.raw_spread);
            float n_ofi    = acc_ofi[sym].normalize_and_update(tick.raw_ofi);
            float n_delta  = acc_delta[sym].normalize_and_update(tick.raw_delta);
            float n_vol    = acc_vol[sym].normalize_and_update(tick.raw_vol);

            ticker_memory[sym].push(n_spread, n_ofi, n_delta, n_vol);

            if (acc_spread[sym].get_count() < 30 || ticker_memory[sym].count < SEQ_LEN) {
                portfolio.process_signal(tick.id, sym, 1, tick.raw_price, tick.raw_spread);
                continue;
            }

            ticker_memory[sym].fill_tensor(raw_data);

            c10::IValue output = module.forward({input_tensor});
            torch::Tensor logits = output.toTensor();
            torch::Tensor probs = torch::softmax(logits, 1);

            const float* p_ptr = probs.data_ptr<float>();
            float prob_sell = p_ptr[0];
            float prob_hold = p_ptr[1];
            float prob_buy  = p_ptr[2];

            // 1. Confidence-Gated Model Decision
            constexpr float MIN_CONFIDENCE = 0.52f;
            int model_action = 1; // Default HOLD

            if (prob_sell >= MIN_CONFIDENCE && prob_sell > prob_hold && prob_sell > prob_buy) {
                model_action = 0; // SELL
            } else if (prob_buy >= MIN_CONFIDENCE && prob_buy > prob_hold && prob_buy > prob_sell) {
                model_action = 2; // BUY
            }

            // 2. Cooldown Execution Check
            int executed_action = 1; // Default HOLD
            if (model_action != 1) {
                if (tick.id - last_trade_tick[sym] > COOLDOWN_TICKS) {
                    executed_action = model_action;
                    last_trade_tick[sym] = tick.id;
                }
            }

            portfolio.process_signal(tick.id, sym, executed_action, tick.raw_price, tick.raw_spread);

            if (tick.id % 1000 == 0) {
                std::cout << "[DEBUG] Tick " << tick.id 
                        << " | Ticker: " << sym 
                        << " | Action: " << executed_action 
                        << " (Raw: " << model_action << ")"
                        << " | Probs: " << prob_sell << ", " 
                        << prob_hold << ", " 
                        << prob_buy << "\n";
            }

            auto end = std::chrono::high_resolution_clock::now();
            double latency = std::chrono::duration<double, std::micro>(end - start).count();
            latencies_us.push_back(latency);

        } else if (stream_finished.load()) {
            break;
        } else {
            _mm_pause();
        }
    }
}

int main() {
    std::cout << "=== Low-Latency Quant ML Paper Trading Engine ===" << std::endl;

    at::set_num_threads(1);

    std::string model_path = "../models/quant_model.pt";
    std::string data_path = "../data/market_ticks.csv";

    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path);
        module.eval();
        std::cout << "[+] TorchScript GRU Model Loaded." << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "[-] Error loading model: " << e.what() << std::endl;
        return -1;
    }

    std::vector<double> latencies_us;
    PortfolioManager portfolio(10000.0, 0.0001); 

    std::cout << "[+] Streaming raw multi-ticker market feed from " << data_path << std::endl;
    auto wall_clock_start = std::chrono::high_resolution_clock::now();

    std::thread producer(file_stream_producer, data_path);
    std::thread consumer(execution_consumer, std::ref(module), std::ref(latencies_us), std::ref(portfolio));

    pin_thread_to_core(producer, 1);
    pin_thread_to_core(consumer, 2);

    producer.join();
    consumer.join();

    auto wall_clock_end = std::chrono::high_resolution_clock::now();
    double total_wall_time_sec = std::chrono::duration<double>(wall_clock_end - wall_clock_start).count();
    size_t total_ticks = latencies_us.size();

    if (total_ticks == 0) return -1;

    std::sort(latencies_us.begin(), latencies_us.end());
    double avg_latency = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0) / total_ticks;
    double p50 = latencies_us[static_cast<size_t>(total_ticks * 0.50)];
    double p99 = latencies_us[static_cast<size_t>(total_ticks * 0.99)];
    double throughput = total_ticks / total_wall_time_sec;

    std::cout << "\n==================================================" << std::endl;
    std::cout << "          SYSTEM & HARDWARE BENCHMARKS            " << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Total Ticks Processed: " << total_ticks << std::endl;
    std::cout << "Avg GRU Model Latency: " << avg_latency << " us" << std::endl;
    std::cout << "p50 Latency:           " << p50 << " us" << std::endl;
    std::cout << "p99 Latency:           " << p99 << " us" << std::endl;
    std::cout << "Throughput:            " << static_cast<uint64_t>(throughput) << " predictions/sec" << std::endl;

    std::cout << "\n==================================================" << std::endl;
    std::cout << "        PAPER TRADING PORTFOLIO RESULTS           " << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Starting Allocation:   $" << 10000.00 << " USD" << std::endl;
    std::cout << "Ending Equity:         $" << portfolio.get_total_equity() << " USD" << std::endl;
    std::cout << "Total Net PnL:         $" << portfolio.get_pnl() << " USD (" 
              << (portfolio.get_pnl() / 10000.0) * 100.0 << "%)" << std::endl;
    std::cout << "Win Rate:              " << portfolio.get_win_rate() << " %" << std::endl;
    std::cout << "Max Drawdown:          " << portfolio.get_max_drawdown() << " %" << std::endl;
    std::cout << "Sharpe Ratio:          " << portfolio.calculate_sharpe_ratio() << std::endl;
    std::cout << "==================================================\n" << std::endl;

    return 0;
}