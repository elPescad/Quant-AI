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
#include <emmintrin.h>
#include <pthread.h>

#include "ring_buffer.hpp"
#include "portfolio.hpp"

struct MarketTick {
    int id;
    float spread;
    float order_imbalance;
    float price_delta_5;
    float volatility_20;
};

LockFreeRingBuffer<MarketTick, 4096> event_queue;
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
        std::string v1, v2, v3, v4;

        if (std::getline(ss, v1, ',') &&
            std::getline(ss, v2, ',') &&
            std::getline(ss, v3, ',') &&
            std::getline(ss, v4, ',')) {

            MarketTick tick;
            tick.id = tick_id++;
            tick.spread = std::stof(v1);
            tick.order_imbalance = std::stof(v2);
            tick.price_delta_5 = std::stof(v3);
            tick.volatility_20 = std::stof(v4);

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

    torch::Tensor input_tensor = torch::zeros({1, 4}, torch::kFloat32);
    float* raw_data = input_tensor.data_ptr<float>();

    while (!stream_finished || event_queue.pop().has_value()) {
        auto popped = event_queue.pop();
        
        if (popped.has_value()) [[likely]] {
            auto start = std::chrono::high_resolution_clock::now();

            MarketTick tick = popped.value();

            raw_data[0] = tick.spread;
            raw_data[1] = tick.order_imbalance;
            raw_data[2] = tick.price_delta_5;
            raw_data[3] = tick.volatility_20;

            c10::IValue output = module.forward({input_tensor});
            torch::Tensor logits = output.toTensor();

            torch::Tensor probs = torch::softmax(logits, 1);
            auto max_prob_result = probs.max(1);

            int action = std::get<1>(max_prob_result).item<int>();
            float confidence = std::get<0>(max_prob_result).item<float>();

            // Require 60% confidence before committing capital
            if (confidence < 0.60f) {
                action = 1; // HOLD
            }

            portfolio.process_signal(tick.id, action, tick.price_delta_5);

            auto end = std::chrono::high_resolution_clock::now();
            double latency = std::chrono::duration<double, std::micro>(end - start).count();
            latencies_us.push_back(latency);
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
        std::cout << "[+] TorchScript Model Loaded." << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "[-] Error loading model: " << e.what() << std::endl;
        return -1;
    }

    std::vector<double> latencies_us;
    PortfolioManager portfolio(10000.0, 0.0005);

    std::cout << "[+] Streaming market ticks from " << data_path << std::endl;
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

    if (total_ticks == 0) {
        std::cerr << "[-] No market ticks were processed." << std::endl;
        return -1;
    }

    std::sort(latencies_us.begin(), latencies_us.end());
    double total_latency_us = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
    double avg_latency = total_latency_us / total_ticks;
    double p50 = latencies_us[total_ticks * 0.50];
    double p99 = latencies_us[total_ticks * 0.99];
    double throughput = total_ticks / total_wall_time_sec;

    std::cout << "\n==================================================" << std::endl;
    std::cout << "          SYSTEM & HARDWARE BENCHMARKS            " << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Total Ticks Processed: " << total_ticks << std::endl;
    std::cout << "Avg Model Latency:     " << avg_latency << " us" << std::endl;
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