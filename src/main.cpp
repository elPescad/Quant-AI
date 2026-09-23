#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <thread>
#include <atomic>
#include <emmintrin.h>
#include <pthread.h>
#include <torch/torch.h>
#include <torch/script.h>
#include "ring_buffer.hpp"

struct MarketTick {
    float spread;
    float order_imbalance;
    float price_delta_5;
    float volatility_20;
};

LockFreeRingBuffer<MarketTick, 4096> event_queue;
const int TOTAL_TICKS = 100000;

void pin_thread(std::thread& th, int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset);
}

void producer_thread_func() {
    for (int i = 0; i < TOTAL_TICKS; ++i) {
        MarketTick tick{0.02f, 0.15f, 0.05f, 0.012f};
        
        // Low-latency busy-spin using x86 pause
        while (!event_queue.push(tick)) [[unlikely]] {
            _mm_pause();
        }
    }
}

void consumer_thread_func(torch::jit::script::Module& module, std::vector<double>& latencies_us) {
    torch::InferenceMode guard;
    latencies_us.reserve(TOTAL_TICKS);

    torch::Tensor input_tensor = torch::zeros({1, 4}, torch::kFloat32);
    float* raw_data = input_tensor.data_ptr<float>();

    int ticks_processed = 0;

    while (ticks_processed < TOTAL_TICKS) {
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
            int action = logits.argmax(1).item<int>();

            auto end = std::chrono::high_resolution_clock::now();
            double latency = std::chrono::duration<double, std::micro>(end - start).count();
            latencies_us.push_back(latency);

            ticks_processed++;
        } else {
            _mm_pause(); // Busy spin if buffer empty
        }
    }
}

int main() {
    std::cout << "=== Low-Latency Quant Engine (Hardware & Thread Optimized) ===" << std::endl;

    at::set_num_threads(1);

    torch::jit::script::Module module;
    try {
        module = torch::jit::load("../models/quant_model.pt");
        module.eval();
        std::cout << "[+] TorchScript Model Loaded." << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "[-] Error loading model: " << e.what() << std::endl;
        return -1;
    }

    std::vector<double> latencies_us;

    std::cout << "[+] Spawning pinned threads (Producer: Core 1, Consumer: Core 2)..." << std::endl;
    auto wall_clock_start = std::chrono::high_resolution_clock::now();

    std::thread producer(producer_thread_func);
    std::thread consumer(consumer_thread_func, std::ref(module), std::ref(latencies_us));

    // Pin producer to core 1, consumer to core 2
    pin_thread(producer, 1);
    pin_thread(consumer, 2);

    producer.join();
    consumer.join();

    auto wall_clock_end = std::chrono::high_resolution_clock::now();
    double total_wall_time_sec = std::chrono::duration<double>(wall_clock_end - wall_clock_start).count();

    std::sort(latencies_us.begin(), latencies_us.end());
    double total_latency_us = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
    double avg_latency = total_latency_us / TOTAL_TICKS;
    double p50 = latencies_us[TOTAL_TICKS * 0.50];
    double p99 = latencies_us[TOTAL_TICKS * 0.99];
    double throughput = TOTAL_TICKS / total_wall_time_sec;

    std::cout << "\n=== Hardware Benchmarks ===" << std::endl;
    std::cout << "Total Ticks Processed: " << TOTAL_TICKS << std::endl;
    std::cout << "Avg Model Latency:     " << avg_latency << " us" << std::endl;
    std::cout << "p50 Latency:           " << p50 << " us" << std::endl;
    std::cout << "p99 Latency:           " << p99 << " us" << std::endl;
    std::cout << "System Throughput:     " << static_cast<uint64_t>(throughput) << " predictions/sec" << std::endl;

    return 0;
}