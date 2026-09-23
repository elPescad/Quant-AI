#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <torch/script.h>
#include "ring_buffer.hpp"

// C++ memory layout matching our order book feature vector
struct MarketTick {
    float spread;
    float order_imbalance;
    float price_delta_5;
    float volatility_20;
};

int main() {
    std::cout << "=== Low-Latency Quant ML Inference Engine ===" << std::endl;

    // 1. Load serialized PyTorch model graph into C++
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("../models/quant_model.pt");
        module.eval(); // Set model to evaluation mode (disables dropout/batchnorm updates)
        std::cout << "[+] PyTorch TorchScript Model loaded successfully." << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "[-] Error loading model: " << e.what() << std::endl;
        return -1;
    }

    // 2. Initialize Lock-Free Event Ring Buffer (Capacity: 1024)
    LockFreeRingBuffer<MarketTick, 1024> event_queue;

    // 3. Benchmarking setup
    const int num_iterations = 10000;
    std::vector<double> latencies_us;
    latencies_us.reserve(num_iterations);

    std::cout << "[+] Executing " << num_iterations << " real-time inference ticks..." << std::endl;

    // Disables autograd gradient tracking inside C++ to maximize inference speed
    torch::NoGradGuard no_grad;

    for (int i = 0; i < num_iterations; ++i) {
        // Simulated incoming live market tick
        MarketTick tick{0.02f, 0.15f, 0.05f, 0.012f};

        auto start = std::chrono::high_resolution_clock::now();

        // Pass raw C++ floats directly into a 1x4 LibTorch Tensor
        torch::Tensor input_tensor = torch::tensor({
            {tick.spread, tick.order_imbalance, tick.price_delta_5, tick.volatility_20}
        }, torch::kFloat32);

        // Run forward pass through C++ LibTorch model
        c10::IValue output = module.forward({input_tensor});
        torch::Tensor logits = output.toTensor();
        
        // Extract predicted class index (0 = Sell, 1 = Hold, 2 = Buy)
        int action = logits.argmax(1).item<int>();

        auto end = std::chrono::high_resolution_clock::now();
        double latency = std::chrono::duration<double, std::micro>(end - start).count();
        latencies_us.push_back(latency);
    }

    // 4. Calculate Performance Metrics
    std::sort(latencies_us.begin(), latencies_us.end());
    double total_time_us = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
    double avg_latency = total_time_us / num_iterations;
    double p50 = latencies_us[num_iterations * 0.50];
    double p99 = latencies_us[num_iterations * 0.99];
    double throughput = (num_iterations / (total_time_us / 1000000.0));

    std::cout << "\n=== Performance Benchmarks ===" << std::endl;
    std::cout << "Avg Latency:  " << avg_latency << " us" << std::endl;
    std::cout << "p50 Latency:  " << p50 << " us" << std::endl;
    std::cout << "p99 Latency:  " << p99 << " us" << std::endl;
    std::cout << "Throughput:   " << static_cast<uint64_t>(throughput) << " predictions/sec" << std::endl;

    return 0;
}