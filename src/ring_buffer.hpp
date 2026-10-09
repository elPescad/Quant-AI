#pragma once
#include <array>
#include <atomic>
#include <cstddef>
#include <optional>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
inline void cpu_relax() { _mm_pause(); }
#elif defined(__aarch64__)
inline void cpu_relax() { asm volatile("yield"); }
#else
inline void cpu_relax() {}
#endif

#include <chrono>
#include <thread>

// Wait strategy for an empty/full queue: spin briefly (lowest latency while data is
// flowing), then yield, then sleep. Busy-spinning forever would burn a whole vCPU even
// when the market is closed, which on a small shared-core VM also eats the CPU budget.
class Backoff {
public:
    void wait() {
        if (n_ < kSpins) {
            cpu_relax();
        } else if (n_ < kSpins + kYields) {
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            return;
        }
        n_++;
    }
    void reset() { n_ = 0; }

private:
    static constexpr int kSpins = 2000;
    static constexpr int kYields = 100;
    int n_ = 0;
};

// Wait-free single-producer / single-consumer ring buffer.
// Exactly one thread may call push() and exactly one other thread may call pop().
template <typename T, size_t Capacity>
class LockFreeRingBuffer {
public:
    // Capacity must be a power of 2 for the bitwise mask
    static_assert(Capacity >= 2 && (Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");

    LockFreeRingBuffer() = default;
    LockFreeRingBuffer(const LockFreeRingBuffer&) = delete;
    LockFreeRingBuffer& operator=(const LockFreeRingBuffer&) = delete;

    bool push(const T& item) {
        const size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail - head_cache_ >= Capacity) {
            // Looks full: refresh our view of the consumer's position
            head_cache_ = head_.load(std::memory_order_acquire);
            if (tail - head_cache_ >= Capacity) return false;
        }
        buffer_[tail & Mask] = item;
        tail_.store(tail + 1, std::memory_order_release);
        return true;
    }

    bool try_pop(T& out) {
        const size_t head = head_.load(std::memory_order_relaxed);
        if (head == tail_cache_) {
            // Looks empty: refresh our view of the producer's position
            tail_cache_ = tail_.load(std::memory_order_acquire);
            if (head == tail_cache_) return false;
        }
        out = buffer_[head & Mask];
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

    std::optional<T> pop() {
        T item;
        if (!try_pop(item)) return std::nullopt;
        return item;
    }

private:
    static constexpr size_t Mask = Capacity - 1;

    // Each index lives on its own cache line next to the cached copy of the other
    // side's index that only its owning thread touches, so the hot path does not
    // bounce cache lines between cores.
    alignas(64) std::atomic<size_t> head_{0}; // written by consumer
    size_t tail_cache_ = 0;                   // consumer-local
    alignas(64) std::atomic<size_t> tail_{0}; // written by producer
    size_t head_cache_ = 0;                   // producer-local
    alignas(64) std::array<T, Capacity> buffer_{};
};
