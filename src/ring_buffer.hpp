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
#include <condition_variable>
#include <mutex>
#include <thread>

// Wait strategy for an empty/full queue: spin briefly (lowest latency while data is
// flowing), then yield, then sleep. Busy-spinning forever would burn a whole vCPU even
// when the market is closed, which on a small shared-core VM also eats the CPU budget.
// The sleep doubles up to 2 ms. A consumer that is idle for longer should block on a
// WakeSignal instead (see exhausted()).
class Backoff {
public:
    void wait() {
        if (n_ < kSpins) {
            cpu_relax();
        } else if (n_ < kSpins + kYields) {
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us_));
            sleep_us_ = sleep_us_ * 2 > kMaxSleepUs ? kMaxSleepUs : sleep_us_ * 2;
            return;
        }
        n_++;
    }
    void reset() {
        n_ = 0;
        sleep_us_ = kMinSleepUs;
    }
    bool exhausted() const { return n_ >= kSpins + kYields; }

private:
    static constexpr int kSpins = 2000;
    static constexpr int kYields = 100;
    static constexpr int kMinSleepUs = 50;
    static constexpr int kMaxSleepUs = 2000;
    int n_ = 0;
    int sleep_us_ = kMinSleepUs;
};

// Lets an idle consumer block until the producer has something for it, instead of polling.
// The flag + fences make a lost wake-up impossible (either the consumer sees the data, or
// the producer sees the consumer asleep and notifies); the timeout is only a backstop.
class WakeSignal {
public:
    // Producer: call after making data (or end of stream) visible
    void notify() {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (sleeping_.load(std::memory_order_relaxed)) {
            std::lock_guard<std::mutex> lock(mu_);
            cv_.notify_one();
        }
    }

    // Consumer: block until notified or `timeout`, unless ready() is already true
    template <class Ready>
    void wait(Ready ready, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mu_);
        sleeping_.store(true, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (!ready()) cv_.wait_for(lock, timeout);
        sleeping_.store(false, std::memory_order_relaxed);
    }

private:
    std::mutex mu_;
    std::condition_variable cv_;
    std::atomic<bool> sleeping_{false};
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

    // Consumer side
    bool empty() const { return head_.load(std::memory_order_relaxed) == tail_.load(std::memory_order_acquire); }

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
