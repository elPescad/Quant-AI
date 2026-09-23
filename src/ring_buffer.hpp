#pragma once
#include <array>
#include <atomic>
#include <cstddef>
#include <optional>

template <typename T, size_t Capacity>
class LockFreeRingBuffer {
public:
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");

    LockFreeRingBuffer() : head_(0), tail_(0) {}

    bool push(const T& item) {
        size_t current_tail = tail_.load(std::memory_order_relaxed);
        size_t current_head = head_.load(std::memory_order_acquire);

        if ((current_tail - current_head) >= Capacity) {
            return false;
        }

        buffer_[current_tail & Mask] = item;
        tail_.store(current_tail + 1, std::memory_order_release);
        return true;
    }

    std::optional<T> pop() {
        size_t current_head = head_.load(std::memory_order_relaxed);
        size_t current_tail = tail_.load(std::memory_order_acquire);

        if (current_head == current_tail) {
            return std::nullopt;
        }

        T item = buffer_[current_head & Mask];
        head_.store(current_head + 1, std::memory_order_release);
        return item;
    }

private:
    static constexpr size_t Mask = Capacity - 1;
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
    std::array<T, Capacity> buffer_;
};