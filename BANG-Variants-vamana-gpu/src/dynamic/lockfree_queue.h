#ifndef LOCKFREE_QUEUE_H
#define LOCKFREE_QUEUE_H

#include <atomic>
#include <vector>
#include <cstddef>

/**
 * Lock-free bounded MPMC queue
 *
 * Multiple producers and consumers can access simultaneously
 * without locks, using only atomic operations.
 */
template<typename T>
class LockFreeQueue {
public:
    explicit LockFreeQueue(size_t capacity)
        : capacity_(capacity), buffer_(capacity), head_(0), tail_(0) {

        // Initialize sequence numbers
        for (size_t i = 0; i < capacity; i++) {
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
        }
    }

    /**
     * Try to enqueue an item
     * @return true if successful, false if queue is full
     */
    bool tryPush(const T& item) {
        Cell* cell;
        size_t pos = tail_.load(std::memory_order_relaxed);

        for (;;) {
            cell = &buffer_[pos % capacity_];
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t diff = (intptr_t)seq - (intptr_t)pos;

            if (diff == 0) {
                // Cell is ready for writing
                if (tail_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                // Queue is full
                return false;
            } else {
                // Another thread beat us, retry
                pos = tail_.load(std::memory_order_relaxed);
            }
        }

        cell->data = item;
        cell->sequence.store(pos + 1, std::memory_order_release);
        return true;
    }

    /**
     * Try to dequeue an item
     * @return true if successful, false if queue is empty
     */
    bool tryPop(T& item) {
        Cell* cell;
        size_t pos = head_.load(std::memory_order_relaxed);

        for (;;) {
            cell = &buffer_[pos % capacity_];
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t diff = (intptr_t)seq - (intptr_t)(pos + 1);

            if (diff == 0) {
                // Cell is ready for reading
                if (head_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                // Queue is empty
                return false;
            } else {
                // Another thread beat us, retry
                pos = head_.load(std::memory_order_relaxed);
            }
        }

        item = cell->data;
        cell->sequence.store(pos + capacity_, std::memory_order_release);
        return true;
    }

    /**
     * Check if queue is empty (approximate)
     */
    bool empty() const {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t tail = tail_.load(std::memory_order_relaxed);
        return head >= tail;
    }

    /**
     * Get approximate size
     */
    size_t size() const {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t tail = tail_.load(std::memory_order_relaxed);
        return (tail > head) ? (tail - head) : 0;
    }

private:
    struct Cell {
        std::atomic<size_t> sequence;
        T data;
    };

    size_t capacity_;
    std::vector<Cell> buffer_;

    // Separate cache lines for head and tail to avoid false sharing
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
};

#endif // LOCKFREE_QUEUE_H
