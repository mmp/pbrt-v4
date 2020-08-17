// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_GPU_WORKQUEUE_H
#define PBRT_GPU_WORKQUEUE_H

#include <pbrt/pbrt.h>

#include <pbrt/gpu/launch.h>
#include <pbrt/util/pstd.h>

#include <cuda/atomic>
#include <utility>

namespace pbrt {

template <typename WorkItem>
class WorkQueue : public SOA<WorkItem> {
  public:
    WorkQueue(int n, Allocator alloc) : SOA<WorkItem>(n, alloc) {}

    PBRT_CPU_GPU
    int Size() const { return size.load(cuda::std::memory_order_relaxed); }

    PBRT_CPU_GPU
    void Reset() { size.store(0, cuda::std::memory_order_relaxed); }

    PBRT_CPU_GPU
    int Push(WorkItem w) {
        int index = size.fetch_add(1, cuda::std::memory_order_relaxed);
        (*this)[index] = w;
        return index;
    }

  protected:
    cuda::atomic<int, cuda::thread_scope_device> size{0};
};

template <typename F, typename WorkItem>
void ForAllQueued(const char *desc, WorkQueue<WorkItem> *q, int maxQueued, F func) {
    GPUParallelFor(desc, maxQueued, [=] PBRT_GPU(int index) {
        if (index >= q->Size())
            return;
        func((*q)[index], index);
    });
}

template <template <typename> class Work, typename... Ts>
class MultiWorkQueueHelper;

template <template <typename> class Work>
class MultiWorkQueueHelper<Work> {
  public:
    MultiWorkQueueHelper(int n, Allocator alloc, pstd::span<const bool>) {}
};

template <template <typename> class WorkItem, typename T, typename... Ts>
class MultiWorkQueueHelper<WorkItem, T, Ts...>
    : public MultiWorkQueueHelper<WorkItem, Ts...> {
  public:
    MultiWorkQueueHelper(int n, Allocator alloc, pstd::span<const bool> haveType)
        : MultiWorkQueueHelper<WorkItem, Ts...>(n, alloc,
                                                haveType.subspan(1, haveType.size())),
          q(haveType.front() ? n : 1, alloc) {}

    template <typename Tsz>
    PBRT_CPU_GPU int Size() const {
        if constexpr (std::is_same_v<Tsz, T>)
            return q.Size();
        else
            return MultiWorkQueueHelper<WorkItem, Ts...>::template Size<Tsz>();
    }

    PBRT_CPU_GPU
    void Reset() {
        q.Reset();
        if constexpr (sizeof...(Ts) > 0)
            MultiWorkQueueHelper<WorkItem, Ts...>::Reset();
    }

    template <typename Tg>
    PBRT_CPU_GPU WorkQueue<WorkItem<Tg>> *Get() {
        if constexpr (std::is_same_v<Tg, T>)
            return &q;
        else
            return MultiWorkQueueHelper<WorkItem, Ts...>::template Get<Tg>();
    }

    template <typename Tq, typename... Args>
    PBRT_CPU_GPU int Push(Args &&... args) {
        if constexpr (std::is_same_v<Tq, T>)
            return q.Push(std::forward<Args>(args)...);
        else
            return MultiWorkQueueHelper<WorkItem, Ts...>::template Push<Tq>(
                std::forward<Args>(args)...);
    }

  private:
    cuda::atomic<int, cuda::thread_scope_device> size{0};
    WorkQueue<WorkItem<T>> q;
};

template <template <typename> class WorkItem, typename... Ts>
class MultiWorkQueue {
  public:
    MultiWorkQueue(int n, Allocator alloc, pstd::span<const bool> haveType)
        : helper(n, alloc, haveType) {}

    template <typename T>
    PBRT_CPU_GPU int Size() const {
        return helper.template Size<T>();
    }

    PBRT_CPU_GPU
    void Reset() { helper.Reset(); }

    template <typename T>
    PBRT_CPU_GPU WorkQueue<WorkItem<T>> *Get() {
        return helper.template Get<T>();
    }

    template <typename T, typename... Args>
    PBRT_CPU_GPU int Push(Args &&... args) {
        return helper.template Push<T>(std::forward<Args>(args)...);
    }

  private:
    MultiWorkQueueHelper<WorkItem, Ts...> helper;
};

}  // namespace pbrt

#endif  // PBRT_GPU_WORKQUEUE_H
