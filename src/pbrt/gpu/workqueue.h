// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_GPU_WORKQUEUE_H
#define PBRT_GPU_WORKQUEUE_H

#include <pbrt/pbrt.h>

#include <pbrt/gpu/launch.h>
#include <pbrt/util/pstd.h>

#include <utility>

#ifdef PBRT_IS_WINDOWS
#if (__CUDA_ARCH__ >= 700)
#define PBRT_HAVE_CUDA_ATOMICS
#endif
#else
#if (__CUDA_ARCH__ >= 600)
#define PBRT_HAVE_CUDA_ATOMICS
#endif
#endif // PBRT_IS_WINDOWS

#ifdef PBRT_HAVE_CUDA_ATOMICS
#include <cuda/atomic>
#endif // PBRT_HAVE_CUDA_ATOMICS

namespace pbrt {

template <typename WorkItem>
class WorkQueue : public SOA<WorkItem> {
  public:
    WorkQueue(int n, Allocator alloc) : SOA<WorkItem>(n, alloc) {}

    PBRT_CPU_GPU
    int Size() const {
#ifdef PBRT_HAVE_CUDA_ATOMICS
        return size.load(cuda::std::memory_order_relaxed);
#else
        return size;
#endif
    }

    PBRT_CPU_GPU
    void Reset() {
#ifdef PBRT_HAVE_CUDA_ATOMICS
        size.store(0, cuda::std::memory_order_relaxed);
#else
        size = 0;
#endif
    }

    PBRT_CPU_GPU
    int Push(WorkItem w) {
        int index = AllocateEntry();
        (*this)[index] = w;
        return index;
    }

  protected:
    PBRT_CPU_GPU
    int AllocateEntry() {
#ifdef PBRT_HAVE_CUDA_ATOMICS
        return size.fetch_add(1, cuda::std::memory_order_relaxed);
#else
#ifdef PBRT_IS_GPU_CODE
        return atomicAdd(&size, 1);
#else
        assert(!"this shouldn't be called");
        return 0;
#endif
#endif
    }

  private:
#ifdef PBRT_HAVE_CUDA_ATOMICS
    cuda::atomic<int, cuda::thread_scope_device> size{0};
#else
    int size = 0;
#endif
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
