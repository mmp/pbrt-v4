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
#endif  // PBRT_IS_WINDOWS

#ifdef PBRT_HAVE_CUDA_ATOMICS
#include <cuda/atomic>
#endif  // PBRT_HAVE_CUDA_ATOMICS

namespace pbrt {

// WorkQueue Definition
template <typename WorkItem>
class WorkQueue : public SOA<WorkItem> {
  public:
    // WorkQueue Public Methods
    WorkQueue(int n, Allocator alloc) : SOA<WorkItem>(n, alloc) {}

    PBRT_CPU_GPU
    int Size() const {
#ifdef PBRT_HAVE_CUDA_ATOMICS
        namespace std = cuda::std;
        return size.load(std::memory_order_relaxed);
#else
        return size;
#endif
    }
    PBRT_CPU_GPU
    void Reset() {
#ifdef PBRT_HAVE_CUDA_ATOMICS
        namespace std = cuda::std;
        size.store(0, std::memory_order_relaxed);
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
    // WorkQueue Protected Methods
    PBRT_CPU_GPU
    int AllocateEntry() {
#ifdef PBRT_HAVE_CUDA_ATOMICS
        namespace std = cuda::std;
        return size.fetch_add(1, std::memory_order_relaxed);
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
    // WorkQueue Private Members
#ifdef PBRT_HAVE_CUDA_ATOMICS
    using GPUAtomicInt = cuda::atomic<int, cuda::thread_scope_device>;
    GPUAtomicInt size{0};
#else
    int size = 0;
#endif
};

// WorkQueue Inline Functions
template <typename F, typename WorkItem>
void ForAllQueued(const char *desc, WorkQueue<WorkItem> *q, int maxQueued, F func) {
    GPUParallelFor(desc, maxQueued, [=] PBRT_GPU(int index) mutable {
        if (index >= q->Size())
            return;
        func((*q)[index]);
    });
}

// MultiWorkQueue Definition
template <typename T>
class MultiWorkQueue;

template <>
class MultiWorkQueue<TypePack<>> {
  public:
    MultiWorkQueue(int, Allocator, pstd::span<const bool>) {}
};

template <typename T, typename... Ts>
class MultiWorkQueue<TypePack<T, Ts...>> : public MultiWorkQueue<TypePack<Ts...>> {
  public:
    // MultiWorkQueue Public Methods
    MultiWorkQueue(int n, Allocator alloc, pstd::span<const bool> haveType)
        : MultiWorkQueue<TypePack<Ts...>>(n, alloc, haveType.subspan(1, haveType.size())),
          q(haveType.front() ? n : 1, alloc) {}

    template <typename Tsz>
    PBRT_CPU_GPU int Size() const {
        if constexpr (std::is_same_v<Tsz, T>)
            return q.Size();
        else
            return MultiWorkQueue<TypePack<Ts...>>::template Size<Tsz>();
    }

    PBRT_CPU_GPU
    void Reset() {
        q.Reset();
        if constexpr (sizeof...(Ts) > 0)
            MultiWorkQueue<TypePack<Ts...>>::Reset();
    }

    template <typename Tg>
    PBRT_CPU_GPU WorkQueue<Tg> *Get() {
        if constexpr (std::is_same_v<Tg, T>)
            return &q;
        else
            return MultiWorkQueue<TypePack<Ts...>>::template Get<Tg>();
    }

    template <typename Tp>
    PBRT_CPU_GPU int Push(Tp item) {
        if constexpr (std::is_same_v<Tp, T>)
            return q.Push(item);
        else
            return MultiWorkQueue<TypePack<Ts...>>::template Push(item);
    }

  private:
    // MultiWorkQueue Private Members
    WorkQueue<T> q;
};

}  // namespace pbrt

#endif  // PBRT_GPU_WORKQUEUE_H
