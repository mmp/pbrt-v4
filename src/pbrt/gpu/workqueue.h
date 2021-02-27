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
    WorkQueue() = default;
    WorkQueue(int n, Allocator alloc) : SOA<WorkItem>(n, alloc) {}
    WorkQueue &operator=(const WorkQueue &w) {
        SOA<WorkItem>::operator=(w);
#ifdef PBRT_HAVE_CUDA_ATOMICS
        size.store(w.size.load());
#else
        size = w.size;
#endif
        return *this;
    }

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

template <typename... Ts>
class MultiWorkQueue<TypePack<Ts...>> {
  public:
    // MultiWorkQueue Public Methods
    template <typename T>
    PBRT_CPU_GPU WorkQueue<T> *Get() {
        return &pstd::get<WorkQueue<T>>(queues);
    }

    MultiWorkQueue(int n, Allocator alloc, pstd::span<const bool> haveType) {
        int index = 0;
        ((*Get<Ts>() = WorkQueue<Ts>(haveType[index++] ? n : 1, alloc)), ...);
    }

    template <typename T>
    PBRT_CPU_GPU int Size() const {
        return Get<T>()->Size();
    }
    template <typename T>
    PBRT_CPU_GPU int Push(const T &value) {
        return Get<T>()->Push(value);
    }
    PBRT_CPU_GPU
    void Reset() { (Get<Ts>()->Reset(), ...); }

  private:
    // MultiWorkQueue Private Members
    pstd::tuple<WorkQueue<Ts>...> queues;
};

}  // namespace pbrt

#endif  // PBRT_GPU_WORKQUEUE_H
