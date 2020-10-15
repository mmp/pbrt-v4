// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_TAGGEDPTR_H
#define PBRT_UTIL_TAGGEDPTR_H

#include <pbrt/pbrt.h>
#include <pbrt/util/check.h>
#include <pbrt/util/containers.h>
#include <pbrt/util/print.h>

#include <algorithm>
#include <string>
#include <type_traits>

namespace pbrt {

// TaggedPointer Helper Templates
template <int n>
struct DispatchSplit {
    template <typename F, typename Tp, typename... Ts>
    PBRT_CPU_GPU inline auto operator()(F func, Tp tp, int tag, TypePack<Ts...> types) {
        // Recursively consider half of the candidate types depending on _tag_
        constexpr int mid = n / 2;
        if (tag - 1 < mid)
            return DispatchSplit<mid>()(
                func, tp, tag, typename TakeFirstN<mid, TypePack<Ts...>>::type());
        else
            return DispatchSplit<n - mid>()(
                func, tp, tag - mid, typename RemoveFirstN<mid, TypePack<Ts...>>::type());
    }
};

template <>
struct DispatchSplit<1> {
    template <typename F, typename Tp, typename... Ts>
    PBRT_CPU_GPU inline auto operator()(F func, Tp tp, int tag, TypePack<Ts...> types) {
        DCHECK_EQ(1, tag);
        static_assert(sizeof...(Ts) == 1);
        using T = typename GetFirst<TypePack<Ts...>>::type;
        return func(tp.template Cast<T>());
    }
};

// FIXME: can we at least DispatchCRef this from the caller and dispatch based on
// whether F's return type is a const reference?
//
// https://stackoverflow.com/a/41538114 :-p

template <int n>
struct DispatchSplitCRef;

template <>
struct DispatchSplitCRef<1> {
    template <typename F, typename Tp, typename... Ts>
    PBRT_CPU_GPU inline auto operator()(F func, Tp tp, int tag, TypePack<Ts...> types)
        -> auto && {
        DCHECK_EQ(1, tag);
        static_assert(sizeof...(Ts) == 1);
        using T = typename GetFirst<TypePack<Ts...>>::type;
        return func(tp.template Cast<T>());
    }
};

template <int n>
struct DispatchSplitCRef {
    template <typename F, typename Tp, typename... Ts>
    PBRT_CPU_GPU inline auto operator()(F func, Tp tp, int tag, TypePack<Ts...> types)
        -> auto && {
        constexpr int mid = n / 2;

        if (tag - 1 < mid)  // 0-based indexing here to be more traditional
            return DispatchSplitCRef<mid>()(
                func, tp, tag, typename TakeFirstN<mid, TypePack<Ts...>>::type());
        else
            return DispatchSplitCRef<n - mid>()(
                func, tp, tag - mid, typename RemoveFirstN<mid, TypePack<Ts...>>::type());
    }
};

template <int n>
struct DispatchSplitCPU;

template <>
struct DispatchSplitCPU<1> {
    template <typename F, typename Tp, typename... Ts>
    inline auto operator()(F func, Tp tp, int tag, TypePack<Ts...> types) {
        DCHECK_EQ(1, tag);
        static_assert(sizeof...(Ts) == 1);

        using T = typename GetFirst<TypePack<Ts...>>::type;
        return func(tp.template Cast<T>());
    }
};

template <int n>
struct DispatchSplitCPU {
    template <typename F, typename Tp, typename... Ts>
    inline auto operator()(F func, Tp tp, int tag, TypePack<Ts...> types) {
        constexpr int mid = n / 2;

        if (tag - 1 < mid)  // 0-based indexing here to be more traditional
            return DispatchSplitCPU<mid>()(
                func, tp, tag, typename TakeFirstN<mid, TypePack<Ts...>>::type());
        else
            return DispatchSplitCPU<n - mid>()(
                func, tp, tag - mid, typename RemoveFirstN<mid, TypePack<Ts...>>::type());
    }
};

// TaggedPointer Definition
template <typename... Ts>
class TaggedPointer {
  public:
    // TaggedPointer Public Types
    using Types = TypePack<Ts...>;

    // TaggedPointer Public Methods
    TaggedPointer() = default;
    template <typename T>
    PBRT_CPU_GPU TaggedPointer(T *ptr) {
        uintptr_t iptr = reinterpret_cast<uintptr_t>(ptr);
        DCHECK_EQ(iptr & ptrMask, iptr);
        constexpr unsigned int type = TypeIndex<T>();
        bits = iptr | ((uintptr_t)type << tagShift);
    }

    PBRT_CPU_GPU
    TaggedPointer(std::nullptr_t np) {}

    PBRT_CPU_GPU
    TaggedPointer(const TaggedPointer &t) { bits = t.bits; }
    PBRT_CPU_GPU
    TaggedPointer &operator=(const TaggedPointer &t) {
        bits = t.bits;
        return *this;
    }

    template <typename T>
    PBRT_CPU_GPU static constexpr unsigned int TypeIndex() {
        using Tp = typename std::remove_cv_t<T>;
        if constexpr (std::is_same_v<Tp, std::nullptr_t>)
            return 0;
        else
            return 1 + pbrt::IndexOf<Tp, Types>::count;
    }

    PBRT_CPU_GPU
    unsigned int Tag() const { return ((bits & tagMask) >> tagShift); }
    template <typename T>
    PBRT_CPU_GPU bool Is() const {
        return Tag() == TypeIndex<T>();
    }

    PBRT_CPU_GPU
    static constexpr unsigned int MaxTag() { return sizeof...(Ts); }
    PBRT_CPU_GPU
    static constexpr unsigned int NumTags() { return MaxTag() + 1; }

    PBRT_CPU_GPU
    explicit operator bool() const { return (bits & ptrMask) != 0; }

    PBRT_CPU_GPU
    bool operator<(const TaggedPointer &tp) const { return bits < tp.bits; }

    template <typename T>
    PBRT_CPU_GPU T *Cast() {
        DCHECK(Is<T>());
        return reinterpret_cast<T *>(ptr());
    }

    template <typename T>
    PBRT_CPU_GPU const T *Cast() const {
        DCHECK(Is<T>());
        return reinterpret_cast<const T *>(ptr());
    }

    template <typename T>
    PBRT_CPU_GPU T *CastOrNullptr() {
        if (Is<T>())
            return reinterpret_cast<T *>(ptr());
        else
            return nullptr;
    }

    template <typename T>
    PBRT_CPU_GPU const T *CastOrNullptr() const {
        if (Is<T>())
            return reinterpret_cast<const T *>(ptr());
        else
            return nullptr;
    }

    std::string ToString() const {
        return StringPrintf("[ TaggedPointer ptr: 0x%p tag: %d ]", ptr(), Tag());
    }

    PBRT_CPU_GPU
    bool operator==(const TaggedPointer &tp) const { return bits == tp.bits; }
    PBRT_CPU_GPU
    bool operator!=(const TaggedPointer &tp) const { return bits != tp.bits; }

    PBRT_CPU_GPU
    void *ptr() { return reinterpret_cast<void *>(bits & ptrMask); }

    PBRT_CPU_GPU
    const void *ptr() const { return reinterpret_cast<const void *>(bits & ptrMask); }

    template <typename F>
    PBRT_CPU_GPU inline auto Dispatch(F func) {
        DCHECK(ptr() != nullptr);
        return DispatchSplit<MaxTag()>()(func, *this, Tag(), Types());
    }

    template <typename F>
    PBRT_CPU_GPU inline auto Dispatch(F func) const {
        DCHECK(ptr() != nullptr);
        constexpr int n = MaxTag();
        return DispatchSplit<n>()(func, *this, Tag(), Types());
    }

    template <typename F>
    PBRT_CPU_GPU inline auto DispatchCRef(F func) -> auto && {
        DCHECK(ptr() != nullptr);
        constexpr int n = MaxTag();
        return DispatchSplitCRef<n>()(func, *this, Tag(), Types());
    }

    template <typename F>
    PBRT_CPU_GPU inline auto DispatchCRef(F func) const -> auto && {
        DCHECK(ptr() != nullptr);
        constexpr int n = MaxTag();
        return DispatchSplitCRef<n>()(func, *this, Tag(), Types());
    }

    template <typename F>
    inline auto DispatchCPU(F func) {
        DCHECK(ptr() != nullptr);
        constexpr int n = MaxTag();
        return DispatchSplitCPU<n>()(func, *this, Tag(), Types());
    }

    template <typename F>
    inline auto DispatchCPU(F func) const {
        DCHECK(ptr() != nullptr);
        constexpr int n = MaxTag();
        return DispatchSplitCPU<n>()(func, *this, Tag(), Types());
    }

    template <typename F>
    static void ForEachType(F func) {
        pbrt::ForEachType(func, Types());
    }

  private:
    static_assert(sizeof(uintptr_t) == 8, "Expected uintptr_t to be 64 bits");
    // TaggedPointer Private Members
    static constexpr int tagShift = 48;
    static constexpr int tagBits = 64 - tagShift;
    static constexpr uint64_t tagMask = ((1ull << tagBits) - 1) << tagShift;
    static constexpr uint64_t ptrMask = ~tagMask;
    uintptr_t bits = 0;
};

}  // namespace pbrt

#endif  // PBRT_UTIL_TAGGEDPTR_H
