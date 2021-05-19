// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_PSTD_H
#define PBRT_UTIL_PSTD_H

#include <pbrt/util/check.h>

#include <float.h>
#include <limits.h>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <list>
#include <new>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>

namespace pstd {

template <typename T>
PBRT_CPU_GPU inline void swap(T &a, T &b) {
    T tmp = std::move(a);
    a = std::move(b);
    b = std::move(tmp);
}

template <class To, class From>
PBRT_CPU_GPU typename std::enable_if_t<sizeof(To) == sizeof(From) &&
                                           std::is_trivially_copyable_v<From> &&
                                           std::is_trivially_copyable_v<To>,
                                       To>
bit_cast(const From &src) noexcept {
    static_assert(std::is_trivially_constructible_v<To>,
                  "This implementation requires the destination type to be trivially "
                  "constructible");
    To dst;
    std::memcpy(&dst, &src, sizeof(To));
    return dst;
}

template <typename T, int N>
class array;

// Specialization for zero element arrays (to make MSVC happy)
template <typename T>
class array<T, 0> {
  public:
    using value_type = T;
    using iterator = value_type *;
    using const_iterator = const value_type *;
    using size_t = std::size_t;

    array() = default;

    PBRT_CPU_GPU
    void fill(const T &v) { assert(!"should never be called"); }

    PBRT_CPU_GPU
    bool operator==(const array<T, 0> &a) const { return true; }
    PBRT_CPU_GPU
    bool operator!=(const array<T, 0> &a) const { return false; }

    PBRT_CPU_GPU
    iterator begin() { return nullptr; }
    PBRT_CPU_GPU
    iterator end() { return nullptr; }
    PBRT_CPU_GPU
    const_iterator begin() const { return nullptr; }
    PBRT_CPU_GPU
    const_iterator end() const { return nullptr; }

    PBRT_CPU_GPU
    size_t size() const { return 0; }

    PBRT_CPU_GPU
    T &operator[](size_t i) {
        assert(!"should never be called");
        static T t;
        return t;
    }
    PBRT_CPU_GPU
    const T &operator[](size_t i) const {
        assert(!"should never be called");
        static T t;
        return t;
    }

    PBRT_CPU_GPU
    T *data() { return nullptr; }
    PBRT_CPU_GPU
    const T *data() const { return nullptr; }
};

template <typename T, int N>
class array {
  public:
    using value_type = T;
    using iterator = value_type *;
    using const_iterator = const value_type *;
    using size_t = std::size_t;

    array() = default;
    PBRT_CPU_GPU
    array(std::initializer_list<T> v) {
        size_t i = 0;
        for (const T &val : v)
            values[i++] = val;
    }

    PBRT_CPU_GPU
    void fill(const T &v) {
        for (int i = 0; i < N; ++i)
            values[i] = v;
    }

    PBRT_CPU_GPU
    bool operator==(const array<T, N> &a) const {
        for (int i = 0; i < N; ++i)
            if (values[i] != a.values[i])
                return false;
        return true;
    }
    PBRT_CPU_GPU
    bool operator!=(const array<T, N> &a) const { return !(*this == a); }

    PBRT_CPU_GPU
    iterator begin() { return values; }
    PBRT_CPU_GPU
    iterator end() { return values + N; }
    PBRT_CPU_GPU
    const_iterator begin() const { return values; }
    PBRT_CPU_GPU
    const_iterator end() const { return values + N; }

    PBRT_CPU_GPU
    size_t size() const { return N; }

    PBRT_CPU_GPU
    T &operator[](size_t i) { return values[i]; }
    PBRT_CPU_GPU
    const T &operator[](size_t i) const { return values[i]; }

    PBRT_CPU_GPU
    T *data() { return values; }
    PBRT_CPU_GPU
    const T *data() const { return values; }

  private:
    T values[N] = {};
};

template <typename T>
class optional {
  public:
    using value_type = T;

    optional() = default;
    PBRT_CPU_GPU
    optional(const T &v) : set(true) { new (ptr()) T(v); }
    PBRT_CPU_GPU
    optional(T &&v) : set(true) { new (ptr()) T(std::move(v)); }
    PBRT_CPU_GPU
    optional(const optional &v) : set(v.has_value()) {
        if (v.has_value())
            new (ptr()) T(v.value());
    }
    PBRT_CPU_GPU
    optional(optional &&v) : set(v.has_value()) {
        if (v.has_value()) {
            new (ptr()) T(std::move(v.value()));
            v.reset();
        }
    }

    PBRT_CPU_GPU
    optional &operator=(const T &v) {
        reset();
        new (ptr()) T(v);
        set = true;
        return *this;
    }
    PBRT_CPU_GPU
    optional &operator=(T &&v) {
        reset();
        new (ptr()) T(std::move(v));
        set = true;
        return *this;
    }
    PBRT_CPU_GPU
    optional &operator=(const optional &v) {
        reset();
        if (v.has_value()) {
            new (ptr()) T(v.value());
            set = true;
        }
        return *this;
    }
    PBRT_CPU_GPU
    optional &operator=(optional &&v) {
        reset();
        if (v.has_value()) {
            new (ptr()) T(std::move(v.value()));
            set = true;
            v.reset();
        }
        return *this;
    }

    PBRT_CPU_GPU
    ~optional() { reset(); }

    PBRT_CPU_GPU
    explicit operator bool() const { return set; }

    PBRT_CPU_GPU
    T value_or(const T &alt) const { return set ? value() : alt; }

    PBRT_CPU_GPU
    T *operator->() { return &value(); }
    PBRT_CPU_GPU
    const T *operator->() const { return &value(); }
    PBRT_CPU_GPU
    T &operator*() { return value(); }
    PBRT_CPU_GPU
    const T &operator*() const { return value(); }
    PBRT_CPU_GPU
    T &value() {
        CHECK(set);
        return *ptr();
    }
    PBRT_CPU_GPU
    const T &value() const {
        CHECK(set);
        return *ptr();
    }

    PBRT_CPU_GPU
    void reset() {
        if (set) {
            value().~T();
            set = false;
        }
    }

    PBRT_CPU_GPU
    bool has_value() const { return set; }

  private:
#ifdef __NVCC__
    // Work-around NVCC bug
    PBRT_CPU_GPU
    T *ptr() { return reinterpret_cast<T *>(&optionalValue); }
    PBRT_CPU_GPU
    const T *ptr() const { return reinterpret_cast<const T *>(&optionalValue); }
#else
    PBRT_CPU_GPU
    T *ptr() { return std::launder(reinterpret_cast<T *>(&optionalValue)); }
    PBRT_CPU_GPU
    const T *ptr() const {
        return std::launder(reinterpret_cast<const T *>(&optionalValue));
    }
#endif

    std::aligned_storage_t<sizeof(T), alignof(T)> optionalValue;
    bool set = false;
};

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const optional<T> &opt) {
    if (opt.has_value())
        return os << "[ pstd::optional<" << typeid(T).name() << "> set: true "
                  << "value: " << opt.value() << " ]";
    else
        return os << "[ pstd::optional<" << typeid(T).name()
                  << "> set: false value: n/a ]";
}

namespace span_internal {

// Wrappers for access to container data pointers.
template <typename C>
PBRT_CPU_GPU inline constexpr auto GetDataImpl(C &c, char) noexcept
    -> decltype(c.data()) {
    return c.data();
}

template <typename C>
PBRT_CPU_GPU inline constexpr auto GetData(C &c) noexcept -> decltype(GetDataImpl(c, 0)) {
    return GetDataImpl(c, 0);
}

// Detection idioms for size() and data().
template <typename C>
using HasSize =
    std::is_integral<typename std::decay<decltype(std::declval<C &>().size())>::type>;

// We want to enable conversion from vector<T*> to span<const T* const> but
// disable conversion from vector<Derived> to span<Base>. Here we use
// the fact that U** is convertible to Q* const* if and only if Q is the same
// type or a more cv-qualified version of U.  We also decay the result type of
// data() to avoid problems with classes which have a member function data()
// which returns a reference.
template <typename T, typename C>
using HasData = std::is_convertible<
    typename std::decay<decltype(GetData(std::declval<C &>()))>::type *, T *const *>;

}  // namespace span_internal

// span implementation partially based on absl::Span from Google's Abseil library.
template <typename T>
class span {
  public:
    // Used to determine whether a Span can be constructed from a container of
    // type C.
    template <typename C>
    using EnableIfConvertibleFrom =
        typename std::enable_if_t<span_internal::HasData<T, C>::value &&
                                  span_internal::HasSize<C>::value>;

    // Used to SFINAE-enable a function when the slice elements are const.
    template <typename U>
    using EnableIfConstView = typename std::enable_if_t<std::is_const<T>::value, U>;

    // Used to SFINAE-enable a function when the slice elements are mutable.
    template <typename U>
    using EnableIfMutableView = typename std::enable_if_t<!std::is_const<T>::value, U>;

    using value_type = typename std::remove_cv<T>::type;
    using iterator = T *;
    using const_iterator = const T *;

    PBRT_CPU_GPU
    span() : ptr(nullptr), n(0) {}
    PBRT_CPU_GPU
    span(T *ptr, size_t n) : ptr(ptr), n(n) {}
    template <size_t N>
    PBRT_CPU_GPU span(T (&a)[N]) : span(a, N) {}
    PBRT_CPU_GPU
    span(std::initializer_list<value_type> v) : span(v.begin(), v.size()) {}

    // Explicit reference constructor for a mutable `span<T>` type. Can be
    // replaced with Makespan() to infer the type parameter.
    template <typename V, typename = EnableIfConvertibleFrom<V>,
              typename = EnableIfMutableView<V>>
    PBRT_CPU_GPU explicit span(V &v) noexcept : span(v.data(), v.size()) {}

    // Implicit reference constructor for a read-only `span<const T>` type
    template <typename V, typename = EnableIfConvertibleFrom<V>,
              typename = EnableIfConstView<V>>
    PBRT_CPU_GPU constexpr span(const V &v) noexcept : span(v.data(), v.size()) {}

    PBRT_CPU_GPU
    iterator begin() { return ptr; }
    PBRT_CPU_GPU
    iterator end() { return ptr + n; }
    PBRT_CPU_GPU
    const_iterator begin() const { return ptr; }
    PBRT_CPU_GPU
    const_iterator end() const { return ptr + n; }

    PBRT_CPU_GPU
    T &operator[](size_t i) {
        DCHECK_LT(i, size());
        return ptr[i];
    }
    PBRT_CPU_GPU
    const T &operator[](size_t i) const {
        DCHECK_LT(i, size());
        return ptr[i];
    }

    PBRT_CPU_GPU
    size_t size() const { return n; };
    PBRT_CPU_GPU
    bool empty() const { return size() == 0; }
    PBRT_CPU_GPU
    T *data() { return ptr; }
    PBRT_CPU_GPU
    const T *data() const { return ptr; }

    PBRT_CPU_GPU
    T front() const { return ptr[0]; }
    PBRT_CPU_GPU
    T back() const { return ptr[n - 1]; }

    PBRT_CPU_GPU
    void remove_prefix(size_t count) {
        // assert(size() >= count);
        ptr += count;
        n -= count;
    }
    PBRT_CPU_GPU
    void remove_suffix(size_t count) {
        // assert(size() > = count);
        n -= count;
    }

    PBRT_CPU_GPU
    span subspan(size_t pos, size_t count) {
        size_t np = count < (size() - pos) ? count : (size() - pos);
        return span(ptr + pos, np);
    }

  private:
    T *ptr;
    size_t n;
};

template <int &... ExplicitArgumentBarrier, typename T>
PBRT_CPU_GPU inline constexpr span<T> MakeSpan(T *ptr, size_t size) noexcept {
    return span<T>(ptr, size);
}

template <int &... ExplicitArgumentBarrier, typename T>
PBRT_CPU_GPU inline span<T> MakeSpan(T *begin, T *end) noexcept {
    return span<T>(begin, end - begin);
}

template <int &... ExplicitArgumentBarrier, typename C>
PBRT_CPU_GPU inline constexpr auto MakeSpan(C &c) noexcept
    -> decltype(MakeSpan(span_internal::GetData(c), c.size())) {
    return MakeSpan(span_internal::GetData(c), c.size());
}

template <int &... ExplicitArgumentBarrier, typename T, size_t N>
PBRT_CPU_GPU inline constexpr span<T> MakeSpan(T (&array)[N]) noexcept {
    return span<T>(array, N);
}

template <int &... ExplicitArgumentBarrier, typename T>
PBRT_CPU_GPU inline constexpr span<const T> MakeConstSpan(T *ptr, size_t size) noexcept {
    return span<const T>(ptr, size);
}

template <int &... ExplicitArgumentBarrier, typename T>
PBRT_CPU_GPU inline span<const T> MakeConstSpan(T *begin, T *end) noexcept {
    return span<const T>(begin, end - begin);
}

template <int &... ExplicitArgumentBarrier, typename C>
PBRT_CPU_GPU inline constexpr auto MakeConstSpan(const C &c) noexcept
    -> decltype(MakeSpan(c)) {
    return MakeSpan(c);
}

template <int &... ExplicitArgumentBarrier, typename T, size_t N>
PBRT_CPU_GPU inline constexpr span<const T> MakeConstSpan(const T (&array)[N]) noexcept {
    return span<const T>(array, N);
}

// memory_resource...

namespace pmr {

class memory_resource {
    static constexpr size_t max_align = alignof(std::max_align_t);

  public:
    virtual ~memory_resource();
    void *allocate(size_t bytes, size_t alignment = max_align) {
        if (bytes == 0)
            return nullptr;
        return do_allocate(bytes, alignment);
    }
    void deallocate(void *p, size_t bytes, size_t alignment = max_align) {
        if (!p)
            return;
        return do_deallocate(p, bytes, alignment);
    }
    bool is_equal(const memory_resource &other) const noexcept {
        return do_is_equal(other);
    }

  private:
    virtual void *do_allocate(size_t bytes, size_t alignment) = 0;
    virtual void do_deallocate(void *p, size_t bytes, size_t alignment) = 0;
    virtual bool do_is_equal(const memory_resource &other) const noexcept = 0;
};

inline bool operator==(const memory_resource &a, const memory_resource &b) noexcept {
    return a.is_equal(b);
}

inline bool operator!=(const memory_resource &a, const memory_resource &b) noexcept {
    return !(a == b);
}

// TODO
struct pool_options {
    size_t max_blocks_per_chunk = 0;
    size_t largest_required_pool_block = 0;
};
class synchronized_pool_resource;
class unsynchronized_pool_resource;

// global memory resources
memory_resource *new_delete_resource() noexcept;
// TODO: memory_resource* null_memory_resource() noexcept;
memory_resource *set_default_resource(memory_resource *r) noexcept;
memory_resource *get_default_resource() noexcept;

class monotonic_buffer_resource : public memory_resource {
  public:
    explicit monotonic_buffer_resource(memory_resource *upstream)
        : upstreamResource(upstream) {}
    monotonic_buffer_resource(size_t blockSize, memory_resource *upstream)
        : blockSize(blockSize), upstreamResource(upstream) {}
#if 0
    // TODO
    monotonic_buffer_resource(void *buffer, size_t buffer_size,
                              memory_resource *upstream);
#endif
    monotonic_buffer_resource() : monotonic_buffer_resource(get_default_resource()) {}
    explicit monotonic_buffer_resource(size_t initial_size)
        : monotonic_buffer_resource(initial_size, get_default_resource()) {}
#if 0
    // TODO
    monotonic_buffer_resource(void *buffer, size_t buffer_size)
        : monotonic_buffer_resource(buffer, buffer_size, get_default_resource()) {}
#endif
    monotonic_buffer_resource(const monotonic_buffer_resource &) = delete;

    ~monotonic_buffer_resource() { release(); }

    monotonic_buffer_resource operator=(const monotonic_buffer_resource &) = delete;

    void release() {
        for (const auto &block : usedBlocks)
            upstreamResource->deallocate(block.ptr, block.size);
        usedBlocks.clear();

        upstreamResource->deallocate(currentBlock.ptr, currentBlock.size);
        currentBlock = MemoryBlock();
    }

    memory_resource *upstream_resource() const { return upstreamResource; }

  protected:
    void *do_allocate(size_t bytes, size_t align) override {
        if (bytes > blockSize) {
            // We've got a big allocation; let the current block be so that
            // smaller allocations have a chance at using up more of it.
            usedBlocks.push_back(
                MemoryBlock{upstreamResource->allocate(bytes, align), bytes});
            return usedBlocks.back().ptr;
        }

        if ((currentBlockPos % align) != 0)
            currentBlockPos += align - (currentBlockPos % align);
        DCHECK_EQ(0, currentBlockPos % align);

        if (currentBlockPos + bytes > currentBlock.size) {
            // Add current block to _usedBlocks_ list
            if (currentBlock.size) {
                usedBlocks.push_back(currentBlock);
                currentBlock = {};
            }

            currentBlock = {
                upstreamResource->allocate(blockSize, alignof(std::max_align_t)),
                blockSize};
            currentBlockPos = 0;
        }

        void *ptr = (char *)currentBlock.ptr + currentBlockPos;
        currentBlockPos += bytes;
        return ptr;
    }

    void do_deallocate(void *p, size_t bytes, size_t alignment) override {
        // no-op
    }

    bool do_is_equal(const memory_resource &other) const noexcept override {
        return this == &other;
    }

  private:
    struct MemoryBlock {
        void *ptr = nullptr;
        size_t size = 0;
    };

    memory_resource *upstreamResource;
    size_t blockSize = 256 * 1024;
    MemoryBlock currentBlock;
    size_t currentBlockPos = 0;
    // TODO: should use the memory_resource for this list's allocations...
    std::list<MemoryBlock> usedBlocks;
};

template <class Tp = std::byte>
class polymorphic_allocator {
  public:
    using value_type = Tp;

    polymorphic_allocator() noexcept { memoryResource = new_delete_resource(); }
    polymorphic_allocator(memory_resource *r) : memoryResource(r) {}
    polymorphic_allocator(const polymorphic_allocator &other) = default;
    template <class U>
    polymorphic_allocator(const polymorphic_allocator<U> &other) noexcept
        : memoryResource(other.resource()) {}

    polymorphic_allocator &operator=(const polymorphic_allocator &rhs) = delete;

    // member functions
    [[nodiscard]] Tp *allocate(size_t n) {
        return static_cast<Tp *>(resource()->allocate(n * sizeof(Tp), alignof(Tp)));
    }
    void deallocate(Tp *p, size_t n) { resource()->deallocate(p, n); }

    void *allocate_bytes(size_t nbytes, size_t alignment = alignof(max_align_t)) {
        return resource()->allocate(nbytes, alignment);
    }
    void deallocate_bytes(void *p, size_t nbytes,
                          size_t alignment = alignof(std::max_align_t)) {
        return resource()->deallocate(p, nbytes, alignment);
    }
    template <class T>
    T *allocate_object(size_t n = 1) {
        return static_cast<T *>(allocate_bytes(n * sizeof(T), alignof(T)));
    }
    template <class T>
    void deallocate_object(T *p, size_t n = 1) {
        deallocate_bytes(p, n * sizeof(T), alignof(T));
    }
    template <class T, class... Args>
    T *new_object(Args &&... args) {
        // NOTE: this doesn't handle constructors that throw exceptions...
        T *p = allocate_object<T>();
        construct(p, std::forward<Args>(args)...);
        return p;
    }
    template <class T>
    void delete_object(T *p) {
        destroy(p);
        deallocate_object(p);
    }

    template <class T, class... Args>
    void construct(T *p, Args &&... args) {
        ::new ((void *)p) T(std::forward<Args>(args)...);
    }

    template <class T>
    void destroy(T *p) {
        p->~T();
    }

    // polymorphic_allocator select_on_container_copy_construction() const;

    memory_resource *resource() const { return memoryResource; }

  private:
    memory_resource *memoryResource;
};

template <class T1, class T2>
bool operator==(const polymorphic_allocator<T1> &a,
                const polymorphic_allocator<T2> &b) noexcept {
    return a.resource() == b.resource();
}

template <class T1, class T2>
bool operator!=(const polymorphic_allocator<T1> &a,
                const polymorphic_allocator<T2> &b) noexcept {
    return !(a == b);
}

}  // namespace pmr

template <typename T, class Allocator = pmr::polymorphic_allocator<T>>
class vector {
  public:
    using value_type = T;
    using allocator_type = Allocator;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = T *;
    using const_pointer = const T *;
    using iterator = T *;
    using const_iterator = const T *;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const iterator>;

    vector(const Allocator &alloc = {}) : alloc(alloc) {}
    vector(size_t count, const T &value, const Allocator &alloc = {}) : alloc(alloc) {
        reserve(count);
        for (size_t i = 0; i < count; ++i)
            this->alloc.template construct<T>(ptr + i, value);
        nStored = count;
    }
    vector(size_t count, const Allocator &alloc = {}) : vector(count, T{}, alloc) {}
    vector(const vector &other, const Allocator &alloc = {}) : alloc(alloc) {
        reserve(other.size());
        for (size_t i = 0; i < other.size(); ++i)
            this->alloc.template construct<T>(ptr + i, other[i]);
        nStored = other.size();
    }
    template <class InputIt>
    vector(InputIt first, InputIt last, const Allocator &alloc = {}) : alloc(alloc) {
        reserve(last - first);
        size_t i = 0;
        for (InputIt iter = first; iter != last; ++iter, ++i)
            this->alloc.template construct<T>(ptr + i, *iter);
        nStored = nAlloc;
    }
    vector(vector &&other) : alloc(other.alloc) {
        nStored = other.nStored;
        nAlloc = other.nAlloc;
        ptr = other.ptr;

        other.nStored = other.nAlloc = 0;
        other.ptr = nullptr;
    }
    vector(vector &&other, const Allocator &alloc) {
        if (alloc == other.alloc) {
            ptr = other.ptr;
            nAlloc = other.nAlloc;
            nStored = other.nStored;

            other.ptr = nullptr;
            other.nAlloc = other.nStored = 0;
        } else {
            reserve(other.size());
            for (size_t i = 0; i < other.size(); ++i)
                alloc.template construct<T>(ptr + i, std::move(other[i]));
            nStored = other.size();
        }
    }
    vector(std::initializer_list<T> init, const Allocator &alloc = {})
        : vector(init.begin(), init.end(), alloc) {}

    vector &operator=(const vector &other) {
        if (this == &other)
            return *this;

        clear();
        reserve(other.size());
        for (size_t i = 0; i < other.size(); ++i)
            alloc.template construct<T>(ptr + i, other[i]);
        nStored = other.size();

        return *this;
    }
    vector &operator=(vector &&other) {
        if (this == &other)
            return *this;

        if (alloc == other.alloc) {
            pstd::swap(ptr, other.ptr);
            pstd::swap(nAlloc, other.nAlloc);
            pstd::swap(nStored, other.nStored);
        } else {
            clear();
            reserve(other.size());
            for (size_t i = 0; i < other.size(); ++i)
                alloc.template construct<T>(ptr + i, std::move(other[i]));
            nStored = other.size();
        }

        return *this;
    }
    vector &operator=(std::initializer_list<T> &init) {
        reserve(init.size());
        clear();
        iterator iter = begin();
        for (const auto &value : init) {
            *iter = value;
            ++iter;
        }
        return *this;
    }

    void assign(size_type count, const T &value) {
        clear();
        reserve(count);
        for (size_t i = 0; i < count; ++i)
            push_back(value);
    }
    template <class InputIt>
    void assign(InputIt first, InputIt last) {
        LOG_FATAL("TODO");
        // TODO
    }
    void assign(std::initializer_list<T> &init) { assign(init.begin(), init.end()); }

    ~vector() {
        clear();
        alloc.deallocate_object(ptr, nAlloc);
    }

    PBRT_CPU_GPU
    iterator begin() { return ptr; }
    PBRT_CPU_GPU
    iterator end() { return ptr + nStored; }
    PBRT_CPU_GPU
    const_iterator begin() const { return ptr; }
    PBRT_CPU_GPU
    const_iterator end() const { return ptr + nStored; }
    PBRT_CPU_GPU
    const_iterator cbegin() const { return ptr; }
    PBRT_CPU_GPU
    const_iterator cend() const { return ptr + nStored; }

    PBRT_CPU_GPU
    reverse_iterator rbegin() { return reverse_iterator(end()); }
    PBRT_CPU_GPU
    reverse_iterator rend() { return reverse_iterator(begin()); }
    PBRT_CPU_GPU
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    PBRT_CPU_GPU
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

    allocator_type get_allocator() const { return alloc; }
    PBRT_CPU_GPU
    size_t size() const { return nStored; }
    PBRT_CPU_GPU
    bool empty() const { return size() == 0; }
    PBRT_CPU_GPU
    size_t max_size() const { return (size_t)-1; }
    PBRT_CPU_GPU
    size_t capacity() const { return nAlloc; }
    void reserve(size_t n) {
        if (nAlloc >= n)
            return;

        T *ra = alloc.template allocate_object<T>(n);
        for (int i = 0; i < nStored; ++i) {
            alloc.template construct<T>(ra + i, std::move(begin()[i]));
            alloc.destroy(begin() + i);
        }

        alloc.deallocate_object(ptr, nAlloc);
        nAlloc = n;
        ptr = ra;
    }
    // TODO: shrink_to_fit

    PBRT_CPU_GPU
    reference operator[](size_type index) {
        DCHECK_LT(index, size());
        return ptr[index];
    }
    PBRT_CPU_GPU
    const_reference operator[](size_type index) const {
        DCHECK_LT(index, size());
        return ptr[index];
    }
    PBRT_CPU_GPU
    reference front() { return ptr[0]; }
    PBRT_CPU_GPU
    const_reference front() const { return ptr[0]; }
    PBRT_CPU_GPU
    reference back() { return ptr[nStored - 1]; }
    PBRT_CPU_GPU
    const_reference back() const { return ptr[nStored - 1]; }
    PBRT_CPU_GPU
    pointer data() { return ptr; }
    PBRT_CPU_GPU
    const_pointer data() const { return ptr; }

    void clear() {
        for (int i = 0; i < nStored; ++i)
            alloc.destroy(&ptr[i]);
        nStored = 0;
    }

    iterator insert(const_iterator, const T &value) {
        // TODO
        LOG_FATAL("TODO");
    }
    iterator insert(const_iterator, T &&value) {
        // TODO
        LOG_FATAL("TODO");
    }
    iterator insert(const_iterator pos, size_type count, const T &value) {
        // TODO
        LOG_FATAL("TODO");
    }
    template <class InputIt>
    iterator insert(const_iterator pos, InputIt first, InputIt last) {
        if (pos == end()) {
            size_t firstOffset = size();
            for (auto iter = first; iter != last; ++iter)
                push_back(*iter);
            return begin() + firstOffset;
        } else
            LOG_FATAL("TODO");
    }
    iterator insert(const_iterator pos, std::initializer_list<T> init) {
        // TODO
        LOG_FATAL("TODO");
    }

    template <class... Args>
    iterator emplace(const_iterator pos, Args &&... args) {
        // TODO
        LOG_FATAL("TODO");
    }
    template <class... Args>
    void emplace_back(Args &&... args) {
        if (nAlloc == nStored)
            reserve(nAlloc == 0 ? 4 : 2 * nAlloc);

        alloc.construct(ptr + nStored, std::forward<Args>(args)...);
        ++nStored;
    }

    iterator erase(const_iterator pos) {
        // TODO
        LOG_FATAL("TODO");
    }
    iterator erase(const_iterator first, const_iterator last) {
        // TODO
        LOG_FATAL("TODO");
    }

    void push_back(const T &value) {
        if (nAlloc == nStored)
            reserve(nAlloc == 0 ? 4 : 2 * nAlloc);

        alloc.construct(ptr + nStored, value);
        ++nStored;
    }
    void push_back(T &&value) {
        if (nAlloc == nStored)
            reserve(nAlloc == 0 ? 4 : 2 * nAlloc);

        alloc.construct(ptr + nStored, std::move(value));
        ++nStored;
    }
    void pop_back() {
        DCHECK(!empty());
        alloc.destroy(ptr + nStored - 1);
        --nStored;
    }

    void resize(size_type n) {
        if (n < size()) {
            for (size_t i = n; i < size(); ++i)
                alloc.destroy(ptr + i);
            if (n == 0) {
                alloc.deallocate_object(ptr, nAlloc);
                ptr = nullptr;
                nAlloc = 0;
            }
        } else if (n > size()) {
            reserve(n);
            for (size_t i = nStored; i < n; ++i)
                alloc.construct(ptr + i);
        }
        nStored = n;
    }
    void resize(size_type count, const value_type &value) {
        // TODO
        LOG_FATAL("TODO");
    }

    void swap(vector &other) {
        // TODO
        LOG_FATAL("TODO");
    }

  private:
    Allocator alloc;
    T *ptr = nullptr;
    size_t nAlloc = 0, nStored = 0;
};

template <typename... Ts>
struct tuple;
template <>
struct tuple<> {
    template <size_t>
    using type = void;
};

template <typename T, typename... Ts>
struct tuple<T, Ts...> : tuple<Ts...> {
    using Base = tuple<Ts...>;

    tuple() = default;
    tuple(const tuple &) = default;
    tuple(tuple &&) = default;
    tuple &operator=(tuple &&) = default;
    tuple &operator=(const tuple &) = default;

    tuple(const T &value, const Ts &... values) : Base(values...), value(value) {}

    tuple(T &&value, Ts &&... values)
        : Base(std::move(values)...), value(std::move(value)) {}

    T value;
};

template <typename... Ts>
tuple(Ts &&...) -> tuple<std::decay_t<Ts>...>;

template <size_t I, typename T, typename... Ts>
PBRT_CPU_GPU auto &get(tuple<T, Ts...> &t) {
    if constexpr (I == 0)
        return t.value;
    else
        return get<I - 1>((tuple<Ts...> &)t);
}

template <size_t I, typename T, typename... Ts>
PBRT_CPU_GPU const auto &get(const tuple<T, Ts...> &t) {
    if constexpr (I == 0)
        return t.value;
    else
        return get<I - 1>((const tuple<Ts...> &)t);
}

template <typename Req, typename T, typename... Ts>
PBRT_CPU_GPU auto &get(tuple<T, Ts...> &t) {
    if constexpr (std::is_same_v<Req, T>)
        return t.value;
    else
        return get<Req>((tuple<Ts...> &)t);
}

template <typename Req, typename T, typename... Ts>
PBRT_CPU_GPU const auto &get(const tuple<T, Ts...> &t) {
    if constexpr (std::is_same_v<Req, T>)
        return t.value;
    else
        return get<Req>((const tuple<Ts...> &)t);
}

template <typename T>
struct complex {
    PBRT_CPU_GPU complex(T re) : re(re), im(0) {}
    PBRT_CPU_GPU complex(T re, T im) : re(re), im(im) {}

    PBRT_CPU_GPU complex operator-() const { return {-re, -im}; }

    PBRT_CPU_GPU complex operator+(complex z) const { return {re + z.re, im + z.im}; }

    PBRT_CPU_GPU complex operator-(complex z) const { return {re - z.re, im - z.im}; }

    PBRT_CPU_GPU complex operator*(complex z) const {
        return {re * z.re - im * z.im, re * z.im + im * z.re};
    }

    PBRT_CPU_GPU complex operator/(complex z) const {
        T scale = 1 / (z.re * z.re + z.im * z.im);
        return {scale * (re * z.re + im * z.im), scale * (im * z.re - re * z.im)};
    }

    friend PBRT_CPU_GPU complex operator+(T value, complex z) {
        return complex(value) + z;
    }

    friend PBRT_CPU_GPU complex operator-(T value, complex z) {
        return complex(value) - z;
    }

    friend PBRT_CPU_GPU complex operator*(T value, complex z) {
        return complex(value) * z;
    }

    friend PBRT_CPU_GPU complex operator/(T value, complex z) {
        return complex(value) / z;
    }

    T re, im;
};

PBRT_CPU_GPU inline float sqrt(float f) {
    return ::sqrtf(f);
}
PBRT_CPU_GPU inline double sqrt(double f) {
    return ::sqrt(f);
}
PBRT_CPU_GPU inline float abs(float f) {
    return ::fabsf(f);
}
PBRT_CPU_GPU inline double abs(double f) {
    return ::fabs(f);
}

PBRT_CPU_GPU inline float copysign(float mag, float sign) {
#ifdef PBRT_IS_GPU_CODE
    return ::copysignf(mag, sign);
#else
    return std::copysign(mag, sign);
#endif
}

PBRT_CPU_GPU inline double copysign(double mag, double sign) {
#ifdef PBRT_IS_GPU_CODE
    return ::copysign(mag, sign);
#else
    return std::copysign(mag, sign);
#endif
}

PBRT_CPU_GPU inline float floor(float arg) {
#ifdef PBRT_IS_GPU_CODE
    return ::floorf(arg);
#else
    return std::floor(arg);
#endif
}

PBRT_CPU_GPU inline double floor(double arg) {
#ifdef PBRT_IS_GPU_CODE
    return ::floor(arg);
#else
    return std::floor(arg);
#endif
}

PBRT_CPU_GPU inline float ceil(float arg) {
#ifdef PBRT_IS_GPU_CODE
    return ::ceilf(arg);
#else
    return std::ceil(arg);
#endif
}

PBRT_CPU_GPU inline double ceil(double arg) {
#ifdef PBRT_IS_GPU_CODE
    return ::ceil(arg);
#else
    return std::ceil(arg);
#endif
}

template <typename T>
PBRT_CPU_GPU T real(const complex<T> &z) {
    return z.re;
}

template <typename T>
PBRT_CPU_GPU T imag(const complex<T> &z) {
    return z.im;
}

template <typename T>
PBRT_CPU_GPU T norm(const complex<T> &z) {
    return z.re * z.re + z.im * z.im;
}

template <typename T>
PBRT_CPU_GPU T abs(const complex<T> &z) {
    return pstd::sqrt(pstd::norm(z));
}

template <typename T>
PBRT_CPU_GPU complex<T> sqrt(const complex<T> &z) {
    T n = pstd::abs(z), t1 = pstd::sqrt(T(.5) * (n + pstd::abs(z.re))),
      t2 = T(.5) * z.im / t1;

    if (n == 0)
        return 0;

    if (z.re >= 0)
        return {t1, t2};
    else
        return {pstd::abs(t2), pstd::copysign(t1, z.im)};
}

}  // namespace pstd

#endif  // PBRT_UTIL_PSTD_H
