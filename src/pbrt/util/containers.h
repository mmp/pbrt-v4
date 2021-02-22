// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_CONTAINERS_H
#define PBRT_UTIL_CONTAINERS_H

#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>
#include <pbrt/util/print.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/vecmath.h>

#include <algorithm>
#include <cstring>
#include <iterator>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>

namespace pbrt {

// TypePack Definition
template <typename... Ts>
struct TypePack {
    static constexpr size_t count = sizeof...(Ts);
};

// TypePack Operations
template <typename T, typename... Ts>
struct IndexOf {
    static constexpr int count = 0;
    static_assert(!std::is_same_v<T, T>, "Type not present in TypePack");
};

template <typename T, typename... Ts>
struct IndexOf<T, TypePack<T, Ts...>> {
    static constexpr int count = 0;
};

template <typename T, typename U, typename... Ts>
struct IndexOf<T, TypePack<U, Ts...>> {
    static constexpr int count = 1 + IndexOf<T, TypePack<Ts...>>::count;
};

template <typename T, typename... Ts>
struct HasType {
    static constexpr bool value = false;
};

template <typename T, typename Tfirst, typename... Ts>
struct HasType<T, TypePack<Tfirst, Ts...>> {
    static constexpr bool value =
        (std::is_same<T, Tfirst>::value || HasType<T, TypePack<Ts...>>::value);
};

template <typename T>
struct GetFirst {};
template <typename T, typename... Ts>
struct GetFirst<TypePack<T, Ts...>> {
    using type = T;
};

template <typename T>
struct RemoveFirst {};
template <typename T, typename... Ts>
struct RemoveFirst<TypePack<T, Ts...>> {
    using type = TypePack<Ts...>;
};

template <int index, typename T, typename... Ts>
struct RemoveFirstN;
template <int index, typename T, typename... Ts>
struct RemoveFirstN<index, TypePack<T, Ts...>> {
    using type = typename RemoveFirstN<index - 1, TypePack<Ts...>>::type;
};

template <typename T, typename... Ts>
struct RemoveFirstN<0, TypePack<T, Ts...>> {
    using type = TypePack<T, Ts...>;
};

template <typename... Ts>
struct Prepend;
template <typename T, typename... Ts>
struct Prepend<T, TypePack<Ts...>> {
    using type = TypePack<T, Ts...>;
};
template <typename... Ts>
struct Prepend<void, TypePack<Ts...>> {
    using type = TypePack<Ts...>;
};

template <int index, typename T, typename... Ts>
struct TakeFirstN;
template <int index, typename T, typename... Ts>
struct TakeFirstN<index, TypePack<T, Ts...>> {
    using type =
        typename Prepend<T, typename TakeFirstN<index - 1, TypePack<Ts...>>::type>::type;
};
template <typename T, typename... Ts>
struct TakeFirstN<1, TypePack<T, Ts...>> {
    using type = TypePack<T>;
};

template <template <typename> class M, typename... Ts>
struct MapType;
template <template <typename> class M, typename T>
struct MapType<M, TypePack<T>> {
    using type = TypePack<M<T>>;
};

template <template <typename> class M, typename T, typename... Ts>
struct MapType<M, TypePack<T, Ts...>> {
    using type = typename Prepend<M<T>, typename MapType<M, TypePack<Ts...>>::type>::type;
};

template <template <typename> class Pred, typename... Ts>
struct FilterTypes;

namespace internal {

template <typename T, bool>
struct FilterTypesHelper;

template <typename T>
struct FilterTypesHelper<T, true> {
    using type = T;
};
template <typename T>
struct FilterTypesHelper<T, false> {
    using type = void;
};

};  // namespace internal

template <template <typename> class Pred, typename T>
struct FilterTypes<Pred, TypePack<T>> {
    using type = typename TypePack<
        typename internal::FilterTypesHelper<T, Pred<T>::value>::type>::type;
};

template <template <typename> class Pred, typename T, typename... Ts>
struct FilterTypes<Pred, TypePack<T, Ts...>> {
    using type =
        typename Prepend<typename internal::FilterTypesHelper<T, Pred<T>::value>::type,
                         TypePack<Ts...>>::type;
};

template <typename F, typename... Ts>
void ForEachType(F func, TypePack<Ts...>);
template <typename F, typename T, typename... Ts>
void ForEachType(F func, TypePack<T, Ts...>) {
    func.template operator()<T>();
    ForEachType(func, TypePack<Ts...>());
}

template <typename F>
void ForEachType(F func, TypePack<>) {}

// Array2D Definition
template <typename T>
class Array2D {
  public:
    // Array2D Type Definitions
    using value_type = T;
    using iterator = value_type *;
    using const_iterator = const value_type *;
    using allocator_type = pstd::pmr::polymorphic_allocator<std::byte>;

    // Array2D Public Methods
    Array2D(allocator_type allocator = {}) : Array2D({{0, 0}, {0, 0}}, allocator) {}

    Array2D(const Bounds2i &extent, Allocator allocator = {})
        : extent(extent), allocator(allocator) {
        int n = extent.Area();
        values = allocator.allocate_object<T>(n);
        for (int i = 0; i < n; ++i)
            allocator.construct(values + i);
    }

    Array2D(const Bounds2i &extent, T def, allocator_type allocator = {})
        : Array2D(extent, allocator) {
        std::fill(begin(), end(), def);
    }
    template <typename InputIt,
              typename = typename std::enable_if_t<
                  !std::is_integral<InputIt>::value &&
                  std::is_base_of<
                      std::input_iterator_tag,
                      typename std::iterator_traits<InputIt>::iterator_category>::value>>
    Array2D(InputIt first, InputIt last, int nx, int ny, allocator_type allocator = {})
        : Array2D({{0, 0}, {nx, ny}}, allocator) {
        std::copy(first, last, begin());
    }
    Array2D(int nx, int ny, allocator_type allocator = {})
        : Array2D({{0, 0}, {nx, ny}}, allocator) {}
    Array2D(int nx, int ny, T def, allocator_type allocator = {})
        : Array2D({{0, 0}, {nx, ny}}, def, allocator) {}
    Array2D(const Array2D &a, allocator_type allocator = {})
        : Array2D(a.begin(), a.end(), a.xSize(), a.ySize(), allocator) {}

    ~Array2D() {
        int n = extent.Area();
        for (int i = 0; i < n; ++i)
            allocator.destroy(values + i);
        allocator.deallocate_object(values, n);
    }

    Array2D(Array2D &&a, allocator_type allocator = {})
        : extent(a.extent), allocator(allocator) {
        if (allocator == a.allocator) {
            values = a.values;
            a.extent = Bounds2i({0, 0}, {0, 0});
            a.values = nullptr;
        } else {
            values = allocator.allocate_object<T>(extent.Area());
            std::copy(a.begin(), a.end(), begin());
        }
    }
    Array2D &operator=(const Array2D &a) = delete;

    Array2D &operator=(Array2D &&other) {
        if (allocator == other.allocator) {
            pstd::swap(extent, other.extent);
            pstd::swap(values, other.values);
        } else if (extent == other.extent) {
            int n = extent.Area();
            for (int i = 0; i < n; ++i) {
                allocator.destroy(values + i);
                allocator.construct(values + i, other.values[i]);
            }
            extent = other.extent;
        } else {
            int n = extent.Area();
            for (int i = 0; i < n; ++i)
                allocator.destroy(values + i);
            allocator.deallocate_object(values, n);

            int no = other.extent.Area();
            values = allocator.allocate_object<T>(no);
            for (int i = 0; i < no; ++i)
                allocator.construct(values + i, other.values[i]);
        }
        return *this;
    }

    PBRT_CPU_GPU
    T &operator[](Point2i p) {
        DCHECK(InsideExclusive(p, extent));
        p.x -= extent.pMin.x;
        p.y -= extent.pMin.y;
        return values[p.x + (extent.pMax.x - extent.pMin.x) * p.y];
    }
    PBRT_CPU_GPU T &operator()(int x, int y) { return (*this)[{x, y}]; }

    PBRT_CPU_GPU
    const T &operator()(int x, int y) const { return (*this)[{x, y}]; }
    PBRT_CPU_GPU
    const T &operator[](Point2i p) const {
        DCHECK(InsideExclusive(p, extent));
        p.x -= extent.pMin.x;
        p.y -= extent.pMin.y;
        return values[p.x + (extent.pMax.x - extent.pMin.x) * p.y];
    }

    PBRT_CPU_GPU
    int size() const { return extent.Area(); }
    PBRT_CPU_GPU
    int xSize() const { return extent.pMax.x - extent.pMin.x; }
    PBRT_CPU_GPU
    int ySize() const { return extent.pMax.y - extent.pMin.y; }

    PBRT_CPU_GPU
    iterator begin() { return values; }
    PBRT_CPU_GPU
    iterator end() { return begin() + size(); }

    PBRT_CPU_GPU
    const_iterator begin() const { return values; }
    PBRT_CPU_GPU
    const_iterator end() const { return begin() + size(); }

    PBRT_CPU_GPU
    operator pstd::span<T>() { return pstd::span<T>(values, size()); }
    PBRT_CPU_GPU
    operator pstd::span<const T>() const { return pstd::span<const T>(values, size()); }

    std::string ToString() const {
        std::string s = StringPrintf("[ Array2D extent: %s values: [", extent);
        for (int y = extent.pMin.y; y < extent.pMax.y; ++y) {
            s += " [ ";
            for (int x = extent.pMin.x; x < extent.pMax.x; ++x) {
                T value = (*this)(x, y);
                s += StringPrintf("%s, ", value);
            }
            s += "], ";
        }
        s += " ] ]";
        return s;
    }

  private:
    // Array2D Private Members
    Bounds2i extent;
    Allocator allocator;
    T *values;
};

template <typename T, int N, class Allocator = pstd::pmr::polymorphic_allocator<T>>
class InlinedVector {
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
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    InlinedVector(const Allocator &alloc = {}) : alloc(alloc) {}
    InlinedVector(size_t count, const T &value, const Allocator &alloc = {})
        : alloc(alloc) {
        reserve(count);
        for (size_t i = 0; i < count; ++i)
            this->alloc.template construct<T>(begin() + i, value);
        nStored = count;
    }
    InlinedVector(size_t count, const Allocator &alloc = {})
        : InlinedVector(count, T{}, alloc) {}
    InlinedVector(const InlinedVector &other, const Allocator &alloc = {})
        : alloc(alloc) {
        reserve(other.size());
        for (size_t i = 0; i < other.size(); ++i)
            this->alloc.template construct<T>(begin() + i, other[i]);
        nStored = other.size();
    }
    template <class InputIt>
    InlinedVector(InputIt first, InputIt last, const Allocator &alloc = {})
        : alloc(alloc) {
        reserve(last - first);
        for (InputIt iter = first; iter != last; ++iter, ++nStored)
            this->alloc.template construct<T>(begin() + nStored, *iter);
    }
    InlinedVector(InlinedVector &&other) : alloc(other.alloc) {
        nStored = other.nStored;
        nAlloc = other.nAlloc;
        ptr = other.ptr;
        if (other.nStored <= N)
            for (int i = 0; i < other.nStored; ++i)
                alloc.template construct<T>(fixed + i, std::move(other.fixed[i]));
        // Leave other.nStored as is, so that the detrius left after we
        // moved out of fixed has its destructors run...
        else
            other.nStored = 0;

        other.nAlloc = 0;
        other.ptr = nullptr;
    }
    InlinedVector(InlinedVector &&other, const Allocator &alloc) {
        LOG_FATAL("TODO");

        if (alloc == other.alloc) {
            ptr = other.ptr;
            nAlloc = other.nAlloc;
            nStored = other.nStored;
            if (other.nStored <= N)
                for (int i = 0; i < other.nStored; ++i)
                    fixed[i] = std::move(other.fixed[i]);

            other.ptr = nullptr;
            other.nAlloc = other.nStored = 0;
        } else {
            reserve(other.size());
            for (size_t i = 0; i < other.size(); ++i)
                alloc.template construct<T>(begin() + i, std::move(other[i]));
            nStored = other.size();
        }
    }
    InlinedVector(std::initializer_list<T> init, const Allocator &alloc = {})
        : InlinedVector(init.begin(), init.end(), alloc) {}

    InlinedVector &operator=(const InlinedVector &other) {
        if (this == &other)
            return *this;

        clear();
        reserve(other.size());
        for (size_t i = 0; i < other.size(); ++i)
            alloc.template construct<T>(begin() + i, other[i]);
        nStored = other.size();

        return *this;
    }
    InlinedVector &operator=(InlinedVector &&other) {
        if (this == &other)
            return *this;

        clear();
        if (alloc == other.alloc) {
            pstd::swap(ptr, other.ptr);
            pstd::swap(nAlloc, other.nAlloc);
            pstd::swap(nStored, other.nStored);
            if (nStored > 0 && ptr == nullptr) {
                for (int i = 0; i < nStored; ++i)
                    alloc.template construct<T>(fixed + i, std::move(other.fixed[i]));
                other.nStored = nStored;  // so that dtors run...
            }
        } else {
            reserve(other.size());
            for (size_t i = 0; i < other.size(); ++i)
                alloc.template construct<T>(begin() + i, std::move(other[i]));
            nStored = other.size();
        }

        return *this;
    }
    InlinedVector &operator=(std::initializer_list<T> &init) {
        clear();
        reserve(init.size());
        for (const auto &value : init) {
            alloc.template construct<T>(begin() + nStored, value);
            ++nStored;
        }
        return *this;
    }

    void assign(size_type count, const T &value) {
        clear();
        reserve(count);
        for (size_t i = 0; i < count; ++i)
            alloc.template construct<T>(begin() + i, value);
        nStored = count;
    }
    template <class InputIt>
    void assign(InputIt first, InputIt last) {
        // TODO
        LOG_FATAL("TODO");
    }
    void assign(std::initializer_list<T> &init) { assign(init.begin(), init.end()); }

    ~InlinedVector() {
        clear();
        alloc.deallocate_object(ptr, nAlloc);
    }

    PBRT_CPU_GPU
    iterator begin() { return ptr ? ptr : fixed; }
    PBRT_CPU_GPU
    iterator end() { return begin() + nStored; }
    PBRT_CPU_GPU
    const_iterator begin() const { return ptr ? ptr : fixed; }
    PBRT_CPU_GPU
    const_iterator end() const { return begin() + nStored; }
    PBRT_CPU_GPU
    const_iterator cbegin() const { return ptr ? ptr : fixed; }
    PBRT_CPU_GPU
    const_iterator cend() const { return begin() + nStored; }

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
    size_t capacity() const { return ptr ? nAlloc : N; }

    void reserve(size_t n) {
        if (capacity() >= n)
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
        return begin()[index];
    }
    PBRT_CPU_GPU
    const_reference operator[](size_type index) const {
        DCHECK_LT(index, size());
        return begin()[index];
    }
    PBRT_CPU_GPU
    reference front() { return *begin(); }
    PBRT_CPU_GPU
    const_reference front() const { return *begin(); }
    PBRT_CPU_GPU
    reference back() { return *(begin() + nStored - 1); }
    PBRT_CPU_GPU
    const_reference back() const { return *(begin() + nStored - 1); }
    PBRT_CPU_GPU
    pointer data() { return ptr ? ptr : fixed; }
    PBRT_CPU_GPU
    const_pointer data() const { return ptr ? ptr : fixed; }

    void clear() {
        for (int i = 0; i < nStored; ++i)
            alloc.destroy(begin() + i);
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
            reserve(size() + (last - first));
            iterator pos = end(), startPos = end();
            for (auto iter = first; iter != last; ++iter, ++pos)
                alloc.template construct<T>(pos, *iter);
            nStored += last - first;
            return pos;
        } else {
            // TODO
            LOG_FATAL("TODO");
        }
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
        // TODO
        LOG_FATAL("TODO");
    }

    iterator erase(const_iterator cpos) {
        iterator pos =
            begin() + (cpos - begin());  // non-const iterator, thank you very much
        while (pos != end() - 1) {
            *pos = std::move(*(pos + 1));
            ++pos;
        }
        alloc.destroy(pos);
        --nStored;
        return begin() + (cpos - begin());
    }
    iterator erase(const_iterator first, const_iterator last) {
        // TODO
        LOG_FATAL("TODO");
    }

    void push_back(const T &value) {
        if (size() == capacity())
            reserve(2 * capacity());

        alloc.construct(begin() + nStored, value);
        ++nStored;
    }
    void push_back(T &&value) {
        if (size() == capacity())
            reserve(2 * capacity());

        alloc.construct(begin() + nStored, std::move(value));
        ++nStored;
    }
    void pop_back() {
        DCHECK(!empty());
        alloc.destroy(begin() + nStored - 1);
        --nStored;
    }

    void resize(size_type n) {
        if (n < size()) {
            for (size_t i = n; n < size(); ++i)
                alloc.destroy(begin() + i);
        } else if (n > size()) {
            reserve(n);
            for (size_t i = nStored; i < n; ++i)
                alloc.construct(begin() + i);
        }
        nStored = n;
    }
    void resize(size_type count, const value_type &value) {
        // TODO
        LOG_FATAL("TODO");
    }

    void swap(InlinedVector &other) {
        // TODO
        LOG_FATAL("TODO");
    }

  private:
    Allocator alloc;
    // ptr non-null is discriminator for whether fixed[] is valid...
    T *ptr = nullptr;
    union {
        T fixed[N];
    };
    size_t nAlloc = 0, nStored = 0;
};

// HashMap Definition
template <typename Key, typename Value, typename Hash,
          typename Allocator =
              pstd::pmr::polymorphic_allocator<pstd::optional<std::pair<Key, Value>>>>
class HashMap {
  public:
    // HashMap Type Definitions
    using TableEntry = pstd::optional<std::pair<Key, Value>>;

    class Iterator {
      public:
        PBRT_CPU_GPU
        Iterator &operator++() {
            while (++ptr < end && !ptr->has_value())
                ;
            return *this;
        }

        PBRT_CPU_GPU
        Iterator operator++(int) {
            Iterator old = *this;
            operator++();
            return old;
        }

        PBRT_CPU_GPU
        bool operator==(const Iterator &iter) const { return ptr == iter.ptr; }
        PBRT_CPU_GPU
        bool operator!=(const Iterator &iter) const { return ptr != iter.ptr; }

        PBRT_CPU_GPU
        std::pair<Key, Value> &operator*() { return ptr->value(); }
        PBRT_CPU_GPU
        const std::pair<Key, Value> &operator*() const { return ptr->value(); }

        PBRT_CPU_GPU
        std::pair<Key, Value> *operator->() { return &ptr->value(); }
        PBRT_CPU_GPU
        const std::pair<Key, Value> *operator->() const { return ptr->value(); }

      private:
        friend class HashMap;
        Iterator(TableEntry *ptr, TableEntry *end) : ptr(ptr), end(end) {}
        TableEntry *ptr;
        TableEntry *end;
    };

    using iterator = Iterator;
    using const_iterator = const iterator;

    // HashMap Public Methods
    PBRT_CPU_GPU
    size_t size() const { return nStored; }
    PBRT_CPU_GPU
    size_t capacity() const { return table.size(); }
    void Clear() {
        table.clear();
        nStored = 0;
    }

    HashMap(Allocator alloc) : table(8, alloc) {}

    HashMap(const HashMap &) = delete;
    HashMap &operator=(const HashMap &) = delete;

    void Insert(const Key &key, const Value &value) {
        size_t offset = FindOffset(key);
        if (table[offset].has_value() == false) {
            // Grow hash table if it is too full
            if (3 * ++nStored > capacity()) {
                Grow();
                offset = FindOffset(key);
            }
        }
        table[offset] = std::make_pair(key, value);
    }

    PBRT_CPU_GPU
    bool HasKey(const Key &key) const { return table[FindOffset(key)].has_value(); }

    PBRT_CPU_GPU
    const Value &operator[](const Key &key) const {
        size_t offset = FindOffset(key);
        CHECK(table[offset].has_value());
        return table[offset]->second;
    }

    PBRT_CPU_GPU
    iterator begin() {
        Iterator iter(table.data(), table.data() + capacity());
        while (iter.ptr < iter.end && !iter.ptr->has_value())
            ++iter.ptr;
        return iter;
    }
    PBRT_CPU_GPU
    iterator end() {
        return Iterator(table.data() + capacity(), table.data() + capacity());
    }

  private:
    // HashMap Private Methods
    PBRT_CPU_GPU
    size_t FindOffset(const Key &key) const {
        size_t baseOffset = Hash()(key) & (capacity() - 1);
        for (int nProbes = 0;; ++nProbes) {
            // Find offset for _key_ using quadratic probing
            size_t offset =
                (baseOffset + nProbes / 2 + nProbes * nProbes / 2) & (capacity() - 1);
            if (table[offset].has_value() == false || key == table[offset]->first)
                return offset;
        }
    }

    void Grow() {
        size_t currentCapacity = capacity();
        pstd::vector<TableEntry> newTable(std::max<size_t>(64, 2 * currentCapacity),
                                          table.get_allocator());
        size_t newCapacity = newTable.size();
        for (size_t i = 0; i < currentCapacity; ++i) {
            // Insert _table[i]_ into _newTable_ if it is set
            if (!table[i].has_value())
                continue;
            size_t baseOffset = Hash()(table[i]->first) & (newCapacity - 1);
            for (int nProbes = 0;; ++nProbes) {
                size_t offset = (baseOffset + nProbes / 2 + nProbes * nProbes / 2) &
                                (newCapacity - 1);
                if (!newTable[offset]) {
                    newTable[offset] = std::move(*table[i]);
                    break;
                }
            }
        }
        table = std::move(newTable);
    }

    // HashMap Private Members
    pstd::vector<TableEntry> table;
    size_t nStored = 0;
};

// SampledGrid Definition
template <typename T>
class SampledGrid {
  public:
    using const_iterator = typename pstd::vector<T>::const_iterator;
    // SampledGrid Public Methods
    SampledGrid() = default;
    SampledGrid(Allocator alloc) : values(alloc) {}
    SampledGrid(pstd::span<const T> v, int nx, int ny, int nz, Allocator alloc)
        : values(v.begin(), v.end(), alloc), nx(nx), ny(ny), nz(nz) {
        CHECK_EQ(nx * ny * nz, values.size());
    }

    size_t BytesAllocated() const { return values.size() * sizeof(T); }
    int xSize() const { return nx; }
    int ySize() const { return ny; }
    int zSize() const { return nz; }

    const_iterator begin() const { return values.begin(); }
    const_iterator end() const { return values.end(); }

    template <typename F>
    PBRT_CPU_GPU auto Lookup(const Point3f &p, F convert) const {
        // Compute voxel coordinates and offsets for _p_
        Point3f pSamples(p.x * nx - .5f, p.y * ny - .5f, p.z * nz - .5f);
        Point3i pi = (Point3i)Floor(pSamples);
        Vector3f d = pSamples - (Point3f)pi;

        // Return trilinearly interpolated voxel values
        auto d00 =
            Lerp(d.x, Lookup(pi, convert), Lookup(pi + Vector3i(1, 0, 0), convert));
        auto d10 = Lerp(d.x, Lookup(pi + Vector3i(0, 1, 0), convert),
                        Lookup(pi + Vector3i(1, 1, 0), convert));
        auto d01 = Lerp(d.x, Lookup(pi + Vector3i(0, 0, 1), convert),
                        Lookup(pi + Vector3i(1, 0, 1), convert));
        auto d11 = Lerp(d.x, Lookup(pi + Vector3i(0, 1, 1), convert),
                        Lookup(pi + Vector3i(1, 1, 1), convert));
        return Lerp(d.z, Lerp(d.y, d00, d10), Lerp(d.y, d01, d11));
    }

    PBRT_CPU_GPU
    T Lookup(const Point3f &p) const {
        return Lookup(p, [] PBRT_CPU_GPU(T value) { return value; });
    }

    template <typename F>
    PBRT_CPU_GPU auto Lookup(const Point3i &p, F convert) const {
        Bounds3i sampleBounds(Point3i(0, 0, 0), Point3i(nx, ny, nz));
        if (!InsideExclusive(p, sampleBounds))
            return convert(T{});
        return convert(values[(p.z * ny + p.y) * nx + p.x]);
    }

    PBRT_CPU_GPU
    T Lookup(const Point3i &p) const {
        return Lookup(p, [] PBRT_CPU_GPU(T value) { return value; });
    }

    template <typename F>
    Float MaxValue(const Bounds3f &bounds, F convert) const {
        Point3f ps[2] = {Point3f(bounds.pMin.x * nx - .5f, bounds.pMin.y * ny - .5f,
                                 bounds.pMin.z * nz - .5f),
                         Point3f(bounds.pMax.x * nx - .5f, bounds.pMax.y * ny - .5f,
                                 bounds.pMax.z * nz - .5f)};
        Point3i pi[2] = {Max(Point3i(Floor(ps[0])), Point3i(0, 0, 0)),
                         Min(Point3i(Floor(ps[1])) + Vector3i(1, 1, 1),
                             Point3i(nx - 1, ny - 1, nz - 1))};

        Float maxValue = Lookup(Point3i(pi[0]), convert);
        for (int z = pi[0].z; z <= pi[1].z; ++z)
            for (int y = pi[0].y; y <= pi[1].y; ++y)
                for (int x = pi[0].x; x <= pi[1].x; ++x)
                    maxValue = std::max(maxValue, Lookup(Point3i(x, y, z), convert));

        return maxValue;
    }

    T MaxValue(const Bounds3f &bounds) const {
        return MaxValue(bounds, [](T value) { return value; });
    }

    std::string ToString() const {
        return StringPrintf("[ SampledGrid nx: %d ny: %d nz: %d values: %s ]", nx, ny, nz,
                            values);
    }

  private:
    // SampledGrid Private Members
    pstd::vector<T> values;
    int nx, ny, nz;
};

}  // namespace pbrt

#endif  // PBRT_UTIL_CONTAINERS_H
