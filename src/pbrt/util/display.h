// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_DISPLAY_H
#define PBRT_UTIL_DISPLAY_H

#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>
#include <pbrt/util/color.h>
#include <pbrt/util/containers.h>
#include <pbrt/util/image.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/vecmath.h>

#include <functional>
#include <string>

namespace pbrt {

// DisplayServer Function Declarations
void ConnectToDisplayServer(const std::string &host);
void DisconnectFromDisplayServer();

void DisplayStatic(
    const std::string &title, const Point2i &resolution,
    std::vector<std::string> channelNames,
    std::function<void(Bounds2i, pstd::span<pstd::span<Float>>)> getValues);

void DisplayDynamic(
    const std::string &title, const Point2i &resolution,
    std::vector<std::string> channelNames,
    std::function<void(Bounds2i, pstd::span<pstd::span<Float>>)> getValues);

void DisplayStatic(const std::string &title, const Image &image,
                   pstd::optional<ImageChannelDesc> channelDesc = {});
void DisplayDynamic(const std::string &title, const Image &image,
                    pstd::optional<ImageChannelDesc> channelDesc = {});

template <typename T>
inline typename std::enable_if_t<std::is_arithmetic<T>::value, void> DisplayStatic(
    const std::string &title, pstd::span<const T> values, int xResolution) {
    CHECK_EQ(0, values.size() % xResolution);
    int yResolution = values.size() / xResolution;
    DisplayStatic(title, {xResolution, yResolution}, {"value"},
                  [=](Bounds2i b, pstd::span<pstd::span<Float>> displayValue) {
                      DCHECK_EQ(1, displayValue.size());
                      int index = 0;
                      for (Point2i p : b)
                          displayValue[0][index++] = values[p.x + p.y * xResolution];
                  });
}

template <typename T>
inline typename std::enable_if_t<std::is_arithmetic<T>::value, void> DisplayDynamic(
    const std::string &title, pstd::span<const T> values, int xResolution) {
    CHECK_EQ(0, values.size() % xResolution);
    int yResolution = values.size() / xResolution;
    DisplayDynamic(title, {xResolution, yResolution}, {"value"},
                   [=](Bounds2i b, pstd::span<pstd::span<Float>> displayValue) {
                       DCHECK_EQ(1, displayValue.size());
                       int index = 0;
                       for (Point2i p : b)
                           displayValue[0][index++] = values[p.x + p.y * xResolution];
                   });
}

namespace detail {

// https://stackoverflow.com/a/31306194
// base case
template <typename...>
using void_t = void;

template <class T, class Index, typename = void>
struct has_subscript_operator : std::false_type {};

template <class T, class Index>
struct has_subscript_operator<T, Index,
                              void_t<decltype(std::declval<T>()[std::declval<Index>()])>>
    : std::true_type {};

template <class T, class Index>
using has_subscript_operator_t = typename has_subscript_operator<T, Index>::type;

}  // namespace detail

template <typename T>
inline typename std::enable_if_t<detail::has_subscript_operator_t<T, int>::value, void>
DisplayStatic(const std::string &title, pstd::span<const T> values,
              const std::vector<std::string> &channelNames, int xResolution) {
    CHECK_EQ(0, values.size() % xResolution);
    int yResolution = values.size() / xResolution;
    DisplayStatic(title, {xResolution, yResolution}, channelNames,
                  [=](Bounds2i b, pstd::span<pstd::span<Float>> displayValue) {
                      DCHECK_EQ(channelNames.size(), displayValue.size());
                      int index = 0;
                      for (Point2i p : b) {
                          int offset = p.x + p.y * xResolution;
                          for (int i = 0; i < channelNames.size(); ++i)
                              displayValue[i][index] = values[offset][i];
                          ++index;
                      }
                  });
}

template <typename T>
inline typename std::enable_if_t<detail::has_subscript_operator_t<T, int>::value, void>
DisplayDynamic(const std::string &title, pstd::span<const T> values,
               const std::vector<std::string> &channelNames, int xResolution) {
    CHECK_EQ(0, values.size() % xResolution);
    int yResolution = values.size() / xResolution;
    DisplayDynamic(title, {xResolution, yResolution}, channelNames,
                   [=](Bounds2i b, pstd::span<pstd::span<Float>> displayValue) {
                       DCHECK_EQ(channelNames.size(), displayValue.size());
                       int index = 0;
                       for (Point2i p : b) {
                           int offset = p.x + p.y * xResolution;
                           for (int i = 0; i < channelNames.size(); ++i)
                               displayValue[i][index] = values[offset][i];
                           ++index;
                       }
                   });
}

// TODO: Array2D equivalents

}  // namespace pbrt

#endif  // PBRT_UTIL_DISPLAY_H
