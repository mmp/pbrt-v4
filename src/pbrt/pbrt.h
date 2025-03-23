// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_PBRT_H
#define PBRT_PBRT_H

#include <stdint.h>
#include <cstddef>

#ifdef PBRT_IS_WINDOWS
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define UNICODE
#endif
#endif  // PBRT_IS_WINDOWS

// GPU Macro Definitions
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define PBRT_IS_GPU_CODE
#endif

#if defined(PBRT_BUILD_GPU_RENDERER) && (defined(__CUDACC__) || defined(__HIPCC__))
#ifndef PBRT_NOINLINE
#define PBRT_NOINLINE __attribute__((noinline))
#endif
#define PBRT_CPU_GPU __host__ __device__
#define PBRT_GPU __device__
#if defined(PBRT_IS_GPU_CODE)
#define PBRT_CONST __device__ const
#else
#define PBRT_CONST const
#endif
#else
#define PBRT_CONST const
#define PBRT_CPU_GPU
#define PBRT_GPU
#endif

#ifdef PBRT_IS_WINDOWS
#define PBRT_CPU_GPU_LAMBDA(...) [ =, *this ] PBRT_CPU_GPU(__VA_ARGS__) mutable
#else
#define PBRT_CPU_GPU_LAMBDA(...) [=] PBRT_CPU_GPU(__VA_ARGS__)
#endif

// Define Cache Line Size Constant
#ifdef PBRT_BUILD_GPU_RENDERER
#define PBRT_L1_CACHE_LINE_SIZE 128
#else
#define PBRT_L1_CACHE_LINE_SIZE 64
#endif

#ifdef PBRT_DBG_LOGGING
#include <stdio.h>
#ifndef PBRT_TO_STRING
#define PBRT_TO_STRING(x) PBRT_TO_STRING2(x)
#define PBRT_TO_STRING2(x) #x
#endif  // !PBRT_TO_STRING
#ifdef PBRT_IS_GPU_CODE
#define PBRT_DBG(...) printf(__FILE__ ":" TO_STRING(__LINE__) ": " __VA_ARGS__)
#else
#define PBRT_DBG(...) fprintf(stderr, __FILE__ ":" TO_STRING(__LINE__) ": " __VA_ARGS__)
#endif  // PBRT_IS_GPU_CODE
#else
#define PBRT_DBG(...)
#endif  // PBRT_DBG_LOGGING

// From ABSL_ARRAYSIZE
#define PBRT_ARRAYSIZE(array) (sizeof(::pbrt::detail::ArraySizeHelper(array)))

namespace pbrt {
namespace detail {

template <typename T, uint64_t N>
auto ArraySizeHelper(const T (&array)[N]) -> char (&)[N];

}  // namespace detail
}  // namespace pbrt

namespace pstd {

namespace pmr {
template <typename T>
class polymorphic_allocator;
}

}  // namespace pstd

namespace pbrt {

// Float Type Definitions
#ifdef PBRT_FLOAT_AS_DOUBLE
using Float = double;
#else
using Float = float;
#endif

#ifdef PBRT_FLOAT_AS_DOUBLE
using FloatBits = uint64_t;
#else
using FloatBits = uint32_t;
#endif  // PBRT_FLOAT_AS_DOUBLE
static_assert(sizeof(Float) == sizeof(FloatBits),
              "Float and FloatBits must have the same size");

template <typename T>
class Vector2;
template <typename T>
class Vector3;
template <typename T>
class Point3;
template <typename T>
class Point2;
template <typename T>
class Normal3;
using Point2f = Point2<Float>;
using Point2i = Point2<int>;
using Point3f = Point3<Float>;
using Vector2f = Vector2<Float>;
using Vector2i = Vector2<int>;
using Vector3f = Vector3<Float>;

template <typename T>
class Bounds2;
using Bounds2f = Bounds2<Float>;
using Bounds2i = Bounds2<int>;
template <typename T>
class Bounds3;
using Bounds3f = Bounds3<Float>;
using Bounds3i = Bounds3<int>;

class AnimatedTransform;
class BilinearPatchMesh;
class Interaction;
class Interaction;
class MediumInteraction;
class Ray;
class RayDifferential;
class SurfaceInteraction;
class SurfaceInteraction;
class Transform;
class TriangleMesh;

class RGB;
class RGBColorSpace;
class RGBSigmoidPolynomial;
class RGBIlluminantSpectrum;
class SampledSpectrum;
class SampledWavelengths;
class SpectrumWavelengths;
class XYZ;
enum class SpectrumType;

class BSDF;
class CameraTransform;
class Image;
class ParameterDictionary;
struct NamedTextures;
class TextureParameterDictionary;
struct ImageMetadata;
struct MediumInterface;
struct PBRTOptions;

class PiecewiseConstant1D;
class PiecewiseConstant2D;
class ProgressReporter;
class RNG;
struct FileLoc;
class Interval;
template <typename T>
class Array2D;

template <typename T>
struct SOA;
class ScratchBuffer;

// Define _Allocator_
using Allocator = pstd::pmr::polymorphic_allocator<std::byte>;

// Initialization and Cleanup Function Declarations
void InitPBRT(const PBRTOptions &opt);
void CleanupPBRT();

}  // namespace pbrt

#endif  // PBRT_PBRT_H
