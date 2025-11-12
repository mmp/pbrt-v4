#pragma once

/***
 * Structs to hold train data inputs/outputs for my Neural Network
 *
 *
 *
 *
 */
#include <pbrt/pbrt.h>
#include <pbrt/ray.h>



#ifdef PBRT_BUILD_GPU_RENDERER
struct InputRayDataGPU
{
    RayDifferential ray;
};


struct OutputRayDataGPU
{
    RayDifferential ray;
    Point3f position;
    SampledSpectrum Li;


    // Considering if this is required
    SampledSpectrum sigma;
};

#endif
using namespace pbrt;


struct InputRayData
{
    RayDifferential rayd;
    Ray ray;

    int pixelIdx;
    int sampleIdx;
    int depth;

    bool HasDifferentials() const { return rayd.hasDifferentials; }
};


struct OutputRayData
{
    Ray ray;
    Point3f position;
    SampledSpectrum Li;

    int pixelIdx;
    int sampleIdx;
    int depth;

    // Considering if this is required
    SampledSpectrum sigma;

};