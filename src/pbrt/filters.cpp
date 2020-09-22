// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/filters.h>

#include <pbrt/paramdict.h>
#include <pbrt/util/print.h>
#include <pbrt/util/rng.h>

namespace pbrt {

std::string FilterHandle::ToString() const {
    if (ptr() == nullptr)
        return "(nullptr)";

    auto ts = [&](auto ptr) { return ptr->ToString(); };
    return DispatchCPU(ts);
}

// Box Filter Method Definitions
std::string BoxFilter::ToString() const {
    return StringPrintf("[ BoxFilter radius: %s ]", radius);
}

BoxFilter *BoxFilter::Create(const ParameterDictionary &parameters, const FileLoc *loc,
                             Allocator alloc) {
    Float xw = parameters.GetOneFloat("xradius", 0.5f);
    Float yw = parameters.GetOneFloat("yradius", 0.5f);
    return alloc.new_object<BoxFilter>(Vector2f(xw, yw));
}

// Gaussian Filter Method Definitions
std::string GaussianFilter::ToString() const {
    return StringPrintf(
        "[ GaussianFilter radius: %s sigma: %f expX: %f expY: %f sampler: %s ]", radius,
        sigma, expX, expY, sampler);
}

GaussianFilter *GaussianFilter::Create(const ParameterDictionary &parameters,
                                       const FileLoc *loc, Allocator alloc) {
    // Find common filter parameters
    Float xw = parameters.GetOneFloat("xradius", 1.5f);
    Float yw = parameters.GetOneFloat("yradius", 1.5f);
    Float sigma = parameters.GetOneFloat("sigma", 0.5f);  // equivalent to old alpha = 2
    return alloc.new_object<GaussianFilter>(Vector2f(xw, yw), sigma, alloc);
}

// Mitchell Filter Method Definitions
std::string MitchellFilter::ToString() const {
    return StringPrintf("[ MitchellFilter radius: %s B: %f C: %f sampler: %s ]", radius,
                        B, C, sampler);
}

MitchellFilter *MitchellFilter::Create(const ParameterDictionary &parameters,
                                       const FileLoc *loc, Allocator alloc) {
    // Find common filter parameters
    Float xw = parameters.GetOneFloat("xradius", 2.f);
    Float yw = parameters.GetOneFloat("yradius", 2.f);
    Float B = parameters.GetOneFloat("B", 1.f / 3.f);
    Float C = parameters.GetOneFloat("C", 1.f / 3.f);
    return alloc.new_object<MitchellFilter>(Vector2f(xw, yw), B, C, alloc);
}

// Sinc Filter Method Definitions
Float LanczosSincFilter::Integral() const {
    Float sum = 0;
    int sqrtSamples = 64;
    int nSamples = sqrtSamples * sqrtSamples;
    Float area = 2 * radius.x * 2 * radius.y;
    RNG rng;
    for (int y = 0; y < sqrtSamples; ++y) {
        for (int x = 0; x < sqrtSamples; ++x) {
            Point2f u((x + rng.Uniform<Float>()) / sqrtSamples,
                      (y + rng.Uniform<Float>()) / sqrtSamples);
            Point2f p(Lerp(u.x, -radius.x, radius.x), Lerp(u.y, -radius.y, radius.y));
            sum += Evaluate(p);
        }
    }
    return sum / nSamples * area;
}

std::string LanczosSincFilter::ToString() const {
    return StringPrintf("[ LanczosSincFilter radius: %s tau: %f sampler: %s ]", radius,
                        tau, sampler);
}

LanczosSincFilter *LanczosSincFilter::Create(const ParameterDictionary &parameters,
                                             const FileLoc *loc, Allocator alloc) {
    Float xw = parameters.GetOneFloat("xradius", 4.);
    Float yw = parameters.GetOneFloat("yradius", 4.);
    Float tau = parameters.GetOneFloat("tau", 3.f);
    return alloc.new_object<LanczosSincFilter>(Vector2f(xw, yw), tau, alloc);
}

// Triangle Filter Method Definitions
std::string TriangleFilter::ToString() const {
    return StringPrintf("[ TriangleFilter radius: %s ]", radius);
}

TriangleFilter *TriangleFilter::Create(const ParameterDictionary &parameters,
                                       const FileLoc *loc, Allocator alloc) {
    // Find common filter parameters
    Float xw = parameters.GetOneFloat("xradius", 2.f);
    Float yw = parameters.GetOneFloat("yradius", 2.f);
    return alloc.new_object<TriangleFilter>(Vector2f(xw, yw));
}

FilterHandle FilterHandle::Create(const std::string &name,
                                  const ParameterDictionary &parameters,
                                  const FileLoc *loc, Allocator alloc) {
    FilterHandle filter = nullptr;
    if (name == "box")
        filter = BoxFilter::Create(parameters, loc, alloc);
    else if (name == "gaussian")
        filter = GaussianFilter::Create(parameters, loc, alloc);
    else if (name == "mitchell")
        filter = MitchellFilter::Create(parameters, loc, alloc);
    else if (name == "sinc")
        filter = LanczosSincFilter::Create(parameters, loc, alloc);
    else if (name == "triangle")
        filter = TriangleFilter::Create(parameters, loc, alloc);
    else
        ErrorExit(loc, "%s: filter type unknown.", name);

    if (!filter)
        ErrorExit(loc, "%s: unable to create filter.", name);

    parameters.ReportUnused();
    return filter;
}

// FilterSampler Method Definitions
FilterSampler::FilterSampler(FilterHandle filter, Allocator alloc)
    : domain(Point2f(-filter.Radius()), Point2f(filter.Radius())),
      values(int(32 * filter.Radius().x), int(32 * filter.Radius().y), alloc),
      distrib(alloc) {
    // Tabularize filter function in _values_
    for (int y = 0; y < values.ySize(); ++y)
        for (int x = 0; x < values.xSize(); ++x) {
            Point2f p = domain.Lerp(
                Point2f((x + 0.5f) / values.xSize(), (y + 0.5f) / values.ySize()));
            values(x, y) = filter.Evaluate(p);
        }

    // Compute sampling distribution for filter
    distrib = PiecewiseConstant2D(values, domain, alloc);
}

std::string FilterSampler::ToString() const {
    return StringPrintf("[ FilterSampler domain: %s values: %s distrib: %s ]", domain,
                        values, distrib);
}

}  // namespace pbrt
