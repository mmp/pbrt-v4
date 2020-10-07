// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_BASE_BSSRDF_H
#define PBRT_BASE_BSSRDF_H

#include <pbrt/pbrt.h>

#include <pbrt/util/pstd.h>
#include <pbrt/util/taggedptr.h>

#include <string>

namespace pbrt {

struct BSSRDFSample;
struct BSSRDFProbeSegment;
struct SubsurfaceInteraction;
struct BSSRDFTable;

// BSSRDFHandle Definition
class TabulatedBSSRDF;

class BSSRDFHandle : public TaggedPointer<TabulatedBSSRDF> {
  public:
    // BSSRDF Interface
    using TaggedPointer::TaggedPointer;

    PBRT_CPU_GPU inline SampledSpectrum S(const Point3f &p, const Vector3f &wi);

    PBRT_CPU_GPU inline BSSRDFProbeSegment Sample(Float u1, const Point2f &u2) const;

    PBRT_CPU_GPU inline BSSRDFSample ProbeIntersectionToSample(
        const SubsurfaceInteraction &si, ScratchBuffer &scratchBuffer) const;
};

}  // namespace pbrt

#endif  // PBRT_BASE_BSSRDF_H
