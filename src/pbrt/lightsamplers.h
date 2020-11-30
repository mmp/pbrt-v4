// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_LIGHTSAMPLERS_H
#define PBRT_LIGHTSAMPLERS_H

#include <pbrt/pbrt.h>

#include <pbrt/base/light.h>
#include <pbrt/base/lightsampler.h>
#include <pbrt/lights.h>  // LightBounds. Should that live elsewhere?
#include <pbrt/util/containers.h>
#include <pbrt/util/hash.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/vecmath.h>

#include <cstdint>
#include <string>

namespace pbrt {

// LightHandleHash Definition
struct LightHandleHash {
    PBRT_CPU_GPU
    size_t operator()(LightHandle lightHandle) const { return Hash(lightHandle.ptr()); }
};

// UniformLightSampler Definition
class UniformLightSampler {
  public:
    // UniformLightSampler Public Methods
    UniformLightSampler(pstd::span<const LightHandle> lights, Allocator alloc)
        : lights(lights.begin(), lights.end(), alloc) {}

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(Float u) const {
        if (lights.empty())
            return {};
        int lightIndex = std::min<int>(u * lights.size(), lights.size() - 1);
        return SampledLight{lights[lightIndex], 1.f / lights.size()};
    }

    PBRT_CPU_GPU
    Float PDF(LightHandle light) const {
        if (lights.empty())
            return 0;
        return 1.f / lights.size();
    }

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, Float u) const {
        return Sample(u);
    }

    PBRT_CPU_GPU
    Float PDF(const LightSampleContext &ctx, LightHandle light) const {
        return PDF(light);
    }

    std::string ToString() const { return "UniformLightSampler"; }

  private:
    // UniformLightSampler Private Members
    pstd::vector<LightHandle> lights;
};

// PowerLightSampler Definition
class PowerLightSampler {
  public:
    // PowerLightSampler Public Methods
    PowerLightSampler(pstd::span<const LightHandle> lights, Allocator alloc);

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(Float u) const {
        if (!aliasTable.size())
            return {};
        Float pdf;
        int lightIndex = aliasTable.Sample(u, &pdf);
        return SampledLight{lights[lightIndex], pdf};
    }

    PBRT_CPU_GPU
    Float PDF(LightHandle light) const {
        if (!aliasTable.size())
            return 0;
        return aliasTable.PDF(lightToIndex[light]);
    }

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, Float u) const {
        return Sample(u);
    }

    PBRT_CPU_GPU
    Float PDF(const LightSampleContext &ctx, LightHandle light) const {
        return PDF(light);
    }

    std::string ToString() const;

  private:
    // PowerLightSampler Private Members
    pstd::vector<LightHandle> lights;
    HashMap<LightHandle, size_t, LightHandleHash> lightToIndex;
    AliasTable aliasTable;
};

// CompactLightBounds Definition
struct CompactLightBounds {
  public:
    // CompactLightBounds Public Methods
    CompactLightBounds() = default;

    PBRT_CPU_GPU
    CompactLightBounds(const LightBounds &lb, const Bounds3f &allb)
        : w(Normalize(lb.w)),
          phi(lb.phi),
          qCosTheta_o(QuantizeCos(lb.cosTheta_o)),
          qCosTheta_e(QuantizeCos(lb.cosTheta_e)),
          twoSided(lb.twoSided) {
        // Quantize bounding box into _qb_
        for (int c = 0; c < 3; ++c) {
            qb[0][c] =
                std::floor(RescaleBounds(lb.bounds[0][c], allb.pMin[c], allb.pMax[c]));
            qb[1][c] =
                std::ceil(RescaleBounds(lb.bounds[1][c], allb.pMin[c], allb.pMax[c]));
        }
    }

    PBRT_CPU_GPU
    CompactLightBounds(const uint16_t b[2][3], const Vector3f &w, Float phi,
                       Float cosTheta_o, Float cosTheta_e, bool twoSided)
        : w(Normalize(w)),
          phi(phi),
          qCosTheta_o(QuantizeCos(cosTheta_o)),
          qCosTheta_e(QuantizeCos(cosTheta_e)),
          twoSided(twoSided) {
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 3; ++j)
                qb[i][j] = b[i][j];
    }

    std::string ToString() const;
    std::string ToString(const Bounds3f &allBounds) const;

    PBRT_CPU_GPU
    bool TwoSided() const { return twoSided; }
    PBRT_CPU_GPU
    Float CosTheta_o() const { return 2 * (qCosTheta_o / 32767.f) - 1; }
    PBRT_CPU_GPU
    Float CosTheta_e() const { return 2 * (qCosTheta_e / 32767.f) - 1; }

    PBRT_CPU_GPU
    Point3f Centroid(const Bounds3f &b) const {
        Bounds3f bbox = Bounds(b);
        return (bbox.pMin + bbox.pMax) / 2;
    }

    PBRT_CPU_GPU
    Bounds3f Bounds(const Bounds3f &allb) const {
        return {Point3f(Lerp(qb[0][0] / 65535.f, allb.pMin.x, allb.pMax.x),
                        Lerp(qb[0][1] / 65535.f, allb.pMin.y, allb.pMax.y),
                        Lerp(qb[0][2] / 65535.f, allb.pMin.z, allb.pMax.z)),
                Point3f(Lerp(qb[1][0] / 65535.f, allb.pMin.x, allb.pMax.x),
                        Lerp(qb[1][1] / 65535.f, allb.pMin.y, allb.pMax.y),
                        Lerp(qb[1][2] / 65535.f, allb.pMin.z, allb.pMax.z))};
    }

    PBRT_CPU_GPU
    Float Importance(Point3f p, Normal3f n, const Bounds3f &allLightBounds) const {
        Bounds3f bounds = Bounds(allLightBounds);
        Float cosTheta_o = CosTheta_o(), cosTheta_e = CosTheta_e();
        // Return importance for light bounds at reference point
        // Compute clamped squared distance to reference point
        Point3f pc = (bounds.pMin + bounds.pMax) / 2;
        Float d2 = DistanceSquared(p, pc);
        d2 = std::max(d2, Length(bounds.Diagonal()) / 2);

        // Compute sine and cosine of angle to vector _w_
        Vector3f wi = Normalize(p - pc);
        Float cosTheta = Dot(Vector3f(w), wi);
        if (twoSided)
            cosTheta = std::abs(cosTheta);
        Float sinTheta = SafeSqrt(1 - Sqr(cosTheta));

        // Define cosine and sine clamped subtraction lambdas
        auto cosSubClamped = [](Float sinTheta_a, Float cosTheta_a, Float sinTheta_b,
                                Float cosTheta_b) -> Float {
            if (cosTheta_a > cosTheta_b)
                return 1;
            return cosTheta_a * cosTheta_b + sinTheta_a * sinTheta_b;
        };

        auto sinSubClamped = [](Float sinTheta_a, Float cosTheta_a, Float sinTheta_b,
                                Float cosTheta_b) -> Float {
            if (cosTheta_a > cosTheta_b)
                return 0;
            return sinTheta_a * cosTheta_b - cosTheta_a * sinTheta_b;
        };

        // Compute $\cos \theta_\roman{u}$ for reference point
        Float cosTheta_u = BoundSubtendedDirections(bounds, p).cosTheta;
        Float sinTheta_u = SafeSqrt(1 - Sqr(cosTheta_u));

        // Compute $\cos \theta_\roman{p}$ and test against $\cos \theta_\roman{e}$
        Float sinTheta_o = SafeSqrt(1 - Sqr(cosTheta_o));
        Float cosTheta_x = cosSubClamped(sinTheta, cosTheta, sinTheta_o, cosTheta_o);
        Float sinTheta_x = sinSubClamped(sinTheta, cosTheta, sinTheta_o, cosTheta_o);
        Float cosTheta_p = cosSubClamped(sinTheta_x, cosTheta_x, sinTheta_u, cosTheta_u);
        if (cosTheta_p <= cosTheta_e)
            return 0;

        // Return final importance at reference point
        Float importance = phi * cosTheta_p / d2;
        DCHECK_GE(importance, -1e-3);
        // Account for $\cos \theta_\roman{i}$ in importance at surfaces
        if (n != Normal3f(0, 0, 0)) {
            Float cosTheta_i = AbsDot(wi, n);
            Float sinTheta_i = SafeSqrt(1 - Sqr(cosTheta_i));
            Float cosThetap_i =
                cosSubClamped(sinTheta_i, cosTheta_i, sinTheta_u, cosTheta_u);
            importance *= cosThetap_i;
        }

        importance = std::max<Float>(importance, 0);
        return importance;
    }

  private:
    // CompactLightBounds Private Methods
    PBRT_CPU_GPU
    static unsigned int QuantizeCos(Float c) {
        CHECK(c >= -1 && c <= 1);
        Float qc = 32767.f * ((c + 1) / 2);
        int q = std::floor(qc);
        CHECK(q >= 0 && q <= 32767);
        return q;
    }

    PBRT_CPU_GPU
    static Float RescaleBounds(Float p, Float min, Float max) {
        CHECK(p >= min && p <= max);
        if (min == max)
            return 0;
        Float pp = Clamp((p - min) / (max - min), 0, 1);
        Float qp = 65535.f * pp;
        CHECK(qp >= 0 && qp <= 65535);
        return qp;
    }

    // CompactLightBounds Private Members
    OctahedralVector w;
    Float phi = 0;
    struct {
        unsigned int qCosTheta_o : 15;
        unsigned int qCosTheta_e : 15;
        unsigned int twoSided : 1;
    };
    uint16_t qb[2][3];
};

// LightBVHNode Definition
struct alignas(32) LightBVHNode {
    // LightBVHNode Public Methods
    LightBVHNode() = default;

    PBRT_CPU_GPU
    LightBVHNode(int lightIndex, const CompactLightBounds &lightBounds)
        : childOrLightIndex(lightIndex), lightBounds(lightBounds) {
        isLeaf = true;
        parentIndex = (1u << 30) - 1;
    }

    PBRT_CPU_GPU
    LightBVHNode(int child0Index, int child1Index, const CompactLightBounds &lightBounds)
        : lightBounds(lightBounds), childOrLightIndex(child1Index) {
        isLeaf = false;
        parentIndex = (1u << 30) - 1;
    }

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, Float u) const;

    std::string ToString() const;

    // LightBVHNode Public Members
    CompactLightBounds lightBounds;
    int childOrLightIndex;
    struct {
        unsigned int parentIndex : 31;
        unsigned int isLeaf : 1;
    };
};

// BVHLightSampler Definition
class BVHLightSampler {
  public:
    // BVHLightSampler Public Methods
    BVHLightSampler(pstd::span<const LightHandle> lights, Allocator alloc);

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, Float u) const {
        // Compute infinite light sampling probability _pInfinite_
        Float pInfinite = Float(infiniteLights.size()) /
                          Float(infiniteLights.size() + (!nodes.empty() ? 1 : 0));

        if (u < pInfinite) {
            // Sample infinite lights with uniform probability
            u = std::min<Float>(u * pInfinite, OneMinusEpsilon);
            int index =
                std::min<int>(u * infiniteLights.size(), infiniteLights.size() - 1);
            Float pdf = pInfinite * 1.f / infiniteLights.size();
            return SampledLight{infiniteLights[index], pdf};

        } else {
            // Traverse light BVH to sample light
            if (nodes.empty())
                return {};
            Point3f p = ctx.p();
            Normal3f n = ctx.ns;
            u = std::min<Float>((u - pInfinite) / (1 - pInfinite), OneMinusEpsilon);
            int nodeIndex = 0;
            Float pdf = (1 - pInfinite);
            while (true) {
                LightBVHNode node = nodes[nodeIndex];
                // Process light BVH node _node_ for light sampling
                if (node.isLeaf) {
                    if (nodeIndex > 0)
                        DCHECK_GT(node.lightBounds.Importance(p, n, allLightBounds), 0);
                    if (nodeIndex > 0 ||
                        node.lightBounds.Importance(p, n, allLightBounds) > 0)
                        return SampledLight{lights[node.childOrLightIndex], pdf};
                    return {};
                } else {
                    // Compute child node importances and randomly sample child node
                    const LightBVHNode *children[2] = {&nodes[nodeIndex + 1],
                                                       &nodes[node.childOrLightIndex]};
                    pstd::array<Float, 2> ci = {
                        children[0]->lightBounds.Importance(p, n, allLightBounds),
                        children[1]->lightBounds.Importance(p, n, allLightBounds)};
                    if (ci[0] == 0 && ci[1] == 0)
                        return {};
                    Float nodePDF;
                    int child = SampleDiscrete(ci, u, &nodePDF, &u);
                    pdf *= nodePDF;
                    nodeIndex = (child == 0) ? (nodeIndex + 1) : node.childOrLightIndex;
                }
            }
        }
    }

    PBRT_CPU_GPU
    Float PDF(const LightSampleContext &ctx, LightHandle light) const {
        // Handle infinite _light_ PDF computation
        if (!lightToNodeIndex.HasKey(light))
            return 1.f / (infiniteLights.size() + (!nodes.empty() ? 1 : 0));

        // Get leaf _LightBVHNode_ for light and test importance
        int nodeIndex = lightToNodeIndex[light];
        Point3f p = ctx.p();
        Normal3f n = ctx.ns;
        if (nodes[nodeIndex].lightBounds.Importance(p, n, allLightBounds) == 0)
            return 0;

        // Compute light's PDF by walking up tree nodes to the root
        Float pdf = 1;
        while (nodeIndex != 0) {
            const LightBVHNode *node = &nodes[nodeIndex];
            const LightBVHNode *parent = &nodes[node->parentIndex];
            Float ci[2] = {
                nodes[node->parentIndex + 1].lightBounds.Importance(p, n, allLightBounds),
                nodes[parent->childOrLightIndex].lightBounds.Importance(p, n,
                                                                        allLightBounds)};
            int childIndex = int(nodeIndex == parent->childOrLightIndex);
            DCHECK_GT(ci[childIndex], 0);
            pdf *= ci[childIndex] / (ci[0] + ci[1]);
            nodeIndex = node->parentIndex;
        }

        // Return final PDF accounting for infinite light sampling probability
        Float pInfinite = Float(infiniteLights.size()) / Float(infiniteLights.size() + 1);
        return pdf * (1 - pInfinite);
    }

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(Float u) const {
        if (lights.empty())
            return {};
        int lightIndex = std::min<int>(u * lights.size(), lights.size() - 1);
        return SampledLight{lights[lightIndex], 1.f / lights.size()};
    }

    PBRT_CPU_GPU
    Float PDF(LightHandle light) const {
        if (lights.empty())
            return 0;
        return 1.f / lights.size();
    }

    std::string ToString() const;

  private:
    // BVHLightSampler Private Methods
    std::pair<int, LightBounds> buildBVH(
        std::vector<std::pair<int, LightBounds>> &bvhLights, int start, int end,
        Allocator alloc);

    // BVHLightSampler Private Members
    pstd::vector<LightHandle> lights, infiniteLights;
    pstd::vector<LightBVHNode> nodes;
    Bounds3f allLightBounds;
    HashMap<LightHandle, int, LightHandleHash> lightToNodeIndex;
};

// ExhaustiveLightSampler Definition
class ExhaustiveLightSampler {
  public:
    ExhaustiveLightSampler(pstd::span<const LightHandle> lights, Allocator alloc);

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, Float u) const;

    PBRT_CPU_GPU
    Float PDF(const LightSampleContext &ctx, LightHandle light) const;

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(Float u) const {
        if (lights.empty())
            return {};

        int lightIndex = std::min<int>(u * lights.size(), lights.size() - 1);
        return SampledLight{lights[lightIndex], 1.f / lights.size()};
    }

    PBRT_CPU_GPU
    Float PDF(LightHandle light) const {
        if (lights.empty())
            return 0;
        return 1.f / lights.size();
    }

    std::string ToString() const;

  private:
    pstd::vector<LightHandle> lights, boundedLights, infiniteLights;
    pstd::vector<LightBounds> lightBounds;
    HashMap<LightHandle, size_t, LightHandleHash> lightToBoundedIndex;
};

inline pstd::optional<SampledLight> LightSamplerHandle::Sample(
    const LightSampleContext &ctx, Float u) const {
    auto s = [&](auto ptr) { return ptr->Sample(ctx, u); };
    return Dispatch(s);
}

inline Float LightSamplerHandle::PDF(const LightSampleContext &ctx,
                                     LightHandle light) const {
    auto pdf = [&](auto ptr) { return ptr->PDF(ctx, light); };
    return Dispatch(pdf);
}

inline pstd::optional<SampledLight> LightSamplerHandle::Sample(Float u) const {
    auto sample = [&](auto ptr) { return ptr->Sample(u); };
    return Dispatch(sample);
}

inline Float LightSamplerHandle::PDF(LightHandle light) const {
    auto pdf = [&](auto ptr) { return ptr->PDF(light); };
    return Dispatch(pdf);
}

}  // namespace pbrt

#endif  // PBRT_LIGHTSAMPLERS_H
