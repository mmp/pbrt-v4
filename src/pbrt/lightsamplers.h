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
