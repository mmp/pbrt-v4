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
class LightBVHNode {
  public:
    LightBVHNode(LightHandle light, const LightBounds &lightBounds)
        : light(light), lightBounds(lightBounds) {
        isLeaf = true;
    }
    LightBVHNode(LightBVHNode *c0, LightBVHNode *c1)
        : lightBounds(Union(c0->lightBounds, c1->lightBounds)) {
        isLeaf = false;
        children[0] = c0;
        children[1] = c1;
    }

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, Float u) const;

    std::string ToString() const;

    LightBounds lightBounds;
    bool isLeaf;
    union {
        LightHandle light;
        LightBVHNode *children[2];
    };
    LightBVHNode *parent = nullptr;
};

// BVHLightSampler Definition
class BVHLightSampler {
  public:
    // BVHLightSampler Public Methods
    BVHLightSampler(pstd::span<const LightHandle> lights, Allocator alloc);

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, Float u) const {
        Point3f p = ctx.p();
        Normal3f n = ctx.ns;
        // FIXME: handle no lights at all w/o a NaN...
        Float pInfinite = Float(infiniteLights.size()) /
                          Float(infiniteLights.size() + (root != nullptr ? 1 : 0));

        if (u < pInfinite) {
            u = std::min<Float>(u * pInfinite, OneMinusEpsilon);
            int index =
                std::min<int>(u * infiniteLights.size(), infiniteLights.size() - 1);
            Float pdf = pInfinite * 1.f / infiniteLights.size();
            return SampledLight{infiniteLights[index], pdf};
        } else {
            if (root == nullptr)
                return {};

            u = std::min<Float>((u - pInfinite) / (1 - pInfinite), OneMinusEpsilon);
            const LightBVHNode *node = root;
            Float pdf = (1 - pInfinite);
            while (true) {
                if (node->isLeaf) {
                    if (node->lightBounds.Importance(p, n) > 0)
                        return SampledLight{node->light, pdf};
                    return {};
                } else {
                    pstd::array<Float, 2> ci = {
                        node->children[0]->lightBounds.Importance(p, n),
                        node->children[1]->lightBounds.Importance(p, n)};
                    if (ci[0] == 0 && ci[1] == 0)
                        // It may happen that we follow a path down the tree and later
                        // find that there aren't any lights that illuminate our point;
                        // a natural consequence of the bounds tightening up on the way
                        // down.
                        return {};

                    Float nodePDF;
                    int child = SampleDiscrete(ci, u, &nodePDF, &u);
                    pdf *= nodePDF;
                    node = node->children[child];
                }
            }
        }
    }

    PBRT_CPU_GPU
    Float PDF(const LightSampleContext &ctx, LightHandle light) const {
        if (!lightToNode.HasKey(light))
            return 1.f / (infiniteLights.size() + (root != nullptr ? 1 : 0));

        LightBVHNode *node = lightToNode[light];
        Float pdf = 1;

        Point3f p = ctx.p();
        Normal3f n = ctx.ns;
        if (node->lightBounds.Importance(p, n) == 0)
            return 0;

        for (; node->parent != nullptr; node = node->parent) {
            pstd::array<Float, 2> ci = {
                node->parent->children[0]->lightBounds.Importance(p, n),
                node->parent->children[1]->lightBounds.Importance(p, n)};
            int childIndex = static_cast<int>(node == node->parent->children[1]);
            DCHECK_GT(ci[childIndex], 0);
            pdf *= ci[childIndex] / (ci[0] + ci[1]);
        }

        Float pInfinite = Float(infiniteLights.size()) / Float(infiniteLights.size() + 1);
        return pdf * (1.f - pInfinite);
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
    LightBVHNode *buildBVH(std::vector<std::pair<LightHandle, LightBounds>> &lights,
                           int start, int end, Allocator alloc, int *nNodes);

    // BVHLightSampler Private Members
    LightBVHNode *root = nullptr;
    pstd::vector<LightHandle> lights, infiniteLights;
    HashMap<LightHandle, LightBVHNode *, LightHandleHash> lightToNode;
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
