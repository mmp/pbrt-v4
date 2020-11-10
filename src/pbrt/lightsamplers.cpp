// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/lightsamplers.h>

#include <pbrt/interaction.h>
#include <pbrt/lights.h>
#include <pbrt/util/bits.h>
#include <pbrt/util/check.h>
#include <pbrt/util/error.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/math.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/print.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/spectrum.h>

#include <atomic>
#include <cstdint>
#include <numeric>
#include <vector>

namespace pbrt {

std::string SampledLight::ToString() const {
    return StringPrintf("[ SampledLight light: %s pdf: %f ]",
                        light ? light.ToString().c_str() : "(nullptr)", pdf);
}

LightSamplerHandle LightSamplerHandle::Create(const std::string &name,
                                              pstd::span<const LightHandle> lights,
                                              Allocator alloc) {
    if (name == "uniform")
        return alloc.new_object<UniformLightSampler>(lights, alloc);
    else if (name == "power")
        return alloc.new_object<PowerLightSampler>(lights, alloc);
    else if (name == "bvh")
        return alloc.new_object<BVHLightSampler>(lights, alloc);
    else if (name == "exhaustive")
        return alloc.new_object<ExhaustiveLightSampler>(lights, alloc);
    else {
        Error(R"(Light sample distribution type "%s" unknown. Using "bvh".)",
              name.c_str());
        return alloc.new_object<BVHLightSampler>(lights, alloc);
    }
}

std::string LightSamplerHandle::ToString() const {
    if (!ptr())
        return "(nullptr)";

    auto ts = [&](auto ptr) { return ptr->ToString(); };
    return DispatchCPU(ts);
}

///////////////////////////////////////////////////////////////////////////
// PowerLightSampler

// PowerLightSampler Method Definitions
PowerLightSampler::PowerLightSampler(pstd::span<const LightHandle> lights,
                                     Allocator alloc)
    : lightToIndex(alloc),
      lights(lights.begin(), lights.end(), alloc),
      aliasTable(alloc) {
    if (lights.empty())
        return;
    // Initialize _lightToIndex_ hash table
    for (size_t i = 0; i < lights.size(); ++i)
        lightToIndex.Insert(lights[i], i);

    // Compute lights' power and initialize alias table
    std::vector<Float> lightPower;
    SampledWavelengths lambda = SampledWavelengths::SampleUniform(0.5);
    for (const auto &light : lights)
        lightPower.push_back(light.Phi(lambda).Average());
    if (std::accumulate(lightPower.begin(), lightPower.end(), 0.f) == 0.f)
        std::fill(lightPower.begin(), lightPower.end(), 1.f);
    aliasTable = AliasTable(lightPower, alloc);
}

std::string PowerLightSampler::ToString() const {
    return StringPrintf("[ PowerLightSampler aliasTable: %s ]", aliasTable);
}

///////////////////////////////////////////////////////////////////////////
// BVHLightSampler

STAT_MEMORY_COUNTER("Memory/Light BVH", lightBVHBytes);
STAT_INT_DISTRIBUTION("Integrator/Lights sampled per lookup", nLightsSampled);

// BVHLightSampler Method Definitions
BVHLightSampler::BVHLightSampler(pstd::span<const LightHandle> lights, Allocator alloc)
    : lights(lights.begin(), lights.end(), alloc),
      infiniteLights(alloc),
      lightToNode(alloc) {
    std::vector<std::pair<LightHandle, LightBounds>> bvhLights;
    // Partition lights into _infiniteLights_ and _bvhLights_
    for (const auto &light : lights) {
        LightBounds lightBounds = light.Bounds();
        if (!lightBounds)
            infiniteLights.push_back(light);
        else if (lightBounds.phi > 0)
            bvhLights.push_back(std::make_pair(light, lightBounds));
    }

    if (bvhLights.empty())
        return;
    int nNodes = 0;
    root = buildBVH(bvhLights, 0, bvhLights.size(), alloc, &nNodes);
    lightBVHBytes += nNodes * sizeof(LightBVHNode);
}

LightBVHNode *BVHLightSampler::buildBVH(
    std::vector<std::pair<LightHandle, LightBounds>> &lights, int start, int end,
    Allocator alloc, int *nNodes) {
    CHECK_LT(start, end);
    (*nNodes)++;
    int nLights = end - start;
    if (nLights == 1) {
        LightBVHNode *node =
            alloc.new_object<LightBVHNode>(lights[start].first, lights[start].second);
        lightToNode.Insert(lights[start].first, node);
        return node;
    }

    Bounds3f bounds, centroidBounds;
    for (int i = start; i < end; ++i) {
        const LightBounds &lb = lights[i].second;
        bounds = Union(bounds, lb.b);
        centroidBounds = Union(centroidBounds, lb.Centroid());
    }

    // Modified SAH
    // Replace # of primitives with emitter power
    // TODO: use the more efficient bounds/cost sweep calculation from v4

    Float minCost = Infinity;
    int minCostSplitBucket = -1, minCostSplitDim = -1;
    constexpr int nBuckets = 12;

    for (int dim = 0; dim < 3; ++dim) {
        if (centroidBounds.pMax[dim] == centroidBounds.pMin[dim]) {
            continue;
        }

        LightBounds bucketLightBounds[nBuckets];

        for (int i = start; i < end; ++i) {
            Point3f pc = lights[i].second.Centroid();
            int b = nBuckets * centroidBounds.Offset(pc)[dim];
            if (b == nBuckets)
                b = nBuckets - 1;
            CHECK_GE(b, 0);
            CHECK_LT(b, nBuckets);
            bucketLightBounds[b] = Union(bucketLightBounds[b], lights[i].second);
        }

        // Compute costs for splitting after each bucket
        Float cost[nBuckets - 1];
        for (int i = 0; i < nBuckets - 1; ++i) {
            LightBounds b0, b1;

            for (int j = 0; j <= i; ++j)
                b0 = Union(b0, bucketLightBounds[j]);
            for (int j = i + 1; j < nBuckets; ++j)
                b1 = Union(b1, bucketLightBounds[j]);

            auto Momega = [](const LightBounds &b) {
                Float theta_w = std::min(b.theta_o + b.theta_e, Pi);
                return 2 * Pi * (1 - std::cos(b.theta_o)) +
                       Pi / 2 *
                           (2 * theta_w * std::sin(b.theta_o) -
                            std::cos(b.theta_o - 2 * theta_w) -
                            2 * b.theta_o * std::sin(b.theta_o) + std::cos(b.theta_o));
            };

            // Can simplify since we always split
            Float Kr = MaxComponentValue(bounds.Diagonal()) / bounds.Diagonal()[dim];
            cost[i] = Kr * (b0.phi * Momega(b0) * b0.b.SurfaceArea() +
                            b1.phi * Momega(b1) * b1.b.SurfaceArea());
        }

        // Find bucket to split at that minimizes SAH metric
        for (int i = 1; i < nBuckets - 1; ++i) {
            if (cost[i] > 0 && cost[i] < minCost) {
                minCost = cost[i];
                minCostSplitBucket = i;
                minCostSplitDim = dim;
            }
        }
    }

    int mid;
    if (minCostSplitDim == -1) {
        mid = (start + end) / 2;
    } else {
        const auto *pmid = std::partition(
            &lights[start], &lights[end - 1] + 1,
            [=](const std::pair<LightHandle, LightBounds> &l) {
                int b = nBuckets *
                        centroidBounds.Offset(l.second.Centroid())[minCostSplitDim];
                if (b == nBuckets)
                    b = nBuckets - 1;
                CHECK_GE(b, 0);
                CHECK_LT(b, nBuckets);
                return b <= minCostSplitBucket;
            });
        mid = pmid - &lights[0];

        if (mid == start || mid == end) {
            mid = (start + end) / 2;
        }
        CHECK(mid > start && mid < end);
    }

    LightBVHNode *node =
        alloc.new_object<LightBVHNode>(buildBVH(lights, start, mid, alloc, nNodes),
                                       buildBVH(lights, mid, end, alloc, nNodes));
    node->children[0]->parent = node->children[1]->parent = node;

    return node;
}

std::string BVHLightSampler::ToString() const {
    return StringPrintf("[ BVHLightSampler root: %s ]",
                        root ? root->ToString() : std::string("(nullptr)"));
}

std::string LightBVHNode::ToString() const {
    std::string s =
        StringPrintf("[ LightBVHNode lightBounds: %s isLeaf: %s ", lightBounds, isLeaf);
    if (isLeaf)
        s += StringPrintf("light: %s ", light);
    else
        s += StringPrintf("children[0]: %s children[1]: %s", *children[0], *children[1]);
    return s + "]";
}

// ExhaustiveLightSampler Method Definitions
ExhaustiveLightSampler::ExhaustiveLightSampler(pstd::span<const LightHandle> lights,
                                               Allocator alloc)
    : lights(lights.begin(), lights.end(), alloc),
      boundedLights(alloc),
      infiniteLights(alloc),
      lightBounds(alloc),
      lightToBoundedIndex(alloc) {
    for (const auto &light : lights) {
        if (LightBounds lb = light.Bounds(); lb) {
            lightToBoundedIndex.Insert(light, boundedLights.size());
            lightBounds.push_back(lb);
            boundedLights.push_back(light);
        } else
            infiniteLights.push_back(light);
    }
}

pstd::optional<SampledLight> ExhaustiveLightSampler::Sample(const LightSampleContext &ctx,
                                                            Float u) const {
    Float pInfinite = Float(infiniteLights.size()) /
                      Float(infiniteLights.size() + (!lightBounds.empty() ? 1 : 0));

    // Note: shared with BVH light sampler...
    if (u < pInfinite) {
        u = std::min<Float>(u * pInfinite, OneMinusEpsilon);
        int index = std::min<int>(u * infiniteLights.size(), infiniteLights.size() - 1);
        Float pdf = pInfinite * 1.f / infiniteLights.size();
        return SampledLight{infiniteLights[index], pdf};
    } else {
        u = std::min<Float>((u - pInfinite) / (1 - pInfinite), OneMinusEpsilon);

        uint64_t seed = MixBits(FloatToBits(u));
        WeightedReservoirSampler<LightHandle> wrs(seed);

        for (size_t i = 0; i < boundedLights.size(); ++i)
            wrs.Add(boundedLights[i], lightBounds[i].Importance(ctx.p(), ctx.n));

        if (!wrs.HasSample())
            return {};

        Float pdf = (1.f - pInfinite) * wrs.SamplePDF();
        return SampledLight{wrs.GetSample(), pdf};
    }
}

Float ExhaustiveLightSampler::PDF(const LightSampleContext &ctx,
                                  LightHandle light) const {
    if (!lightToBoundedIndex.HasKey(light))
        return 1.f / (infiniteLights.size() + (!lightBounds.empty() ? 1 : 0));

    Float importanceSum = 0;
    Float lightImportance = 0;
    for (size_t i = 0; i < boundedLights.size(); ++i) {
        Float importance = lightBounds[i].Importance(ctx.p(), ctx.n);
        importanceSum += importance;
        if (light == boundedLights[i])
            lightImportance = importance;
    }

    Float pInfinite = Float(infiniteLights.size()) /
                      Float(infiniteLights.size() + (!lightBounds.empty() ? 1 : 0));
    Float pdf = lightImportance / importanceSum * (1. - pInfinite);
    return pdf;
}

std::string ExhaustiveLightSampler::ToString() const {
    return StringPrintf("[ ExhaustiveLightSampler lightBounds: %s]", lightBounds);
}

}  // namespace pbrt
