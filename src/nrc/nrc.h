// pbrt-v4 NRC milestone 2: thin C++ wrapper around tiny-cuda-nn.
//
// This header is included by host code in the wavefront integrator. It does
// *not* drag in any tiny-cuda-nn or CUDA headers; the implementation lives
// in nrc.cu and is hidden behind a pimpl. This lets the integrator stay
// plain-C++ in most translation units.

#ifndef PBRT_NRC_NRC_H
#define PBRT_NRC_NRC_H

#include <cstddef>
#include <cstdint>

namespace pbrt {
namespace nrc {

// NeuralRadianceCache
//
// Owns a tiny-cuda-nn model (FullyFusedMLP, identity encoding, Adam, L2),
// matching the configuration used in src/nrc/milestone1.cu so behaviour is
// reproducible across milestones.
//
// All pointers passed to Train()/Inference() are *device* pointers (raw
// CUDA memory, including cudaMallocManaged). Layouts are column-major to
// match tcnn's GPUMatrix default:
//   inputs:  [nInputDims  x batchSize], stride 1 between rows
//   targets: [nOutputDims x batchSize]
//   outputs: [nOutputDims x batchSize]
//
// batchSize must be a multiple of tcnn::batch_size_granularity (typically
// 128). Use RoundUpBatch() to obtain a valid size.
class NeuralRadianceCache {
  public:
    NeuralRadianceCache(uint32_t batchSize, uint32_t nInputDims,
                        uint32_t nOutputDims);
    ~NeuralRadianceCache();

    NeuralRadianceCache(const NeuralRadianceCache &) = delete;
    NeuralRadianceCache &operator=(const NeuralRadianceCache &) = delete;

    // One training step. Returns the (post-step) loss reported by tcnn.
    float Train(const float *dInputs, const float *dTargets);

    // Forward pass only. Writes nOutputDims*batchSize floats into dOutputs.
    void Inference(const float *dInputs, float *dOutputs);

    size_t NumParams() const;

    uint32_t BatchSize() const { return batchSize; }
    uint32_t NInputDims() const { return nInputDims; }
    uint32_t NOutputDims() const { return nOutputDims; }

    // Round n up to a valid tcnn batch size.
    static uint32_t RoundUpBatch(uint32_t n);

  private:
    struct Impl;
    Impl *impl;
    uint32_t batchSize;
    uint32_t nInputDims;
    uint32_t nOutputDims;
};

}  // namespace nrc
}  // namespace pbrt

#endif  // PBRT_NRC_NRC_H
