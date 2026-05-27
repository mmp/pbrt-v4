// pbrt-v4 NRC milestone 2: tiny-cuda-nn-backed neural radiance cache.
//
// Mirrors the config from src/nrc/milestone1.cu (identity encoding,
// FullyFusedMLP 2x64 ReLU, Adam, L2 loss). The integrator owns one of
// these and calls Train()/Inference() once per scanline pass.

#include <nrc/nrc.h>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/gpu_matrix.h>

#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace pbrt {
namespace nrc {

struct NeuralRadianceCache::Impl {
    tcnn::TrainableModel model;
    cudaStream_t stream = nullptr;
    float lastLoss = 0.f;

    Impl(uint32_t nIn, uint32_t nOut)
        : model(tcnn::create_from_config(
              nIn, nOut,
              {{"loss", {{"otype", "L2"}}},
               {"optimizer", {{"otype", "Adam"}, {"learning_rate", 1e-3}}},
               {"encoding", {{"otype", "Identity"}}},
               {"network",
                {{"otype", "FullyFusedMLP"},
                 {"activation", "ReLU"},
                 {"output_activation", "None"},
                 {"n_neurons", 64},
                 {"n_hidden_layers", 2}}}})) {}
};

NeuralRadianceCache::NeuralRadianceCache(uint32_t batchSize_, uint32_t nIn,
                                         uint32_t nOut)
    : batchSize(batchSize_), nInputDims(nIn), nOutputDims(nOut) {
    if (batchSize % tcnn::batch_size_granularity != 0) {
        throw std::runtime_error(
            "NeuralRadianceCache: batchSize must be a multiple of " +
            std::to_string(tcnn::batch_size_granularity));
    }
    impl = new Impl(nIn, nOut);
}

NeuralRadianceCache::~NeuralRadianceCache() {
    delete impl;
}

uint32_t NeuralRadianceCache::RoundUpBatch(uint32_t n) {
    const uint32_t g = tcnn::batch_size_granularity;
    return ((n + g - 1) / g) * g;
}

size_t NeuralRadianceCache::NumParams() const {
    return impl->model.trainer->n_params();
}

float NeuralRadianceCache::Train(const float *dInputs, const float *dTargets) {
    // tcnn::GPUMatrix non-owning view over caller-provided device memory.
    tcnn::GPUMatrix<float> inputs(const_cast<float *>(dInputs), nInputDims,
                                  batchSize);
    tcnn::GPUMatrix<float> targets(const_cast<float *>(dTargets), nOutputDims,
                                   batchSize);

    auto ctx = impl->model.trainer->training_step(inputs, targets);
    impl->lastLoss = impl->model.trainer->loss(*ctx);
    return impl->lastLoss;
}

void NeuralRadianceCache::Inference(const float *dInputs, float *dOutputs) {
    tcnn::GPUMatrix<float> inputs(const_cast<float *>(dInputs), nInputDims,
                                  batchSize);
    tcnn::GPUMatrix<float> outputs(dOutputs, nOutputDims, batchSize);
    impl->model.network->inference(inputs, outputs);
}

}  // namespace nrc
}  // namespace pbrt
