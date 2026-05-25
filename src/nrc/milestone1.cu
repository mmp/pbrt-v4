#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/gpu_matrix.h>

#include <cuda_runtime.h>

#include <iostream>
#include <random>
#include <vector>
#include <stdexcept>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                                     cudaGetErrorString(err));                  \
        }                                                                      \
    } while (0)

static void fill_training_batch(
    tcnn::GPUMatrix<float>& inputs,
    tcnn::GPUMatrix<float>& targets,
    uint32_t n_input_dims,
    uint32_t n_output_dims,
    uint32_t batch_size
) {
    std::vector<float> h_inputs(n_input_dims * batch_size);
    std::vector<float> h_targets(n_output_dims * batch_size);

    static std::mt19937 rng(1234);
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // GPUMatrix is column-major by default:
    // value(row, col) -> data[col * rows + row]
    // Given a 16-dimensional input, it should output the first three values as RGB.
    for (uint32_t col = 0; col < batch_size; ++col) {
        float x = dist(rng);
        float y = dist(rng);
        float z = dist(rng);

        for (uint32_t row = 0; row < n_input_dims; ++row) {
            h_inputs[col * n_input_dims + row] = 0.0f;
        }

        h_inputs[col * n_input_dims + 0] = x;
        h_inputs[col * n_input_dims + 1] = y;
        h_inputs[col * n_input_dims + 2] = z;

        // Dummy RGB target:
        // The network should learn approximately:
        // R = x, G = y, B = z
        h_targets[col * n_output_dims + 0] = x;
        h_targets[col * n_output_dims + 1] = y;
        h_targets[col * n_output_dims + 2] = z;
    }

    CUDA_CHECK(cudaMemcpy(
        inputs.data(),
        h_inputs.data(),
        h_inputs.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        targets.data(),
        h_targets.data(),
        h_targets.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));
}

int main() {
    try {
        constexpr uint32_t n_input_dims = 16;
        constexpr uint32_t n_output_dims = 3;
        constexpr uint32_t batch_size = 32768;
        constexpr uint32_t n_steps = 200;

        nlohmann::json config = {
            {"loss", {
                {"otype", "L2"}
            }},
            {"optimizer", {
                {"otype", "Adam"},
                {"learning_rate", 1e-3}
            }},
            {"encoding", {
                {"otype", "Identity"}
            }},
            {"network", {
                {"otype", "FullyFusedMLP"},
                {"activation", "ReLU"},
                {"output_activation", "None"},
                {"n_neurons", 64},
                {"n_hidden_layers", 2}
            }}
        };

        auto model = tcnn::create_from_config(
            n_input_dims,
            n_output_dims,
            config
        );

        std::cout << "Created tiny-cuda-nn model.\n";
        std::cout << "Parameters: " << model.trainer->n_params() << "\n";

        tcnn::GPUMatrix<float> inputs(n_input_dims, batch_size);
        tcnn::GPUMatrix<float> targets(n_output_dims, batch_size);

        for (uint32_t step = 0; step < n_steps; ++step) {
            fill_training_batch(
                inputs,
                targets,
                n_input_dims,
                n_output_dims,
                batch_size
            );

            auto ctx = model.trainer->training_step(inputs, targets);
            float loss = model.trainer->loss(*ctx);

            if (step % 20 == 0) {
                std::cout << "step " << step
                          << " loss = " << loss
                          << "\n";
            }
        }

        tcnn::GPUMatrix<float> outputs(n_output_dims, batch_size);
        model.network->inference(inputs, outputs);

        CUDA_CHECK(cudaDeviceSynchronize());

        auto h_inputs = inputs.to_cpu_vector();
        auto h_outputs = outputs.to_cpu_vector();

        std::cout << "\nExample inference:\n";

        for (uint32_t i = 0; i < 5; ++i) {
            float x = h_inputs[i * n_input_dims + 0];
            float y = h_inputs[i * n_input_dims + 1];
            float z = h_inputs[i * n_input_dims + 2];

            float r = h_outputs[i * n_output_dims + 0];
            float g = h_outputs[i * n_output_dims + 1];
            float b = h_outputs[i * n_output_dims + 2];

            std::cout << "target = [" << x << ", " << y << ", " << z << "]  "
                      << "prediction = [" << r << ", " << g << ", " << b << "]\n";
        }

        std::cout << "\nMilestone 1 passed: tiny-cuda-nn trains and runs inference inside the PBRT build.\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}