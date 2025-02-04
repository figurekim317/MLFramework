#include "MyDL/Linear.h"
#include <iostream>
#include <stdexcept>

namespace MyDL {

/**
 * Constructor for initializing a Linear layer.
 * @param in_features Number of input features.
 * @param out_features Number of output features.
 */
Linear::Linear(int in_features, int out_features)
    : W({in_features, out_features}), b({1, out_features}) {
    // Initialize weights with small values (0.01 for demonstration)
    for (size_t i = 0; i < W.size(); i++) {
        W.data[i] = 0.01f;
    }

    // Bias is initialized to zero
}

/**
 * Sets pre-trained weights for the layer.
 * @param w_data Vector containing weight values in row-major order.
 */
void Linear::setWeights(const std::vector<float>& w_data) {
    if (w_data.size() != W.size()) {
        throw std::runtime_error("Linear::setWeights: size mismatch");
    }
    W.data = w_data;
}

/**
 * Sets pre-trained bias values for the layer.
 * @param b_data Vector containing bias values.
 */
void Linear::setBias(const std::vector<float>& b_data) {
    if (b_data.size() != (size_t)W.shape[1]) {
        throw std::runtime_error("Linear::setBias: size mismatch");
    }
    for (int i = 0; i < W.shape[1]; i++) {
        b.data[i] = b_data[i];
    }
}

/**
 * Performs the forward pass of the linear transformation.
 * @param x Input tensor of shape `[N, in_features]`.
 * @return Output tensor of shape `[N, out_features]`.
 */
Tensor Linear::forward(const Tensor& x) {
    if (x.shape[1] != W.shape[0]) {
        throw std::runtime_error("Linear::forward: Input tensor shape does not match weight dimensions.");
    }

    // Perform matrix multiplication: out = x * W
    Tensor out = F::matmul(x, W);

    // Perform bias addition (broadcasting over batch dimension)
    int N = out.shape[0];       // Number of samples in batch
    int out_feats = out.shape[1]; // Number of output features

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < out_feats; j++) {
            out.data[i * out_feats + j] += b.data[j];
        }
    }
    return out;
}

/**
 * Prints the layer parameters (weights and biases).
 */
void Linear::printParameters() const {
    std::cout << "Weights:\n";
    for (size_t i = 0; i < W.size(); i++) {
        std::cout << W.data[i] << " ";
        if ((i + 1) % W.shape[1] == 0) std::cout << std::endl;
    }
    std::cout << "Bias:\n";
    for (float v : b.data) std::cout << v << " ";
    std::cout << std::endl;
}

} // namespace MyDL
