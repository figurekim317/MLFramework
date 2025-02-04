#ifndef MYDL_BATCHNORM2D_H
#define MYDL_BATCHNORM2D_H

#include "Module.h"
#include "Tensor.h"
#include <vector>
#include <cmath>
#include <iostream>

namespace MyDL {

/**
 * BatchNorm2d: Implements batch normalization for 2D feature maps.
 * 
 * Formula: 
 *   y = gamma * (x - mean) / sqrt(var + epsilon) + beta
 *
 * - `gamma` (scale) and `beta` (shift) are learnable parameters.
 * - Running mean and variance are updated during training.
 * - In inference mode, stored running mean/variance are used.
 */
class BatchNorm2d : public Module {
private:
    int num_features; // Number of channels in input tensor (C)
    float epsilon;    // Small constant for numerical stability

    Tensor gamma;     // Scale parameter (learnable)
    Tensor beta;      // Shift parameter (learnable)
    Tensor running_mean;  // Running mean (for inference)
    Tensor running_var;   // Running variance (for inference)

public:
    /**
     * Constructor for BatchNorm2d.
     * @param num_features Number of input channels (C).
     * @param eps Small value added to variance for numerical stability.
     */
    BatchNorm2d(int num_features, float eps = 1e-5)
        : num_features(num_features), epsilon(eps),
          gamma({num_features}, std::vector<float>(num_features, 1.0f)), // Initialize gamma to 1
          beta({num_features}, std::vector<float>(num_features, 0.0f)),  // Initialize beta to 0
          running_mean({num_features}, std::vector<float>(num_features, 0.0f)),
          running_var({num_features}, std::vector<float>(num_features, 1.0f)) {}

    /**
     * Forward pass of BatchNorm2d.
     * @param x Input tensor of shape [N, C, H, W].
     * @return Normalized output tensor of the same shape.
     */
    Tensor forward(const Tensor& x) override;

    /** Sets gamma (scale) parameter manually */
    void setGamma(const std::vector<float>& values) {
        if (values.size() != num_features) {
            throw std::invalid_argument("Gamma size mismatch.");
        }
        gamma.data = values;
    }

    /** Sets beta (shift) parameter manually */
    void setBeta(const std::vector<float>& values) {
        if (values.size() != num_features) {
            throw std::invalid_argument("Beta size mismatch.");
        }
        beta.data = values;
    }

    /** Sets running mean manually (for inference mode) */
    void setRunningMean(const std::vector<float>& values) {
        if (values.size() != num_features) {
            throw std::invalid_argument("Running mean size mismatch.");
        }
        running_mean.data = values;
    }

    /** Sets running variance manually (for inference mode) */
    void setRunningVar(const std::vector<float>& values) {
        if (values.size() != num_features) {
            throw std::invalid_argument("Running variance size mismatch.");
        }
        running_var.data = values;
    }

    /** Prints the internal parameters */
    void printParameters() const {
        std::cout << "Gamma: ";
        for (float v : gamma.data) std::cout << v << " ";
        std::cout << "\nBeta: ";
        for (float v : beta.data) std::cout << v << " ";
        std::cout << "\nRunning Mean: ";
        for (float v : running_mean.data) std::cout << v << " ";
        std::cout << "\nRunning Variance: ";
        for (float v : running_var.data) std::cout << v << " ";
        std::cout << std::endl;
    }
};

} // namespace MyDL

#endif // MYDL_BATCHNORM2D_H