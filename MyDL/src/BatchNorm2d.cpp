#include "MyDL/BatchNorm2d.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <stdexcept>

namespace MyDL {

/**
 * Forward pass of BatchNorm2d.
 * Applies batch normalization to each channel independently.
 * @param x Input tensor of shape [N, C, H, W].
 * @return Normalized output tensor of the same shape.
 */
Tensor BatchNorm2d::forward(const Tensor& x) {
    if (x.shape.size() != 4) {
        throw std::invalid_argument("Input tensor must have shape [N, C, H, W]");
    }

    int N = x.shape[0]; // Batch size
    int C = x.shape[1]; // Number of channels
    int H = x.shape[2]; // Height
    int W = x.shape[3]; // Width

    if (C != num_features) {
        throw std::invalid_argument("Mismatch between input channels and BatchNorm channels.");
    }

    // Compute per-channel mean and variance
    std::vector<float> mean(C, 0.0f);
    std::vector<float> variance(C, 0.0f);

    int num_pixels = N * H * W; // Number of elements per channel

    // Compute mean for each channel
    for (int c = 0; c < C; ++c) {
        float sum = 0.0f;
        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    sum += x.data[n * C * H * W + c * H * W + h * W + w];
                }
            }
        }
        mean[c] = sum / num_pixels;
    }

    // Compute variance for each channel
    for (int c = 0; c < C; ++c) {
        float sum_sq = 0.0f;
        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    float diff = x.data[n * C * H * W + c * H * W + h * W + w] - mean[c];
                    sum_sq += diff * diff;
                }
            }
        }
        variance[c] = sum_sq / num_pixels;
    }

    // Normalize input tensor and apply gamma and beta
    Tensor output(x.shape);
    for (int c = 0; c < C; ++c) {
        float inv_std = 1.0f / std::sqrt(variance[c] + epsilon);
        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int index = n * C * H * W + c * H * W + h * W + w;
                    float normalized = (x.data[index] - mean[c]) * inv_std;
                    output.data[index] = gamma.data[c] * normalized + beta.data[c];
                }
            }
        }
    }

    // Update running mean and variance (assuming momentum of 0.9)
    float momentum = 0.9f;
    for (int c = 0; c < C; ++c) {
        running_mean.data[c] = momentum * running_mean.data[c] + (1.0f - momentum) * mean[c];
        running_var.data[c] = momentum * running_var.data[c] + (1.0f - momentum) * variance[c];
    }

    return output;
}

} // namespace MyDL