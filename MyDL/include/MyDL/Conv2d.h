#ifndef MYDL_CONV2D_H
#define MYDL_CONV2D_H

#include <vector>
#include <stdexcept>
#include "Module.h"
#include "Functional.h"

namespace MyDL {

/**
 * Convolutional Layer (2D Convolution)
 *
 * The `Conv2d` layer applies a **2D convolution operation** to an input tensor, typically used
 * in image processing and deep learning architectures like CNNs, ResNet, and YOLO.
 *
 * Formula:
 *   `output(N, C_out, H_out, W_out) = Conv2D(input(N, C_in, H, W), weight(C_out, C_in, K_H, K_W)) + bias`
 *
 * - **input**: Input tensor of shape `[N, C_in, H, W]`
 * - **weight**: Kernel tensor of shape `[C_out, C_in, K_H, K_W]`
 * - **bias**: Bias tensor of shape `[C_out]` (optional)
 * - **stride**: Step size for sliding the kernel (default = 1)
 * - **padding**: Number of zero-padding pixels around the input (default = 0)
 * - **dilation**: Spacing between kernel elements (default = 1)
 *
 * **Features:**
 * - Supports configurable **stride, padding, and dilation**.
 * - Can be used in **convolutional neural networks (CNNs)**.
 * - Efficiently processes batch input with multiple channels.
 */
class Conv2d : public Module {
private:
    int in_channels, out_channels;
    int kernel_height, kernel_width;
    int stride, padding, dilation;

    Tensor weight;  // [out_channels, in_channels, kernel_height, kernel_width]
    Tensor bias;    // [out_channels]

public:
    /**
     * Constructor to initialize a Conv2D layer.
     * @param in_channels Number of input channels (C_in).
     * @param out_channels Number of output channels (C_out).
     * @param kernel_height Height of the kernel (K_H).
     * @param kernel_width Width of the kernel (K_W).
     * @param stride Step size for moving the kernel across the input.
     * @param padding Zero-padding around the input.
     * @param dilation Spacing between kernel elements.
     */
    Conv2d(int in_channels, int out_channels, int kernel_height, int kernel_width,
           int stride = 1, int padding = 0, int dilation = 1)
        : in_channels(in_channels), out_channels(out_channels),
          kernel_height(kernel_height), kernel_width(kernel_width),
          stride(stride), padding(padding), dilation(dilation),
          weight({out_channels, in_channels, kernel_height, kernel_width}),
          bias({out_channels}) {
        
        // Initialize weight and bias with small random values (for demonstration)
        for (size_t i = 0; i < weight.size(); i++) {
            weight.data[i] = 0.01f;  // Small random initialization
        }
        for (size_t i = 0; i < bias.size(); i++) {
            bias.data[i] = 0.0f;  // Bias initialized to zero
        }
    }

    /**
     * Sets pre-trained weights for the convolution kernel.
     * @param w_data Weight values in row-major order.
     */
    void setWeights(const std::vector<float>& w_data) {
        if (w_data.size() != weight.size()) {
            throw std::runtime_error("Conv2d::setWeights: size mismatch");
        }
        weight.data = w_data;
    }

    /**
     * Sets pre-trained bias values.
     * @param b_data Bias values.
     */
    void setBias(const std::vector<float>& b_data) {
        if (b_data.size() != (size_t)out_channels) {
            throw std::runtime_error("Conv2d::setBias: size mismatch");
        }
        bias.data = b_data;
    }

    /**
     * Performs the forward pass of the 2D convolution operation.
     * @param x Input tensor of shape `[N, C_in, H, W]`.
     * @return Output tensor of shape `[N, C_out, H_out, W_out]`.
     */
    Tensor forward(const Tensor& x) override;

    /**
     * Prints the layer parameters (weights and biases).
     */
    void printParameters() const {
        std::cout << "Conv2D Layer Parameters:\n";
        std::cout << "Weight shape: [" << weight.shape[0] << ", " << weight.shape[1] << ", "
                  << weight.shape[2] << ", " << weight.shape[3] << "]\n";
        std::cout << "Bias shape: [" << bias.shape[0] << "]\n";
    }
};

} // namespace MyDL

#endif // MYDL_CONV2D_H
