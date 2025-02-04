#include "MyDL/Conv2d.h"
#include <iostream>
#include <stdexcept>

namespace MyDL {

/**
 * Forward pass of the 2D convolution operation.
 * Implements a sliding window operation over the input tensor.
 * @param x Input tensor of shape `[N, C_in, H, W]`.
 * @return Output tensor of shape `[N, C_out, H_out, W_out]`.
 */
Tensor Conv2d::forward(const Tensor& x) {
    if (x.shape.size() != 4) {
        throw std::runtime_error("Conv2d::forward: Input tensor must have shape [N, C_in, H, W]");
    }

    int N = x.shape[0]; // Batch size
    int C_in = x.shape[1]; // Input channels
    int H = x.shape[2]; // Input height
    int W = x.shape[3]; // Input width

    if (C_in != in_channels) {
        throw std::runtime_error("Conv2d::forward: Input channels do not match layer's expected in_channels.");
    }

    // Compute output height and width based on stride and padding
    int H_out = (H - kernel_height + 2 * padding) / stride + 1;
    int W_out = (W - kernel_width + 2 * padding) / stride + 1;

    // Initialize output tensor
    Tensor output({N, out_channels, H_out, W_out});

    // Perform convolution for each output channel
    for (int n = 0; n < N; ++n) { // Batch
        for (int oc = 0; oc < out_channels; ++oc) { // Output channels
            for (int oh = 0; oh < H_out; ++oh) { // Output height
                for (int ow = 0; ow < W_out; ++ow) { // Output width
                    float sum = 0.0f;

                    // Sliding window convolution
                    for (int ic = 0; ic < in_channels; ++ic) { // Input channels
                        for (int kh = 0; kh < kernel_height; ++kh) { // Kernel height
                            for (int kw = 0; kw < kernel_width; ++kw) { // Kernel width
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;

                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    int input_idx = n * C_in * H * W + ic * H * W + ih * W + iw;
                                    int kernel_idx = oc * in_channels * kernel_height * kernel_width
                                                     + ic * kernel_height * kernel_width + kh * kernel_width + kw;

                                    sum += x.data[input_idx] * weight.data[kernel_idx];
                                }
                            }
                        }
                    }
                    // Add bias and store in output
                    int output_idx = n * out_channels * H_out * W_out + oc * H_out * W_out + oh * W_out + ow;
                    output.data[output_idx] = sum + bias.data[oc];
                }
            }
        }
    }
    return output;
}

} // namespace MyDL
