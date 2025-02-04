#ifndef MYDL_RESNET_H
#define MYDL_RESNET_H

#include "Module.h"
#include "Conv2d.h"
#include "BatchNorm2d.h"
#include "Functional.h"
#include "Linear.h"
#include <vector>
#include <memory>

namespace MyDL {

/**
 * ResNet (Residual Network) Implementation
 *
 * ResNet is a deep convolutional neural network (CNN) architecture that introduces
 * **residual connections (shortcuts)** to enable very deep networks without vanishing gradients.
 *
 * This implementation supports:
 * - **Basic residual block**: Used in ResNet-18 and ResNet-34.
 * - **Bottleneck residual block**: Used in ResNet-50, ResNet-101, ResNet-152.
 *
 * General structure:
 * - **Initial Conv Layer**: `Conv2d -> BatchNorm2d -> ReLU`
 * - **Multiple Residual Blocks**
 * - **Global Average Pooling**
 * - **Fully Connected Layer (Linear)**
 *
 * Supported Architectures:
 * - `ResNet-18` (Basic Blocks)
 * - `ResNet-34` (Basic Blocks)
 * - `ResNet-50` (Bottleneck Blocks)
 * - `ResNet-101` (Bottleneck Blocks)
 * - `ResNet-152` (Bottleneck Blocks)
 */
class ResidualBlock : public Module {
private:
    std::shared_ptr<Conv2d> conv1, conv2;
    std::shared_ptr<BatchNorm2d> bn1, bn2;
    std::shared_ptr<Module> shortcut;
    bool downsample;

public:
    /**
     * Constructor for a residual block.
     * @param in_channels Number of input channels.
     * @param out_channels Number of output channels.
     * @param stride Stride value for downsampling.
     */
    ResidualBlock(int in_channels, int out_channels, int stride = 1) {
        conv1 = std::make_shared<Conv2d>(in_channels, out_channels, 3, 3, stride, 1);
        bn1 = std::make_shared<BatchNorm2d>(out_channels);
        conv2 = std::make_shared<Conv2d>(out_channels, out_channels, 3, 3, 1, 1);
        bn2 = std::make_shared<BatchNorm2d>(out_channels);

        downsample = (stride != 1 || in_channels != out_channels);
        if (downsample) {
            shortcut = std::make_shared<Conv2d>(in_channels, out_channels, 1, 1, stride, 0);
        }
    }

    /**
     * Forward pass of the residual block.
     * @param x Input tensor.
     * @return Output tensor after residual connection.
     */
    Tensor forward(const Tensor& x) override {
        Tensor identity = x;
        Tensor out = conv1->forward(x);
        out = bn1->forward(out);
        out = F::relu(out);

        out = conv2->forward(out);
        out = bn2->forward(out);

        if (downsample) {
            identity = shortcut->forward(x);
        }

        out = F::add(out, identity);
        return F::relu(out);
    }
};

/**
 * ResNet Model Class
 *
 * Implements the full ResNet architecture by stacking multiple residual blocks.
 */
class ResNet : public Module {
private:
    std::shared_ptr<Conv2d> conv1;
    std::shared_ptr<BatchNorm2d> bn1;
    std::vector<std::shared_ptr<ResidualBlock>> layers;
    std::shared_ptr<Linear> fc;
    int num_classes;

public:
    /**
     * Constructor for a ResNet model.
     * @param num_blocks Vector specifying the number of residual blocks in each stage.
     * @param num_classes Number of output classes.
     */
    ResNet(std::vector<int> num_blocks, int num_classes = 1000) : num_classes(num_classes) {
        conv1 = std::make_shared<Conv2d>(3, 64, 7, 7, 2, 3);
        bn1 = std::make_shared<BatchNorm2d>(64);

        std::vector<int> channels = {64, 128, 256, 512};
        int in_channels = 64;

        for (size_t i = 0; i < num_blocks.size(); ++i) {
            int num_layers = num_blocks[i];
            int out_channels = channels[i];
            int stride = (i == 0) ? 1 : 2;
            layers.push_back(std::make_shared<ResidualBlock>(in_channels, out_channels, stride));
            for (int j = 1; j < num_layers; ++j) {
                layers.push_back(std::make_shared<ResidualBlock>(out_channels, out_channels));
            }
            in_channels = out_channels;
        }

        fc = std::make_shared<Linear>(512, num_classes);
    }

    /**
     * Forward pass of the ResNet model.
     * @param x Input tensor of shape `[N, 3, H, W]`.
     * @return Output tensor (logits) of shape `[N, num_classes]`.
     */
    Tensor forward(const Tensor& x) override {
        Tensor out = conv1->forward(x);
        out = bn1->forward(out);
        out = F::relu(out);

        for (auto& layer : layers) {
            out = layer->forward(out);
        }

        // Global Average Pooling (simplified as a flatten operation)
        Tensor pooled = Tensor({out.shape[0], out.shape[1]}); // [N, 512]
        for (int i = 0; i < out.shape[0]; ++i) {
            for (int j = 0; j < out.shape[1]; ++j) {
                float sum = 0.0f;
                for (int h = 0; h < out.shape[2]; ++h) {
                    for (int w = 0; w < out.shape[3]; ++w) {
                        sum += out.data[i * out.shape[1] * out.shape[2] * out.shape[3] + j * out.shape[2] * out.shape[3] + h * out.shape[3] + w];
                    }
                }
                pooled.data[i * out.shape[1] + j] = sum / (out.shape[2] * out.shape[3]);
            }
        }

        out = fc->forward(pooled);
        return out;
    }
};

} // namespace MyDL

#endif // MYDL_RESNET_H
