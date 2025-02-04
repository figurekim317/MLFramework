#include "MyDL/ResNet.h"
#include <iostream>

namespace MyDL {

/**
 * Forward pass of the residual block.
 * Applies two convolutions, batch normalization, ReLU, and a skip connection.
 */
Tensor ResidualBlock::forward(const Tensor& x) {
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

/**
 * Forward pass of the full ResNet model.
 * Applies an initial convolution, multiple residual blocks, and a final FC layer.
 */
Tensor ResNet::forward(const Tensor& x) {
    Tensor out = conv1->forward(x);
    out = bn1->forward(out);
    out = F::relu(out);

    for (auto& layer : layers) {
        out = layer->forward(out);
    }

    // Global Average Pooling
    Tensor pooled = Tensor({out.shape[0], out.shape[1]});
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

} // namespace MyDL
