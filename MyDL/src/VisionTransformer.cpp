#include "MyDL/VisionTransformer.h"
#include <iostream>
#include <stdexcept>

namespace MyDL {

/**
 * Forward pass of the Vision Transformer model.
 * Processes input token sequence using a Transformer encoder and classifies using the CLS token.
 */
Tensor VisionTransformer::forward(const Tensor& x) {
    if (x.shape.size() != 2 || x.shape[0] < 1) {
        throw std::runtime_error("VisionTransformer: Invalid input shape. Expected [seq_len, embed_dim].");
    }

    // 1) Transformer Encoder
    Tensor out = transformer_->forward(x);

    // 2) Extract CLS Token (first row in the sequence)
    Tensor cls_token({1, embed_dim_});
    for (int i = 0; i < embed_dim_; i++) {
        cls_token.data[i] = out.data[i];  // Copy first row (CLS token)
    }

    // 3) Fully Connected Classifier
    return classifier_->forward(cls_token);  // shape=[1, num_classes]
}

/**
 * Prints the Vision Transformer model configuration.
 */
void VisionTransformer::printModuleInfo() const {
    std::cout << "VisionTransformer Model:\n";
    std::cout << "  Embed Dim: " << embed_dim_ << "\n";
    std::cout << "  Num Classes: " << num_classes_ << "\n";
    transformer_->printModuleInfo();
    classifier_->printModuleInfo();
}

} // namespace MyDL
