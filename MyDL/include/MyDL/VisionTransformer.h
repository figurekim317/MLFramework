#ifndef MYDL_VISION_TRANSFORMER_H
#define MYDL_VISION_TRANSFORMER_H

#include "Module.h"
#include "Transformer.h"
#include "Linear.h"

namespace MyDL {

/**
 * Vision Transformer (ViT) Implementation
 *
 * Vision Transformer (ViT) is a deep learning model designed for image classification 
 * using transformer-based architecture instead of CNNs. It divides an image into 
 * patches and processes them as a sequence.
 *
 * **Structure:**
 * - **Input Tensor:** Assumed to be `[seq_len, embed_dim]`, where `seq_len` is the 
 *   number of patches, and `embed_dim` is the feature dimension per patch.
 * - **Transformer Encoder:** Multiple stacked Transformer blocks (self-attention + FFN).
 * - **CLS Token Extraction:** The first token (representing the entire image) is used for classification.
 * - **Final Classifier:** A fully connected layer that maps CLS token output to class logits.
 *
 * **Key Features:**
 * - Supports configurable Transformer layers, heads, and embedding dimensions.
 * - Implements the core ViT structure without patch embedding (assumes preprocessed input).
 * - Uses a CLS token to aggregate sequence information.
 */
class VisionTransformer : public Module {
private:
    std::shared_ptr<TransformerModel> transformer_;
    std::shared_ptr<Linear> classifier_;
    int embed_dim_;
    int num_classes_;
    int seq_len_;

public:
    /**
     * Constructor for Vision Transformer.
     * @param num_layers Number of Transformer blocks in the encoder.
     * @param embed_dim Feature dimension per token (patch embedding size).
     * @param num_heads Number of self-attention heads.
     * @param ff_dim Feedforward hidden dimension.
     * @param num_classes Number of output classes.
     */
    VisionTransformer(int num_layers, int embed_dim, int num_heads, int ff_dim, int num_classes)
        : embed_dim_(embed_dim), num_classes_(num_classes) 
    {
        transformer_ = std::make_shared<TransformerModel>(num_layers, embed_dim_, num_heads, ff_dim);
        classifier_  = std::make_shared<Linear>(embed_dim_, num_classes_);
    }

    /**
     * Forward pass of the Vision Transformer.
     * @param x Input tensor of shape `[seq_len, embed_dim]`, representing patch embeddings.
     * @return Output tensor of shape `[1, num_classes]`, representing class logits.
     */
    Tensor forward(const Tensor& x) override {
        if (x.shape.size() != 2 || x.shape[0] < 1) {
            throw std::runtime_error("VisionTransformer: Invalid input shape. Expected [seq_len, embed_dim].");
        }

        // 1) Transformer Encoder
        Tensor out = transformer_->forward(x);

        // 2) Extract CLS Token (first token in sequence)
        Tensor cls_token({1, embed_dim_});
        for (int i = 0; i < embed_dim_; i++) {
            cls_token.data[i] = out.data[i];  // Copy first row (CLS token)
        }

        // 3) Fully Connected Classifier
        Tensor logits = classifier_->forward(cls_token);  // shape=[1, num_classes]
        return logits;
    }

    /**
     * Prints the structure and configuration of the Vision Transformer.
     */
    void printModuleInfo() const override {
        std::cout << "VisionTransformer Model:\n";
        std::cout << "  Embed Dim: " << embed_dim_ << "\n";
        std::cout << "  Num Classes: " << num_classes_ << "\n";
        transformer_->printModuleInfo();
        classifier_->printModuleInfo();
    }
};

} // namespace MyDL

#endif // MYDL_VISION_TRANSFORMER_H
