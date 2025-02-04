#ifndef MYDL_TRANSFORMER_H
#define MYDL_TRANSFORMER_H

#include <vector>
#include <cmath>
#include <memory>
#include "Module.h"
#include "Functional.h"
#include "Linear.h"

namespace MyDL {

/**
 * MultiHeadAttention class: Implements scaled dot-product attention.
 * This simplified version treats the entire input as a single head.
 * For a real multi-head version, the input should be split into multiple heads.
 */
class MultiHeadAttention : public Module {
private:
    int embed_dim_;
    int num_heads_;

    // Linear layers for Query (Q), Key (K), Value (V), and output projection (Wo)
    std::unique_ptr<Linear> Wq, Wk, Wv, Wo;

public:
    /**
     * Constructor to initialize multi-head attention parameters.
     * @param embed_dim The embedding dimension.
     * @param num_heads The number of attention heads (currently not split).
     */
    MultiHeadAttention(int embed_dim, int num_heads)
        : embed_dim_(embed_dim), num_heads_(num_heads),
          Wq(std::make_unique<Linear>(embed_dim_, embed_dim_)),
          Wk(std::make_unique<Linear>(embed_dim_, embed_dim_)),
          Wv(std::make_unique<Linear>(embed_dim_, embed_dim_)),
          Wo(std::make_unique<Linear>(embed_dim_, embed_dim_)) {}

    /**
     * Forward pass of multi-head attention.
     * @param x Input tensor of shape [seq_len, embed_dim].
     * @return Output tensor after self-attention and projection.
     */
    Tensor forward(const Tensor& x) override {
        // Compute Q, K, V
        Tensor Q = Wq->forward(x);
        Tensor K = Wk->forward(x);
        Tensor V = Wv->forward(x);

        // Transpose K for dot-product attention
        Tensor K_T = F::transpose(K);

        // Compute scaled dot-product attention: scores = QK^T / sqrt(embed_dim)
        Tensor scores = F::matmul(Q, K_T);
        float scale = 1.0f / std::sqrt(static_cast<float>(embed_dim_));
        for (size_t i = 0; i < scores.size(); i++) {
            scores.data[i] *= scale;
        }

        // Apply softmax to get attention weights
        scores = F::softmax(scores);

        // Compute attention output: attn_out = scores * V
        Tensor attn_out = F::matmul(scores, V);

        // Apply final projection
        return Wo->forward(attn_out);
    }
};


/**
 * FeedForward class: Implements a simple two-layer MLP with ReLU activation.
 * Structure: Linear(embed_dim -> ff_dim) -> ReLU -> Linear(ff_dim -> embed_dim)
 */
class FeedForward : public Module {
private:
    std::unique_ptr<Linear> linear1, linear2;

public:
    /**
     * Constructor to initialize the feedforward network.
     * @param embed_dim Input and output dimension.
     * @param ff_dim Hidden dimension (usually larger than embed_dim).
     */
    FeedForward(int embed_dim, int ff_dim)
        : linear1(std::make_unique<Linear>(embed_dim, ff_dim)),
          linear2(std::make_unique<Linear>(ff_dim, embed_dim)) {}

    /**
     * Forward pass of the feedforward network.
     * @param x Input tensor.
     * @return Output tensor after two linear layers and ReLU.
     */
    Tensor forward(const Tensor& x) override {
        Tensor hidden = linear1->forward(x);
        hidden = F::relu(hidden);
        return linear2->forward(hidden);
    }
};


/**
 * TransformerBlock class: A single layer of a Transformer model.
 * Structure: (MultiHeadAttention + Add & Norm) -> (FeedForward + Add & Norm)
 * Note: LayerNorm is not implemented in this simplified version.
 */
class TransformerBlock : public Module {
private:
    std::unique_ptr<MultiHeadAttention> mha;
    std::unique_ptr<FeedForward> ffn;

public:
    /**
     * Constructor for a single Transformer block.
     * @param embed_dim Embedding dimension.
     * @param num_heads Number of attention heads.
     * @param ff_dim Dimension of the feedforward network.
     */
    TransformerBlock(int embed_dim, int num_heads, int ff_dim)
        : mha(std::make_unique<MultiHeadAttention>(embed_dim, num_heads)),
          ffn(std::make_unique<FeedForward>(embed_dim, ff_dim)) {}

    /**
     * Forward pass of the Transformer block.
     * @param x Input tensor.
     * @return Output tensor after attention, feedforward, and residual connections.
     */
    Tensor forward(const Tensor& x) override {
        // Self-attention
        Tensor attn_out = mha->forward(x);

        // Residual connection 1: Add input and attention output
        Tensor x1 = F::add(x, attn_out);

        // Feedforward network
        Tensor ff_out = ffn->forward(x1);

        // Residual connection 2: Add attention result and feedforward output
        return F::add(x1, ff_out);
    }
};


/**
 * TransformerModel class: Stacks multiple Transformer blocks.
 */
class TransformerModel : public Module {
private:
    std::vector<std::unique_ptr<TransformerBlock>> blocks_;

public:
    /**
     * Constructor for a Transformer model.
     * @param num_layers Number of Transformer blocks.
     * @param embed_dim Embedding dimension.
     * @param num_heads Number of attention heads.
     * @param ff_dim Feedforward hidden dimension.
     */
    TransformerModel(int num_layers, int embed_dim, int num_heads, int ff_dim) {
        for (int i = 0; i < num_layers; i++) {
            blocks_.emplace_back(std::make_unique<TransformerBlock>(embed_dim, num_heads, ff_dim));
        }
    }

    /**
     * Forward pass of the Transformer model.
     * @param x Input tensor.
     * @return Output tensor after all Transformer blocks.
     */
    Tensor forward(const Tensor& x) override {
        Tensor out = x;
        for (auto& block : blocks_) {
            out = block->forward(out);
        }
        return out;
    }
};

} // namespace MyDL

#endif // MYDL_TRANSFORMER_H
