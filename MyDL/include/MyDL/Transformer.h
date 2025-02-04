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
 * Positional Encoding for Transformer Models
 *
 * - **Purpose**: Transformers lack inherent sequential processing, so positional encoding
 *   adds spatial information to token embeddings.
 * - **Two types:**
 *   1. **Fixed Sinusoidal Encoding** (default): Uses sine/cosine functions.
 *   2. **Learnable Encoding** (optional): A trainable tensor for position embeddings.
 */
class PositionalEncoding : public Module {
private:
    Tensor pos_encoding_; // Stores computed positional embeddings
    bool learnable_;

public:
    /**
     * Constructor for Positional Encoding.
     * @param seq_len Maximum sequence length.
     * @param embed_dim Embedding dimension per token.
     * @param learnable If true, the position embedding is trainable.
     */
    PositionalEncoding(int seq_len, int embed_dim, bool learnable = false)
        : learnable_(learnable) 
    {
        pos_encoding_ = Tensor({seq_len, embed_dim});

        if (learnable_) {
            // If learnable, initialize randomly
            for (size_t i = 0; i < pos_encoding_.size(); i++) {
                pos_encoding_.data[i] = static_cast<float>(rand()) / RAND_MAX * 0.02f;
            }
        } else {
            // Sinusoidal encoding (fixed)
            for (int pos = 0; pos < seq_len; pos++) {
                for (int i = 0; i < embed_dim; i += 2) {
                    float div_term = std::pow(10000.0f, static_cast<float>(i) / embed_dim);
                    pos_encoding_.data[pos * embed_dim + i] = std::sin(pos / div_term);
                    if (i + 1 < embed_dim) {
                        pos_encoding_.data[pos * embed_dim + i + 1] = std::cos(pos / div_term);
                    }
                }
            }
        }
    }

    /**
     * Applies positional encoding to input tensor.
     * @param x Input tensor of shape `[seq_len, embed_dim]`.
     * @return Output tensor with positional encoding applied.
     */
    Tensor forward(const Tensor& x) override {
        if (x.shape.size() != 2 || x.shape[0] != pos_encoding_.shape[0]) {
            throw std::runtime_error("PositionalEncoding: Input shape mismatch!");
        }
        return F::add(x, pos_encoding_);
    }
};

/**
 * MultiHeadAttention class: Implements scaled dot-product attention.
 */
class MultiHeadAttention : public Module {
private:
    int embed_dim_;
    int num_heads_;

    std::unique_ptr<Linear> Wq, Wk, Wv, Wo;

public:
    MultiHeadAttention(int embed_dim, int num_heads)
        : embed_dim_(embed_dim), num_heads_(num_heads),
          Wq(std::make_unique<Linear>(embed_dim_, embed_dim_)),
          Wk(std::make_unique<Linear>(embed_dim_, embed_dim_)),
          Wv(std::make_unique<Linear>(embed_dim_, embed_dim_)),
          Wo(std::make_unique<Linear>(embed_dim_, embed_dim_)) {}

    Tensor forward(const Tensor& x) override {
        Tensor Q = Wq->forward(x);
        Tensor K = Wk->forward(x);
        Tensor V = Wv->forward(x);

        Tensor K_T = F::transpose(K);
        Tensor scores = F::matmul(Q, K_T);

        float scale = 1.0f / std::sqrt(static_cast<float>(embed_dim_));
        for (size_t i = 0; i < scores.size(); i++) {
            scores.data[i] *= scale;
        }

        scores = F::softmax(scores);
        Tensor attn_out = F::matmul(scores, V);
        return Wo->forward(attn_out);
    }
};

/**
 * FeedForward class: Implements a simple two-layer MLP with ReLU activation.
 */
class FeedForward : public Module {
private:
    std::unique_ptr<Linear> linear1, linear2;

public:
    FeedForward(int embed_dim, int ff_dim)
        : linear1(std::make_unique<Linear>(embed_dim, ff_dim)),
          linear2(std::make_unique<Linear>(ff_dim, embed_dim)) {}

    Tensor forward(const Tensor& x) override {
        Tensor hidden = linear1->forward(x);
        hidden = F::relu(hidden);
        return linear2->forward(hidden);
    }
};

/**
 * TransformerBlock class: A single layer of a Transformer model.
 */
class TransformerBlock : public Module {
private:
    std::unique_ptr<MultiHeadAttention> mha;
    std::unique_ptr<FeedForward> ffn;

public:
    TransformerBlock(int embed_dim, int num_heads, int ff_dim)
        : mha(std::make_unique<MultiHeadAttention>(embed_dim, num_heads)),
          ffn(std::make_unique<FeedForward>(embed_dim, ff_dim)) {}

    Tensor forward(const Tensor& x) override {
        Tensor attn_out = mha->forward(x);
        Tensor x1 = F::add(x, attn_out);
        Tensor ff_out = ffn->forward(x1);
        return F::add(x1, ff_out);
    }
};

/**
 * TransformerModel class: Stacks multiple Transformer blocks and applies positional encoding.
 */
class TransformerModel : public Module {
private:
    std::shared_ptr<PositionalEncoding> pos_encoding_;
    std::vector<std::unique_ptr<TransformerBlock>> blocks_;

public:
    TransformerModel(int num_layers, int embed_dim, int num_heads, int ff_dim, int seq_len)
    {
        pos_encoding_ = std::make_shared<PositionalEncoding>(seq_len, embed_dim);
        for (int i = 0; i < num_layers; i++) {
            blocks_.emplace_back(std::make_unique<TransformerBlock>(embed_dim, num_heads, ff_dim));
        }
    }

    Tensor forward(const Tensor& x) override {
        Tensor out = pos_encoding_->forward(x);
        for (auto& block : blocks_) {
            out = block->forward(out);
        }
        return out;
    }
};

} // namespace MyDL

#endif // MYDL_TRANSFORMER_H
