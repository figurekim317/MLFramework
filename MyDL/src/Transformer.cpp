#include "MyDL/Transformer.h"
#include "MyDL/Functional.h"

#include <cmath>
#include <iostream>
#include <limits>

namespace MyDL {

/**
 * Forward pass of PositionalEncoding.
 * Adds positional encoding to the input sequence to retain order information.
 * @param x Input tensor of shape [seq_len, embed_dim].
 * @return Output tensor with positional encoding added.
 */
Tensor PositionalEncoding::forward(const Tensor& x) {
    if (x.shape.size() != 2 || x.shape[0] != pos_encoding_.shape[0]) {
        throw std::runtime_error("PositionalEncoding: Input shape mismatch!");
    }
    return F::add(x, pos_encoding_);
}

/**
 * Forward pass of MultiHeadAttention.
 * @param x Input tensor of shape [seq_len, embed_dim].
 * @return Output tensor after self-attention and projection.
 */
Tensor MultiHeadAttention::forward(const Tensor& x) {
    // Compute Query, Key, and Value projections
    Tensor Q = Wq->forward(x);
    Tensor K = Wk->forward(x);
    Tensor V = Wv->forward(x);

    // Transpose K: Convert shape from [seq_len, embed_dim] to [embed_dim, seq_len]
    Tensor K_T = F::transpose(K);

    // Compute scaled dot-product attention: scores = QK^T / sqrt(embed_dim)
    Tensor scores = F::matmul(Q, K_T);
    float scale = 1.0f / std::sqrt(static_cast<float>(embed_dim_));
    for (size_t i = 0; i < scores.size(); i++) {
        scores.data[i] *= scale;
    }

    // Apply softmax to obtain attention weights
    scores = F::softmax(scores);

    // Compute attention output: attn_out = scores * V
    Tensor attn_out = F::matmul(scores, V);

    // Apply output projection
    return Wo->forward(attn_out);
}

/**
 * Forward pass of FeedForward network.
 * Applies two linear transformations with a ReLU activation in between.
 * @param x Input tensor.
 * @return Output tensor after the feedforward network.
 */
Tensor FeedForward::forward(const Tensor& x) {
    Tensor hidden = linear1->forward(x);
    hidden = F::relu(hidden);
    return linear2->forward(hidden);
}

/**
 * Forward pass of a single TransformerBlock.
 * Applies multi-head attention and feedforward network with residual connections.
 * @param x Input tensor.
 * @return Output tensor after the Transformer block.
 */
Tensor TransformerBlock::forward(const Tensor& x) {
    // Multi-head attention
    Tensor attn_out = mha->forward(x);

    // Residual connection: Add input and attention output
    Tensor x1 = F::add(x, attn_out);

    // Feedforward network
    Tensor ff_out = ffn->forward(x1);

    // Residual connection: Add attention output and feedforward output
    return F::add(x1, ff_out);
}

/**
 * Forward pass of the Transformer model.
 * Applies positional encoding followed by multiple TransformerBlocks sequentially.
 * @param x Input tensor.
 * @return Output tensor after all Transformer blocks.
 */
Tensor TransformerModel::forward(const Tensor& x) {
    // Apply positional encoding before processing through Transformer blocks
    Tensor out = pos_encoding_->forward(x);
    
    // Pass through Transformer blocks
    for (auto& block : blocks_) {
        out = block->forward(out);
    }
    return out;
}

} // namespace MyDL
