#ifndef MYDL_TRANSFORMER_H
#define MYDL_TRANSFORMER_H

#include <vector>
#include <cmath>
#include <memory>
#include "Module.h"
#include "Functional.h"
#include "Linear.h"

namespace MyDL {

// MultiHeadAttention (단순화)
class MultiHeadAttention : public Module {
private:
    int embed_dim_;
    int num_heads_;

    // Q, K, V, Output projection
    Linear *Wq, *Wk, *Wv, *Wo;

public:
    MultiHeadAttention(int embed_dim, int num_heads)
        : embed_dim_(embed_dim), num_heads_(num_heads)
    {
        // 단일 head처럼 사용
        Wq = new Linear(embed_dim_, embed_dim_);
        Wk = new Linear(embed_dim_, embed_dim_);
        Wv = new Linear(embed_dim_, embed_dim_);
        Wo = new Linear(embed_dim_, embed_dim_);
    }

    ~MultiHeadAttention() {
        delete Wq; delete Wk; delete Wv; delete Wo;
    }

    Tensor forward(const Tensor& x) override {
        // x: [seq_len, embed_dim]
        Tensor Q = Wq->forward(x);
        Tensor K = Wk->forward(x);
        Tensor V = Wv->forward(x);

        // K^T
        int seq_len = K.shape[0];
        int dim = K.shape[1];
        Tensor K_T({dim, seq_len});
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < dim; j++) {
                K_T.data[j*seq_len + i] = K.data[i*dim + j];
            }
        }

        // scores = QK^T / sqrt(dim)
        Tensor scores = F::matmul(Q, K_T);
        float scale = 1.0f / std::sqrt((float)dim);
        for (size_t i = 0; i < scores.size(); i++) {
            scores.data[i] *= scale;
        }

        // softmax
        scores = F::softmax(scores);

        // attn_out = scores * V
        Tensor attn_out = F::matmul(scores, V);

        // project
        Tensor out = Wo->forward(attn_out);
        return out; // [seq_len, embed_dim]
    }
};


// FeedForward: Linear->ReLU->Linear
class FeedForward : public Module {
private:
    Linear *linear1;
    Linear *linear2;

public:
    FeedForward(int embed_dim, int hidden_dim_ff) {
        linear1 = new Linear(embed_dim, hidden_dim_ff);
        linear2 = new Linear(hidden_dim_ff, embed_dim);
    }
    ~FeedForward() {
        delete linear1; 
        delete linear2;
    }

    Tensor forward(const Tensor& x) override {
        Tensor hidden = linear1->forward(x);
        hidden = F::relu(hidden);
        Tensor out = linear2->forward(hidden);
        return out;
    }
};


// TransformerBlock: (MHA + Add&Norm) + (FFN + Add&Norm)
class TransformerBlock : public Module {
private:
    MultiHeadAttention *mha;
    FeedForward *ffn;

    // LayerNorm은 여기서는 생략/단순화
public:
    TransformerBlock(int embed_dim, int num_heads, int ff_dim) {
        mha = new MultiHeadAttention(embed_dim, num_heads);
        ffn = new FeedForward(embed_dim, ff_dim);
    }
    ~TransformerBlock() {
        delete mha;
        delete ffn;
    }

    Tensor forward(const Tensor& x) override {
        // x -> MHA
        Tensor attn_out = mha->forward(x);

        // Add
        Tensor x1 = F::add(x, attn_out);

        // FFN
        Tensor ff_out = ffn->forward(x1);

        // Add
        Tensor x2 = F::add(x1, ff_out);

        return x2;
    }
};


// TransformerModel: N개의 Block
class TransformerModel : public Module {
private:
    std::vector<TransformerBlock*> blocks_;
public:
    TransformerModel(int num_layers, int embed_dim, int num_heads, int ff_dim) {
        for (int i = 0; i < num_layers; i++) {
            blocks_.push_back(new TransformerBlock(embed_dim, num_heads, ff_dim));
        }
    }
    ~TransformerModel() {
        for (auto b : blocks_) {
            delete b;
        }
    }

    Tensor forward(const Tensor& x) override {
        Tensor out = x;
        for (auto block : blocks_) {
            out = block->forward(out);
        }
        return out;
    }
};

} // namespace MyDL

#endif // MYDL_TRANSFORMER_H
