#ifndef MYDL_VISION_TRANSFORMER_H
#define MYDL_VISION_TRANSFORMER_H

#include "Module.h"
#include "Transformer.h"
#include "Linear.h"

namespace MyDL {

// 간단한 VisionTransformer 예시
// (PatchEmbedding은 생략하고, [seq_len, embed_dim] Tensor가 들어온다고 가정)
class VisionTransformer : public Module {
private:
    TransformerModel* transformer_;
    Linear* classifier_;
    int embed_dim_;
    int num_classes_;

public:
    VisionTransformer(int num_layers, int embed_dim, int num_heads, int ff_dim, int num_classes)
        : embed_dim_(embed_dim), num_classes_(num_classes)
    {
        transformer_ = new TransformerModel(num_layers, embed_dim_, num_heads, ff_dim);
        classifier_  = new Linear(embed_dim_, num_classes_);
    }

    ~VisionTransformer() {
        delete transformer_;
        delete classifier_;
    }

    Tensor forward(const Tensor& x) override {
        // x: [seq_len, embed_dim]
        // 1) Transformer
        Tensor out = transformer_->forward(x);

        // 2) CLS token = out[0, :]
        //   여기서는 seq_len의 첫 번째 row를 사용
        if (out.shape[0] < 1) {
            throw std::runtime_error("VisionTransformer: input seq_len < 1!");
        }

        Tensor cls_token({1, embed_dim_});
        for (int i = 0; i < embed_dim_; i++) {
            cls_token.data[i] = out.data[i]; // 첫 row 복사
        }

        // 3) Classifier
        Tensor logits = classifier_->forward(cls_token); // shape=[1, num_classes]
        return logits;
    }
};

} // namespace MyDL

#endif // MYDL_VISION_TRANSFORMER_H
