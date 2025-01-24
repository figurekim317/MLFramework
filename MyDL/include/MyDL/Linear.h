#ifndef MYDL_LINEAR_H
#define MYDL_LINEAR_H

#include <vector>
#include <stdexcept>
#include "Module.h"
#include "Functional.h"

namespace MyDL {

// Linear Layer: y = xW + b
class Linear : public Module {
private:
    Tensor W; // [in_features, out_features]
    Tensor b; // [1, out_features]

public:
    Linear(int in_features, int out_features) {
        W = Tensor({in_features, out_features});
        b = Tensor({1, out_features});
        // 간단 초기값(데모용)
        for (size_t i = 0; i < W.size(); i++) {
            W.data[i] = 0.01f;
        }
        // bias는 0
    }

    // 학습된 weight/bias 로드할 때 사용 가능
    void setWeights(const std::vector<float>& w_data) {
        if (w_data.size() != W.size()) {
            throw std::runtime_error("Linear::setWeights: size mismatch");
        }
        W.data = w_data;
    }

    void setBias(const std::vector<float>& b_data) {
        if (b_data.size() != (size_t)W.shape[1]) {
            throw std::runtime_error("Linear::setBias: size mismatch");
        }
        for (int i = 0; i < W.shape[1]; i++) {
            b.data[i] = b_data[i];
        }
    }

    Tensor forward(const Tensor& x) override {
        // x shape: [N, in_features]
        // W shape: [in_features, out_features]
        // out = xW + b
        Tensor out = F::matmul(x, W);
        // broadcast add b
        int N = out.shape[0];
        int out_feats = out.shape[1];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < out_feats; j++) {
                out.data[i*out_feats + j] += b.data[j];
            }
        }
        return out;
    }
};

} // namespace MyDL

#endif // MYDL_LINEAR_H
