#ifndef MYDL_FUNCTIONAL_H
#define MYDL_FUNCTIONAL_H

#include "Tensor.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace MyDL {
namespace F {

// 2D matmul: A=[N,M], B=[M,P] => C=[N,P]
inline Tensor matmul(const Tensor& A, const Tensor& B) {
    if (A.shape.size() != 2 || B.shape.size() != 2) {
        throw std::runtime_error("matmul: only supports 2D Tensors");
    }
    int N = A.shape[0], M = A.shape[1];
    int M2 = B.shape[0], P = B.shape[1];
    if (M != M2) {
        throw std::runtime_error("matmul: dimension mismatch");
    }

    Tensor C({N, P});
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < M; ++k) {
                sum += A.data[i*M + k] * B.data[k*P + j];
            }
            C.data[i*P + j] = sum;
        }
    }
    return C;
}

// element-wise add (same shape)
inline Tensor add(const Tensor& A, const Tensor& B) {
    if (A.size() != B.size()) {
        throw std::runtime_error("add: size mismatch");
    }
    Tensor C = A;
    for (size_t i = 0; i < C.size(); i++) {
        C.data[i] += B.data[i];
    }
    return C;
}

// ReLU
inline Tensor relu(const Tensor& x) {
    Tensor y = x;
    for (auto &v : y.data) {
        if (v < 0.0f) v = 0.0f;
    }
    return y;
}

// softmax (row-wise)
inline Tensor softmax(const Tensor& x) {
    // x shape: [N, dim]
    if (x.shape.size() != 2) {
        throw std::runtime_error("softmax: only supports 2D");
    }
    int N = x.shape[0];
    int dim = x.shape[1];
    Tensor y({N, dim});

    for (int i = 0; i < N; i++) {
        float maxVal = -1e30f;
        for (int d = 0; d < dim; d++) {
            float val = x.data[i*dim + d];
            if (val > maxVal) maxVal = val;
        }
        float sumExp = 0.0f;
        for (int d = 0; d < dim; d++) {
            sumExp += std::exp(x.data[i*dim + d] - maxVal);
        }
        for (int d = 0; d < dim; d++) {
            y.data[i*dim + d] = std::exp(x.data[i*dim + d] - maxVal) / sumExp;
        }
    }
    return y;
}

} // namespace F
} // namespace MyDL

#endif // MYDL_FUNCTIONAL_H
