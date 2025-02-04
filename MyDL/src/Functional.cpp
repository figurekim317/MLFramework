#include "MyDL/Functional.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <limits>

namespace MyDL {
namespace F {

/**
 * Performs matrix multiplication between two 2D tensors.
 * @param A First input tensor with shape [M, K].
 * @param B Second input tensor with shape [K, N].
 * @return Resulting tensor with shape [M, N].
 */
Tensor matmul(const Tensor& A, const Tensor& B) {
    if (A.shape.size() != 2 || B.shape.size() != 2) {
        throw std::invalid_argument("matmul: Both tensors must be 2D matrices.");
    }
    int M = A.shape[0], K = A.shape[1];
    int K_B = B.shape[0], N = B.shape[1];

    if (K != K_B) {
        throw std::invalid_argument("matmul: Inner dimensions must match (A.shape[1] == B.shape[0]).");
    }

    Tensor result({M, N});
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A.data[i * K + k] * B.data[k * N + j];
            }
            result.data[i * N + j] = sum;
        }
    }
    return result;
}

/**
 * Element-wise addition of two tensors.
 * @param A First input tensor.
 * @param B Second input tensor (must have the same shape as A).
 * @return Tensor resulting from element-wise addition.
 */
Tensor add(const Tensor& A, const Tensor& B) {
    if (A.shape != B.shape) {
        throw std::invalid_argument("add: Tensors must have the same shape.");
    }

    Tensor result(A.shape);
    for (size_t i = 0; i < A.size(); ++i) {
        result.data[i] = A.data[i] + B.data[i];
    }
    return result;
}

/**
 * Applies the ReLU activation function element-wise.
 * ReLU(x) = max(0, x)
 * @param x Input tensor.
 * @return Output tensor with ReLU applied.
 */
Tensor relu(const Tensor& x) {
    Tensor result(x.shape);
    for (size_t i = 0; i < x.size(); ++i) {
        result.data[i] = std::max(0.0f, x.data[i]);
    }
    return result;
}

/**
 * Applies softmax activation along the last dimension of the tensor.
 * @param x Input tensor.
 * @return Tensor with softmax applied.
 */
Tensor softmax(const Tensor& x) {
    if (x.shape.empty()) {
        throw std::runtime_error("softmax: Cannot apply softmax to an empty tensor.");
    }

    int lastDimSize = x.shape.back(); // Last dimension size
    int numGroups = x.size() / lastDimSize; // Number of groups for softmax application

    Tensor result(x.shape);
    for (int g = 0; g < numGroups; ++g) {
        float maxVal = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < lastDimSize; ++i) {
            maxVal = std::max(maxVal, x.data[g * lastDimSize + i]);
        }
        
        float sumExp = 0.0f;
        for (int i = 0; i < lastDimSize; ++i) {
            result.data[g * lastDimSize + i] = std::exp(x.data[g * lastDimSize + i] - maxVal);
            sumExp += result.data[g * lastDimSize + i];
        }

        for (int i = 0; i < lastDimSize; ++i) {
            result.data[g * lastDimSize + i] /= sumExp;
        }
    }
    return result;
}

/**
 * Transposes a 2D tensor (matrix) by swapping its rows and columns.
 * @param x Input tensor with shape [M, N].
 * @return Transposed tensor with shape [N, M].
 */
Tensor transpose(const Tensor& x) {
    if (x.shape.size() != 2) {
        throw std::invalid_argument("transpose: Input tensor must be a 2D matrix.");
    }

    int M = x.shape[0], N = x.shape[1];
    Tensor result({N, M});
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            result.data[j * M + i] = x.data[i * N + j];
        }
    }
    return result;
}

} // namespace F
} // namespace MyDL
