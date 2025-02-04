#ifndef MYDL_FUNCTIONAL_H
#define MYDL_FUNCTIONAL_H

#include "Tensor.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <limits>

namespace MyDL {
namespace F {

/**
 * Functional Module for Basic Tensor Operations
 * 
 * The `Functional` module (`MyDL::F`) provides a collection of fundamental 
 * operations commonly used in deep learning frameworks. These functions 
 * perform essential mathematical computations on tensors, such as:
 * 
 * - **Matrix Multiplication (`matmul`)**: Computes the dot product of two 2D tensors.
 * - **Element-wise Addition (`add`)**: Performs addition between two tensors of the same shape.
 * - **Activation Functions**:
 *   - **ReLU (`relu`)**: Applies the Rectified Linear Unit activation function.
 *   - **Softmax (`softmax`)**: Converts values into probabilities along the last dimension.
 * - **Tensor Manipulation**:
 *   - **Transpose (`transpose`)**: Swaps the dimensions of a 2D tensor.
 * 
 * These functions are optimized for inference-only deep learning applications
 * and serve as the core building blocks for operations inside `Module` layers 
 * (such as `Linear`, `MultiHeadAttention`, `BatchNorm2d`, etc.).
 */

/**
 * Performs matrix multiplication between two 2D tensors.
 * @param A First input tensor with shape [M, K].
 * @param B Second input tensor with shape [K, N].
 * @return Resulting tensor with shape [M, N].
 */
Tensor matmul(const Tensor& A, const Tensor& B);

/**
 * Element-wise addition of two tensors.
 * @param A First input tensor.
 * @param B Second input tensor (must have the same shape as A).
 * @return Tensor resulting from element-wise addition.
 */
Tensor add(const Tensor& A, const Tensor& B);

/**
 * Applies the ReLU (Rectified Linear Unit) activation function element-wise.
 * ReLU(x) = max(0, x)
 * @param x Input tensor.
 * @return Output tensor with ReLU applied.
 */
Tensor relu(const Tensor& x);

/**
 * Applies softmax activation along the last dimension of the tensor.
 * @param x Input tensor.
 * @return Tensor with softmax applied.
 */
Tensor softmax(const Tensor& x);

/**
 * Transposes a 2D tensor (matrix) by swapping its rows and columns.
 * @param x Input tensor with shape [M, N].
 * @return Transposed tensor with shape [N, M].
 */
Tensor transpose(const Tensor& x);

} // namespace F
} // namespace MyDL

#endif // MYDL_FUNCTIONAL_H
