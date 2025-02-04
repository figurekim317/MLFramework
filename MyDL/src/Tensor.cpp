#include "MyDL/Tensor.h"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <limits>

namespace MyDL {

/** Default constructor: initializes an empty tensor */
Tensor::Tensor() {}

/** Returns the total number of elements in the tensor */
int Tensor::getTotalSize() const {
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

/** Reshapes the tensor */
void Tensor::reshape(const std::vector<int>& newShape) {
    int newTotalSize = std::accumulate(newShape.begin(), newShape.end(), 1, std::multiplies<int>());
    if (newTotalSize != getTotalSize()) {
        throw std::invalid_argument("New shape must match the total number of elements.");
    }
    shape = newShape;
}

/** Prints the tensor details */
void Tensor::print() const {
    std::cout << "Tensor Shape: [ ";
    for (int s : shape) std::cout << s << " ";
    std::cout << "]\nData: [ ";
    for (float val : data) std::cout << val << " ";
    std::cout << "]\n";
}

/** Performs element-wise addition */
Tensor Tensor::operator+(const Tensor& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Shape mismatch in tensor addition.");
    }
    Tensor result(shape);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

/** Applies ReLU activation */
Tensor Tensor::relu() const {
    Tensor result(shape);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = std::max(0.0f, data[i]);
    }
    return result;
}

/** Applies Softmax to the last dimension */
Tensor Tensor::softmax() const {
    if (shape.empty()) {
        throw std::runtime_error("Cannot apply softmax to an empty tensor.");
    }

    int lastDimSize = shape.back();
    int numGroups = getTotalSize() / lastDimSize;

    Tensor result(shape);
    for (int g = 0; g < numGroups; ++g) {
        float maxVal = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < lastDimSize; ++i) {
            maxVal = std::max(maxVal, data[g * lastDimSize + i]);
        }
        
        float sumExp = 0.0f;
        for (int i = 0; i < lastDimSize; ++i) {
            result.data[g * lastDimSize + i] = std::exp(data[g * lastDimSize + i] - maxVal);
            sumExp += result.data[g * lastDimSize + i];
        }

        for (int i = 0; i < lastDimSize; ++i) {
            result.data[g * lastDimSize + i] /= sumExp;
        }
    }
    return result;
}

} // namespace MyDL
