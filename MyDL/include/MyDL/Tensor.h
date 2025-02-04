#ifndef MYDL_TENSOR_H
#define MYDL_TENSOR_H

#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <numeric> // For std::accumulate

namespace MyDL {

class Tensor {
public:
    std::vector<float> data; // Flat buffer storing tensor elements
    std::vector<int> shape;  // Shape of the tensor (e.g., {batch, channels, height, width})

    /** Default constructor (empty tensor) */
    Tensor() {}

    /** 
     * Constructor that initializes a tensor with a given shape. 
     * The data buffer is initialized with zeros.
     * @param shape_ Vector representing the shape of the tensor.
     */
    Tensor(const std::vector<int>& shape_) : shape(shape_) {
        int totalSize = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
        data.resize(totalSize, 0.0f);
    }

    /**
     * Constructor that initializes a tensor with a given shape and data values.
     * @param shape_ Shape of the tensor.
     * @param values Initial data for the tensor (should match total number of elements).
     */
    Tensor(const std::vector<int>& shape_, const std::vector<float>& values) : shape(shape_), data(values) {
        if (data.size() != getTotalSize()) {
            throw std::invalid_argument("Data size does not match the specified shape.");
        }
    }

    /** Returns the total number of elements in the tensor */
    size_t size() const { return data.size(); }

    /** Returns the total number of elements computed from the shape */
    int getTotalSize() const {
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    }

    /** Overloaded operator to access elements by index */
    float& operator[](size_t idx) { return data[idx]; }
    const float& operator[](size_t idx) const { return data[idx]; }

    /**
     * Reshapes the tensor. The new shape must have the same number of elements.
     * @param newShape The target shape to which the tensor should be reshaped.
     */
    void reshape(const std::vector<int>& newShape) {
        int newTotalSize = std::accumulate(newShape.begin(), newShape.end(), 1, std::multiplies<int>());
        if (newTotalSize != getTotalSize()) {
            throw std::invalid_argument("New shape must match the total number of elements.");
        }
        shape = newShape;
    }

    /** Prints the tensor shape and data */
    void print() const {
        std::cout << "Tensor Shape: [ ";
        for (int s : shape) std::cout << s << " ";
        std::cout << "]\nData: [ ";
        for (float val : data) std::cout << val << " ";
        std::cout << "]\n";
    }

    /**
     * Performs element-wise addition with another tensor.
     * Both tensors must have the same shape.
     * @param other Tensor to add.
     * @return Resulting tensor.
     */
    Tensor operator+(const Tensor& other) const {
        if (shape != other.shape) {
            throw std::invalid_argument("Shape mismatch in tensor addition.");
        }
        Tensor result(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    /**
     * Applies the ReLU activation function element-wise.
     * @return A new tensor with ReLU applied.
     */
    Tensor relu() const {
        Tensor result(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = std::max(0.0f, data[i]);
        }
        return result;
    }

    /**
     * Applies softmax to the last dimension of the tensor.
     * @return A new tensor with softmax applied.
     */
    Tensor softmax() const {
        if (shape.empty()) {
            throw std::runtime_error("Cannot apply softmax to an empty tensor.");
        }

        int lastDimSize = shape.back(); // Last dimension size
        int numGroups = getTotalSize() / lastDimSize; // Number of groups for softmax application

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
};

} // namespace MyDL

#endif // MYDL_TENSOR_H
