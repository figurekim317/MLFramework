#ifndef MYDL_TENSOR_H
#define MYDL_TENSOR_H

#include <vector>
#include <cmath>
#include <stdexcept>

namespace MyDL {

class Tensor {
public:
    std::vector<float> data;
    std::vector<int> shape; // 예: {N,C,H,W} 또는 {seq_len, embed_dim}

    Tensor() {}

    // 특정 shape로 초기화, data는 0으로
    Tensor(const std::vector<int>& shape_) : shape(shape_) {
        int totalSize = 1;
        for (int s : shape_) totalSize *= s;
        data.resize(totalSize, 0.0f);
    }

    // 전체 원소 개수
    size_t size() const { return data.size(); }

    float& operator[](size_t idx) { return data[idx]; }
    const float& operator[](size_t idx) const { return data[idx]; }
};

} // namespace MyDL

#endif // MYDL_TENSOR_H
