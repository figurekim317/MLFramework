#ifndef MYDL_MODULE_H
#define MYDL_MODULE_H

#include "Tensor.h"

namespace MyDL {

// PyTorch nn.Module 유사 추상 클래스
class Module {
public:
    virtual ~Module() {}

    // 모든 레이어/모델이 구현해야 하는 forward 인터페이스
    virtual Tensor forward(const Tensor& x) = 0;

    // 추후 parameter(), toDevice() 등 추가 가능
};

} // namespace MyDL

#endif // MYDL_MODULE_H
