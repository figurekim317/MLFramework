#ifndef MYDL_WEIGHTSLOADER_H
#define MYDL_WEIGHTSLOADER_H

#include <string>
#include <memory>
#include <iostream>
#include "Module.h"

namespace MyDL {

// 가중치 로딩 함수 (구현 예시)
// 실제로는 파일 파싱 + 각 Layer의 setWeights/setBias를 호출해야 함.
inline bool loadWeights(std::shared_ptr<Module> model, const std::string& path) {
    // 데모: 실제 구현 없이 "로드했다" 가정
    std::cout << "[loadWeights] Loading weights from: " << path << std::endl;
    // TODO: 모델 내 각 레이어를 순회하며 파라미터를 매칭하고 세팅하는 로직 필요
    return true;
}

} // namespace MyDL

#endif // MYDL_WEIGHTSLOADER_H
