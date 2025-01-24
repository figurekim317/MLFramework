#include <iostream>
#include <memory>

#include "MyDL/Tensor.h"
#include "MyDL/VisionTransformer.h"
#include "MyDL/WeightsLoader.h"

int main() {
    // 1) 모델 생성
    int num_layers  = 2;
    int embed_dim   = 16;
    int num_heads   = 2;
    int ff_dim      = 32;
    int num_classes = 10;

    std::shared_ptr<MyDL::Module> model = std::make_shared<MyDL::VisionTransformer>(
        num_layers, embed_dim, num_heads, ff_dim, num_classes
    );

    // 2) 가중치 로드 (실제 구현 시 weights 파일 필요)
    MyDL::loadWeights(model, "vit_weights.bin");

    // 3) 추론 테스트용 입력 (seq_len=4, embed_dim=16)
    MyDL::Tensor input({4, embed_dim});
    for (size_t i = 0; i < input.size(); i++) {
        input.data[i] = 0.01f * (i+1);
    }

    // 4) forward
    MyDL::Tensor logits = model->forward(input);

    // 5) 결과 출력
    std::cout << "Logits output: ";
    for (auto val : logits.data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
