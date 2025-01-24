#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <stdexcept>

/***********************************************
 * 1) 간단한 Image 구조체
 *    - 실제론 OpenCV 등을 써서 Mat 형태 등을 사용.
 ************************************************/
struct Image {
    int width;
    int height;
    int channels; // 예: 3 (RGB)
    std::vector<float> data; // size = width * height * channels
};

/***********************************************
 * 2) 행렬 타입 정의 (seq_len x embed_dim 등의 2D)
 ************************************************/
typedef std::vector<std::vector<float>> Matrix;

/***********************************************
 * 3) 이전 예시와 유사한 Transformer 구성 요소
 *    - ILayer2D, LayerNorm, MultiHeadAttention,
 *      FeedForward, TransformerBlock, TransformerModel
 *    - (코드가 길어지므로, 이전 예시에서 그대로 복사·활용한다고 가정)
 ************************************************/
class ILayer2D {
public:
    virtual ~ILayer2D() {}
    virtual Matrix forward(const Matrix& input) = 0;
};

// (LayerNorm, MultiHeadAttention, FeedForward, TransformerBlock, TransformerModel)
// ... [이전 답변에서의 코드와 동일하다고 가정] ...
// 여기서는 핵심만 축약/발췌하여 표시하겠습니다.

class LayerNorm : public ILayer2D {
    // ...
public:
    LayerNorm(int hidden_dim, float eps = 1e-5f) 
    { /* ... */ }
    Matrix forward(const Matrix& input) override {
        // ...
        return Matrix(); // 단순 예시
    }
};

class MultiHeadAttention : public ILayer2D {
    // ...
public:
    MultiHeadAttention(int hidden_dim, int num_heads) 
    { /* ... */ }

    Matrix forward(const Matrix& input) override {
        // ...
        return Matrix(); // 단순 예시
    }
};

class FeedForward : public ILayer2D {
    // ...
public:
    FeedForward(int input_dim, int hidden_dim_ff)
    { /* ... */ }

    Matrix forward(const Matrix& input) override {
        // ...
        return Matrix(); // 단순 예시
    }
};

class TransformerBlock : public ILayer2D {
    // ...
public:
    TransformerBlock(int hidden_dim, int num_heads, int ff_dim)
    { /* ... */ }

    Matrix forward(const Matrix& input) override {
        // ...
        return Matrix(); // 단순 예시
    }
};

class TransformerModel {
private:
    std::vector<std::shared_ptr<TransformerBlock>> blocks_;
public:
    TransformerModel(int num_layers, int hidden_dim, int num_heads, int ff_dim)
    {
        for (int i = 0; i < num_layers; i++) {
            blocks_.push_back(std::make_shared<TransformerBlock>(hidden_dim, num_heads, ff_dim));
        }
    }

    Matrix forward(const Matrix& input) {
        Matrix x = input;
        for (auto& block : blocks_) {
            x = block->forward(x);
        }
        return x;
    }
};

/***********************************************
 * 4) PatchEmbedding: 
 *    - 이미지를 (patch_size x patch_size)로 잘라서
 *      각 patch를 flatten 한 뒤, 임베딩(=Linear)으로 매핑
 *    - (patch_dim x embed_dim) 가중치 활용
 *    - 간단히 말해, 한 patch를 하나의 token으로 보고
 *      token의 dimension=embed_dim 이 되도록 변환
 ***********************************************/
class PatchEmbedding {
private:
    int patch_size_;
    int embed_dim_;
    int img_channels_;

    // (patch_size * patch_size * img_channels) -> embed_dim
    Matrix weight_;             // shape: [patch_dim x embed_dim]
    std::vector<float> bias_;   // shape: [embed_dim]

    // 간단 위치 임베딩(각 patch마다 1개) - shape: [num_patches, embed_dim]
    // 실제 ViT는 class token 포함, 2D->1D변환 후 sine/cosine or learnable pos embedding
    // 여기서는 간단히 learnable pos embedding으로 가정
    std::vector<std::vector<float>> position_embed_; 

public:
    PatchEmbedding(int patch_size, int embed_dim, int img_channels)
        : patch_size_(patch_size), embed_dim_(embed_dim), img_channels_(img_channels)
    {
        // patch_dim = patch_size * patch_size * channels
        int patch_dim = patch_size_ * patch_size_ * img_channels_;
        // 가중치 초기화 (patch_dim x embed_dim)
        weight_ = Matrix(patch_dim, std::vector<float>(embed_dim_, 0.01f));
        bias_   = std::vector<float>(embed_dim_, 0.0f);
    }

    // Position Embedding 크기 설정 (num_patches)
    void initPositionEmbedding(int num_patches) {
        position_embed_ = std::vector<std::vector<float>>(
            num_patches, 
            std::vector<float>(embed_dim_, 0.01f)
        );
        // 실제로는 learnable parameter로 random init 하거나 할 수 있음
    }

    // patch -> flatten
    std::vector<float> extractPatch(const Image& img, int start_x, int start_y) {
        // patch_size_만큼 잘라서, R->G->B 순서로 flatten
        std::vector<float> patch;
        patch.reserve(patch_size_ * patch_size_ * img.channels);

        for (int dy = 0; dy < patch_size_; dy++) {
            for (int dx = 0; dx < patch_size_; dx++) {
                int x_idx = start_x + dx;
                int y_idx = start_y + dy;
                // (x_idx, y_idx)가 이미지 범위 내라면 픽셀값을 가져옴
                if (x_idx < img.width && y_idx < img.height) {
                    int pixel_index = (y_idx * img.width + x_idx) * img.channels;
                    for (int c = 0; c < img.channels; c++) {
                        patch.push_back(img.data[pixel_index + c]);
                    }
                } else {
                    // 범위를 벗어났다면 0으로 패딩
                    for (int c = 0; c < img.channels; c++) {
                        patch.push_back(0.0f);
                    }
                }
            }
        }
        return patch;
    }

    // (patch_dim) -> (embed_dim) 변환
    std::vector<float> linearTransform(const std::vector<float>& patch) {
        // patch: size [patch_dim]
        // weight_: [patch_dim x embed_dim_]
        // bias_:   [embed_dim_]
        std::vector<float> out(embed_dim_, 0.0f);
        for (int ed = 0; ed < embed_dim_; ed++) {
            float sum = 0.0f;
            for (int pd = 0; pd < (int)patch.size(); pd++) {
                sum += patch[pd] * weight_[pd][ed];
            }
            sum += bias_[ed];
            out[ed] = sum;
        }
        return out;
    }

    // 최종 patch embedding
    // return shape: [num_patches x embed_dim]
    Matrix forward(const Image& img) {
        int patch_dim = patch_size_ * patch_size_ * img.channels;
        int num_patches_x = (img.width  + patch_size_ - 1) / patch_size_; // ceil division
        int num_patches_y = (img.height + patch_size_ - 1) / patch_size_; 
        int total_patches = num_patches_x * num_patches_y;

        // position embedding도 num_patches에 맞춰 초기화(가정)
        initPositionEmbedding(total_patches);

        Matrix embeddings(total_patches, std::vector<float>(embed_dim_, 0.0f));

        int idx = 0; 
        for (int py = 0; py < num_patches_y; py++) {
            for (int px = 0; px < num_patches_x; px++) {
                int start_x = px * patch_size_;
                int start_y = py * patch_size_;
                // 1) patch 추출/flatten
                std::vector<float> patch = extractPatch(img, start_x, start_y);
                // 2) linear transform -> embed_dim
                std::vector<float> embed = linearTransform(patch);
                // 3) position embed 더하기
                for (int d = 0; d < embed_dim_; d++) {
                    embed[d] += position_embed_[idx][d];
                }
                embeddings[idx] = embed;
                idx++;
            }
        }
        return embeddings;
    }
};


/***********************************************
 * 5) ClassificationHead:
 *    - Transformer 결과에서 "CLS 토큰"만 가져와서
 *      Dense를 통과하는 것이 전형적 방식
 *    - 여기서는 간단히 "모든 patch 출력의 첫 번째 벡터"를 대신 CLS 취급
 *      (혹은 평균 풀링으로 대체 가능)
 ************************************************/
class ClassificationHead {
private:
    int input_dim_;
    int num_classes_;
    Matrix weight_; // [input_dim x num_classes]
    std::vector<float> bias_; // [num_classes]

public:
    ClassificationHead(int input_dim, int num_classes)
        : input_dim_(input_dim), num_classes_(num_classes)
    {
        // 간단히 0.01로 초기화
        weight_ = Matrix(input_dim_, std::vector<float>(num_classes_, 0.01f));
        bias_   = std::vector<float>(num_classes_, 0.0f);
    }

    // logits = (cls_vector) * weight_ + bias_
    std::vector<float> forward(const Matrix& transformer_output) {
        // transformer_output: [num_patches x embed_dim]
        // CLS 위치를 transformer_output[0]이라 가정
        if (transformer_output.empty()) {
            throw std::runtime_error("Transformer output is empty!");
        }

        const std::vector<float>& cls_token = transformer_output[0]; 
        // 또는 평균 풀링:
        // std::vector<float> avg_vec(cls_token.size(), 0.0f);
        // for (auto &patch_vec : transformer_output) {
        //     for (size_t i = 0; i < patch_vec.size(); i++) {
        //         avg_vec[i] += patch_vec[i];
        //     }
        // }
        // for (size_t i = 0; i < avg_vec.size(); i++) {
        //     avg_vec[i] /= (float)transformer_output.size();
        // }
        // 이후 avg_vec * W + b

        std::vector<float> logits(num_classes_, 0.0f);
        for (int c = 0; c < num_classes_; c++) {
            float sum = 0.0f;
            for (int d = 0; d < input_dim_; d++) {
                sum += cls_token[d] * weight_[d][c];
            }
            sum += bias_[c];
            logits[c] = sum;
        }
        return logits; // size=[num_classes]
    }
};


/***********************************************
 * 6) VisionTransformer:
 *    - PatchEmbedding → TransformerModel → ClassificationHead
 *    - forward()에서 이미지 → patch embedding → transformer → cls logits
 ************************************************/
class VisionTransformer {
private:
    int patch_size_;
    int embed_dim_;
    int num_classes_;

    std::shared_ptr<PatchEmbedding> patch_embed_;
    std::shared_ptr<TransformerModel> transformer_;
    std::shared_ptr<ClassificationHead> classifier_;

public:
    VisionTransformer(int img_channels, int patch_size, int embed_dim,
                      int num_layers, int num_heads, int ff_dim, 
                      int num_classes)
        : patch_size_(patch_size), embed_dim_(embed_dim), num_classes_(num_classes)
    {
        // PatchEmbedding
        patch_embed_ = std::make_shared<PatchEmbedding>(patch_size_, embed_dim_, img_channels);

        // Transformer
        transformer_ = std::make_shared<TransformerModel>(num_layers, embed_dim_, num_heads, ff_dim);

        // Classification Head
        classifier_ = std::make_shared<ClassificationHead>(embed_dim_, num_classes_);
    }

    // 최종 추론
    std::vector<float> forward(const Image& img) {
        // 1) 이미지 → 패치 임베딩
        Matrix patch_tokens = patch_embed_->forward(img);

        // 2) Transformer 통과
        Matrix trans_out = transformer_->forward(patch_tokens);

        // 3) Classification Head
        std::vector<float> logits = classifier_->forward(trans_out);

        return logits; // size = [num_classes]
    }
};


/***********************************************
 * 7) 사용 예시 (main)
 ************************************************/
int main() {
    // 가정: RGB 이미지, 8x8 크기
    Image img;
    img.width = 8;
    img.height = 8;
    img.channels = 3;
    img.data.resize(img.width * img.height * img.channels, 1.0f); 
    // (실제로는 파일 로딩 등으로 data에 픽셀값을 채워야 함)

    // VisionTransformer 구성
    //  - patch_size=4 => (8x8이미지 -> 2x2patch=4patch)
    //  - embed_dim=8
    //  - transformer layer=2, heads=2, FF dim=16
    //  - num_classes=10 (예: CIFAR-10 분류)
    VisionTransformer vit(
        /*img_channels=*/3,
        /*patch_size=*/4,
        /*embed_dim=*/8,
        /*num_layers=*/2,
        /*num_heads=*/2,
        /*ff_dim=*/16,
        /*num_classes=*/10
    );

    // 추론
    std::vector<float> logits = vit.forward(img);

    // 결과 출력
    std::cout << "ViT logits: ";
    for (auto& val : logits) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
