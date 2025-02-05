#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cstdlib> // For std::atoi
#include "MyDL/Tensor.h"
#include "MyDL/VisionTransformer.h"
#include "MyDL/Transformer.h"
#include "MyDL/ResNet.h"
#include "MyDL/WeightsLoader.h"

void printUsage() {
    std::cout << "Usage: ./MyDLApp --model <vit|resnet|transformer> [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --model [vit|resnet|transformer] : Specify model type\n";
    std::cout << "  --seq_len <int>                  : Sequence length (for ViT, Transformer)\n";
    std::cout << "  --embed_dim <int>                : Embedding dimension (for ViT, Transformer)\n";
    std::cout << "  --num_classes <int>              : Number of output classes (default: 10)\n";
    std::cout << "  --weights <path>                 : Path to pre-trained weights file\n";
    std::cout << "  --help                           : Show this help message\n";
}

int main(int argc, char* argv[]) {
    // Default values
    std::string modelType = "";
    int seqLen = 4;       // For ViT, Transformer
    int embedDim = 16;    // For ViT, Transformer
    int numClasses = 10;  // Number of output classes
    std::string weightsPath = "";

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--model" && i + 1 < argc) {
            modelType = argv[++i];
        } else if (arg == "--seq_len" && i + 1 < argc) {
            seqLen = std::atoi(argv[++i]);
        } else if (arg == "--embed_dim" && i + 1 < argc) {
            embedDim = std::atoi(argv[++i]);
        } else if (arg == "--num_classes" && i + 1 < argc) {
            numClasses = std::atoi(argv[++i]);
        } else if (arg == "--weights" && i + 1 < argc) {
            weightsPath = argv[++i];
        } else if (arg == "--help") {
            printUsage();
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            printUsage();
            return 1;
        }
    }

    // Validate model selection
    if (modelType != "vit" && modelType != "resnet" && modelType != "transformer") {
        std::cerr << "Error: Unsupported model type. Choose from 'vit', 'resnet', or 'transformer'.\n";
        return 1;
    }

    // Display chosen configuration
    std::cout << "Running MyDL Script with the following configuration:\n";
    std::cout << "  Model Type    : " << modelType << "\n";
    if (modelType == "vit" || modelType == "transformer") {
        std::cout << "  Sequence Length: " << seqLen << "\n";
        std::cout << "  Embedding Dim : " << embedDim << "\n";
    }
    std::cout << "  Num Classes   : " << numClasses << "\n";
    std::cout << "  Weights Path  : " << (weightsPath.empty() ? "None (random init)" : weightsPath) << "\n";

    // Create the model
    std::shared_ptr<MyDL::Module> model;

    if (modelType == "vit") {
        model = std::make_shared<MyDL::VisionTransformer>(2, embedDim, 2, 32, numClasses, seqLen);
    } else if (modelType == "transformer") {
        model = std::make_shared<MyDL::TransformerModel>(2, embedDim, 2, 32, seqLen);
    } else if (modelType == "resnet") {
        model = std::make_shared<MyDL::ResNet>(std::vector<int>{2, 2, 2, 2}, numClasses);  // ResNet-18 style
    }

    // Load weights if provided
    if (!weightsPath.empty()) {
        MyDL::WeightsLoader::loadWeights(model, weightsPath);
    }

    // Generate dummy input tensor
    MyDL::Tensor input;
    if (modelType == "vit" || modelType == "transformer") {
        input = MyDL::Tensor({seqLen, embedDim});  // For ViT & Transformer
    } else if (modelType == "resnet") {
        input = MyDL::Tensor({1, 3, 224, 224});  // For ResNet (batch size 1, RGB image 224x224)
    }

    for (size_t i = 0; i < input.size(); i++) {
        input.data[i] = 0.01f * (i + 1);
    }

    // Run inference
    MyDL::Tensor logits = model->forward(input);

    // Print output logits
    std::cout << "Inference Output (Logits): ";
    for (auto val : logits.data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
