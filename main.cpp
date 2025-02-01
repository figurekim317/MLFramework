#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cstdlib> // For std::atoi
#include "MyDL/Tensor.h"
#include "MyDL/VisionTransformer.h"
#include "MyDL/WeightsLoader.h"

void printUsage() {
    std::cout << "Usage: ./MyDLApp --model vit --seq_len <int> --embed_dim <int> --weights <path>\n";
    std::cout << "Options:\n";
    std::cout << "  --model [vit]          : Specify model type (currently only 'vit' is supported)\n";
    std::cout << "  --seq_len <int>        : Sequence length (number of patches)\n";
    std::cout << "  --embed_dim <int>      : Embedding dimension\n";
    std::cout << "  --weights <path>       : Path to pre-trained weights file\n";
    std::cout << "  --help                 : Show this help message\n";
}

int main(int argc, char* argv[]) {
    // Default values
    std::string modelType = "vit";
    int seqLen = 4;
    int embedDim = 16;
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
    if (modelType != "vit") {
        std::cerr << "Error: Unsupported model type. Currently, only 'vit' is supported.\n";
        return 1;
    }

    // Display chosen configuration
    std::cout << "Running MyDL Script with the following configuration:\n";
    std::cout << "  Model Type    : " << modelType << "\n";
    std::cout << "  Sequence Length: " << seqLen << "\n";
    std::cout << "  Embedding Dim : " << embedDim << "\n";
    std::cout << "  Weights Path  : " << (weightsPath.empty() ? "None (random init)" : weightsPath) << "\n";

    // Create the model
    std::shared_ptr<MyDL::Module> model = std::make_shared<MyDL::VisionTransformer>(
        2, embedDim, 2, 32, 10
    );

    // Load weights if provided
    if (!weightsPath.empty()) {
        MyDL::loadWeights(model, weightsPath);
    }

    // Generate dummy input tensor
    MyDL::Tensor input({seqLen, embedDim});
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
