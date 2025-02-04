#include "MyDL/WeightsLoader.h"
#include <fstream>
#include <stdexcept>

namespace MyDL {

/**
 * Loads model weights from a binary file and assigns them to the corresponding parameters.
 * This function ensures that all model parameters are correctly mapped to the file's data.
 *
 * @param model Shared pointer to the model (inheriting from `Module`).
 * @param path Path to the weight file (binary format).
 * @return True if loading succeeds, False otherwise.
 */
bool WeightsLoader::loadWeights(std::shared_ptr<Module> model, const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "[WeightsLoader] Error: Could not open file: " << path << std::endl;
        return false;
    }

    std::vector<std::shared_ptr<Tensor>> params = model->parameters();
    if (params.empty()) {
        std::cerr << "[WeightsLoader] Warning: Model has no parameters to load." << std::endl;
        return false;
    }

    for (auto& param : params) {
        size_t tensorSize = param->size() * sizeof(float);
        if (!file.read(reinterpret_cast<char*>(param->data.data()), tensorSize)) {
            std::cerr << "[WeightsLoader] Error: Failed to read sufficient data from file." << std::endl;
            return false;
        }
    }

    file.close();
    std::cout << "[WeightsLoader] Successfully loaded weights from: " << path << std::endl;
    return true;
}

} // namespace MyDL
