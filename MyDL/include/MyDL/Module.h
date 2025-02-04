#ifndef MYDL_MODULE_H
#define MYDL_MODULE_H

#include "Tensor.h"
#include <vector>
#include <iostream>
#include <memory>
#include <map>
#include <fstream>

namespace MyDL {

/**
 * Abstract Base Class for Neural Network Modules
 *
 * The `Module` class is the foundation for all layers and models in MyDL,
 * similar to `torch.nn.Module` in PyTorch.
 *
 * Responsibilities:
 * - Defines the `forward()` method (must be implemented by subclasses).
 * - Manages parameters (weights, biases) automatically.
 * - Supports submodule registration (for hierarchical models).
 * - Provides utility functions like `printModuleInfo()`, `parameters()`, `save_weights()`, `load_weights()`.
 *
 * Key Features:
 * - **Submodules & Hierarchical Structure**: Models like Transformer can register submodules.
 * - **Parameter Management**: Automatically tracks learnable tensors.
 * - **Serialization**: Can save/load weights for inference.
 */
class Module {
private:
    std::map<std::string, std::shared_ptr<Tensor>> parameters_; // Stores named parameters
    std::map<std::string, std::shared_ptr<Module>> submodules_; // Stores submodules

public:
    /**
     * Virtual destructor to ensure proper cleanup of derived classes.
     */
    virtual ~Module() {}

    /**
     * Pure virtual function that must be implemented by all derived layers/models.
     * @param x Input tensor.
     * @return Output tensor after applying the layer's computation.
     */
    virtual Tensor forward(const Tensor& x) = 0;

    /**
     * Registers a parameter (trainable tensor) in the module.
     * @param name Name of the parameter.
     * @param tensor Shared pointer to a Tensor object.
     */
    void register_parameter(const std::string& name, std::shared_ptr<Tensor> tensor) {
        parameters_[name] = tensor;
    }

    /**
     * Retrieves all registered parameters (weights and biases).
     * @return Vector of shared pointers to parameter tensors.
     */
    std::vector<std::shared_ptr<Tensor>> parameters() {
        std::vector<std::shared_ptr<Tensor>> params;
        for (auto& p : parameters_) {
            params.push_back(p.second);
        }
        return params;
    }

    /**
     * Registers a submodule (e.g., Linear, TransformerBlock) inside a larger model.
     * @param name Name of the submodule.
     * @param module Shared pointer to the submodule.
     */
    void register_module(const std::string& name, std::shared_ptr<Module> module) {
        submodules_[name] = module;
    }

    /**
     * Prints the module hierarchy and registered parameters.
     */
    virtual void printModuleInfo() const {
        std::cout << "Module Info:\n";
        for (const auto& param : parameters_) {
            std::cout << "  Parameter: " << param.first << " (size: " << param.second->size() << ")\n";
        }
        for (const auto& submod : submodules_) {
            std::cout << "  Submodule: " << submod.first << "\n";
            submod.second->printModuleInfo(); // Recursively print submodule info
        }
    }

    /**
     * Saves the model's parameters to a binary file.
     * @param filename Name of the file to save to.
     */
    void save_weights(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open file for saving weights.");
        }
        for (auto& param : parameters_) {
            file.write(reinterpret_cast<char*>(param.second->data.data()), param.second->size() * sizeof(float));
        }
        file.close();
    }

    /**
     * Loads model parameters from a binary file.
     * @param filename Name of the file to load from.
     */
    void load_weights(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open file for loading weights.");
        }
        for (auto& param : parameters_) {
            file.read(reinterpret_cast<char*>(param.second->data.data()), param.second->size() * sizeof(float));
        }
        file.close();
    }
};

} // namespace MyDL

#endif // MYDL_MODULE_H
