#include "MyDL/Module.h"
#include <iostream>
#include <fstream>

namespace MyDL {

/**
 * Registers a parameter (trainable tensor) in the module.
 * @param name Name of the parameter.
 * @param tensor Shared pointer to a Tensor object.
 */
void Module::register_parameter(const std::string& name, std::shared_ptr<Tensor> tensor) {
    parameters_[name] = tensor;
}

/**
 * Retrieves all registered parameters (weights and biases).
 * @return Vector of shared pointers to parameter tensors.
 */
std::vector<std::shared_ptr<Tensor>> Module::parameters() {
    std::vector<std::shared_ptr<Tensor>> params;
    for (auto& p : parameters_) {
        params.push_back(p.second);
    }
    return params;
}

/**
 * Registers a submodule inside a larger model.
 * @param name Name of the submodule.
 * @param module Shared pointer to the submodule.
 */
void Module::register_module(const std::string& name, std::shared_ptr<Module> module) {
    submodules_[name] = module;
}

/**
 * Prints the module hierarchy and registered parameters.
 */
void Module::printModuleInfo() const {
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
void Module::save_weights(const std::string& filename) {
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
void Module::load_weights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for loading weights.");
    }
    for (auto& param : parameters_) {
        file.read(reinterpret_cast<char*>(param.second->data.data()), param.second->size() * sizeof(float));
    }
    file.close();
}

} // namespace MyDL
