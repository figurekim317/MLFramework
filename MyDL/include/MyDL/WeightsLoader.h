#ifndef MYDL_WEIGHTSLOADER_H
#define MYDL_WEIGHTSLOADER_H

#include <string>
#include <memory>
#include <iostream>
#include <fstream>
#include "Module.h"

namespace MyDL {

/**
 * WeightsLoader: Utility for loading model weights from binary files
 *
 * This module provides functionality to **load pre-trained model weights** 
 * from a file and apply them to the corresponding layers in a `Module`-based model.
 *
 * - Supports **binary weight files** to efficiently store and load parameters.
 * - Iterates through **registered parameters** in a module and assigns loaded values.
 * - Works with **any model inheriting from `Module`** (e.g., `Linear`, `Conv2d`, etc.).
 *
 * **Usage Example:**
 * ```cpp
 * std::shared_ptr<MyDL::Module> model = std::make_shared<MyDL::TransformerModel>(...);
 * MyDL::loadWeights(model, "model_weights.bin");
 * ```
 */
class WeightsLoader {
public:
    /**
     * Loads model weights from a binary file and assigns them to the corresponding parameters.
     * @param model Shared pointer to the model (inheriting from `Module`).
     * @param path Path to the weight file (binary format).
     * @return True if loading succeeds, False otherwise.
     */
    static bool loadWeights(std::shared_ptr<Module> model, const std::string& path);
};

} // namespace MyDL

#endif // MYDL_WEIGHTSLOADER_H
