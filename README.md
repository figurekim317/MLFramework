# MyDL Framework

MyDL is a lightweight C++ framework for inference-only deep learning models.  
Inspired by PyTorch’s modular design, MyDL provides a **Tensor** class, a **Module** base class (for layers/models), fundamental operations (`matmul`, `add`, `relu`, `softmax`, etc.), and modules for Transformers (VisionTransformer) and CNN-based architectures (such as ResNet, YOLO).  

This document outlines:

1. The overall framework structure  
2. How to extend MyDL for various models (ResNet, YOLO, ViT, etc.)  
3. How to load weights and run inference  
4. Future plans, including mobile optimization  

---

## Table of Contents

- [Features at a Glance](#features-at-a-glance)
- [Directory Structure](#directory-structure)
- [Key Classes and Modules](#key-classes-and-modules)
  - [1. Tensor Class](#1-tensor-class)
  - [2. Module (Abstract Class)](#2-module-abstract-class)
  - [3. Functional Operations](#3-functional-operations)
  - [4. Layers: Linear, Conv2d, BatchNorm2d, etc.](#4-layers-linear-conv2d-batchnorm2d-etc)
  - [5. Transformer and VisionTransformer](#5-transformer-and-visiontransformer)
  - [6. ResNet (Example CNN Model)](#6-resnet-example-cnn-model)
  - [7. YOLO (Example Object Detection Model)](#7-yolo-example-object-detection-model)
  - [8. Weights Loader](#8-weights-loader)
- [Build and Run Instructions](#build-and-run-instructions)
- [Example Usage](#example-usage)
- [Mobile Optimization](#mobile-optimization)
- [Future Improvements](#future-improvements)
- [Conclusion](#conclusion)

---

## Directory Structure

```
MyDL/
├── include/
│   └── MyDL/
│       ├── Tensor.h
│       ├── Module.h
│       ├── Functional.h
│       ├── Linear.h
│       ├── Transformer.h
│       ├── VisionTransformer.h
│       ├── ResNet.h          
│       ├── Yolo.h               
│       ├── Conv2d.h             
│       ├── BatchNorm2d.h        
│       └── WeightsLoader.h
├── src/
│   ├── Tensor.cpp
│   ├── Module.cpp
│   ├── Functional.cpp
│   ├── Linear.cpp
│   ├── Transformer.cpp
│   ├── VisionTransformer.cpp
│   ├── ResNet.cpp               
│   ├── Yolo.cpp                 
│   ├── Conv2d.cpp               
│   ├── BatchNorm2d.cpp          
│   ├── WeightsLoader.cpp
│   └── main.cpp

```

- **include/MyDL/**:  
  - `Tensor.h`: Defines the `Tensor` data structure and shape management.  
  - `Module.h`: Abstract base class for all layers/models (implements `forward` interface).  
  - `Functional.h`: Basic operations like `matmul`, `add`, `relu`, `softmax`, etc.  
  - `Linear.h`: Fully connected (dense) layer.  
  - `Conv2d.h`: 2D convolution layer.  
  - `BatchNorm2d.h`: 2D batch normalization layer.  
  - `Transformer.h`: Modules for multi-head attention, feed-forward network, Transformer block/model.  
  - `VisionTransformer.h`: A simplified Vision Transformer (ViT) implementation (input assumed as `[seq_len, embed_dim]`).  
  - `ResNet.h`: Example ResNet model definition.  
  - `Yolo.h`: Example YOLO model definition.  
  - `WeightsLoader.h`: Loading of pre-trained weights from a file.

- **src/**:  
  - Each corresponding `.cpp` file implements the declarations in the `.h` files.  
  - `main.cpp` can serve as a simple demo that creates a model, loads weights, and performs inference.

---

## Key Classes and Modules

### 1. Tensor Class

- Manages a **1D buffer** of floats and a **shape** (`std::vector<int>`).  
- Typical usage involves creating a Tensor with a shape (e.g., `[batch_size, channels, height, width]` for images) and populating its data buffer.  
- Provides helper methods for indexing, reshaping, or slicing if needed.

### 2. Module (Abstract Class)

- All layers and models derive from `MyDL::Module`.  
- Must implement:
  ```cpp
  Tensor forward(const Tensor& x) override;
  ```
- May hold internal parameters such as weights, biases, or submodules (e.g., a ResNet might contain multiple convolutional blocks).

### 3. Functional Operations

Functional operations are collected under the namespace `MyDL::F`. These include:

- `matmul(const Tensor& A, const Tensor& B)`: Matrix multiplication for 2D tensors.
- `add(const Tensor& A, const Tensor& B)`: Element-wise addition.
- `relu(const Tensor& x)`: Element-wise ReLU activation.
- `softmax(const Tensor& x)`: Row-wise or channel-wise softmax.

These fundamental functions can be optimized later using SIMD (NEON, AVX), multithreading (OpenMP), or GPU acceleration (Vulkan, CUDA, Metal).

---

### 4. Layers: Linear, Conv2d, BatchNorm2d, etc.

#### Linear
- Implements a fully connected layer:  
  \[
  \text{out} = \text{in} \times W + b
  \]
- `setWeights` and `setBias` methods allow loading or initializing parameters.

#### Conv2d (Example)
- Implements a **2D convolution** layer with:
  - Kernel size, padding, stride, dilation.
  - `weight` (4D tensor) and `bias` (1D tensor).
- Can be further optimized using **im2col** and matrix multiplication.

#### BatchNorm2d (Example)
- Applies **batch normalization** on 2D feature maps.
- Has parameters:
  - `gamma` and `beta` for scaling and shifting.
  - Running mean and variance for normalization.

---

### 5. Transformer and VisionTransformer

#### MultiHeadAttention
- Currently simplified to a **single head**, but extendable to multiple heads.
- Takes **queries, keys, and values** to compute attention-weighted outputs.

#### FeedForward
- Typically a **two-layer MLP** with an activation (ReLU) in between.

#### TransformerBlock
- Combines **MultiHeadAttention** and **FeedForward** with:
  - Residual connections.
  - (Optional) Layer normalization.

#### TransformerModel
- Stacks multiple **TransformerBlock** modules sequentially.

#### VisionTransformer
- Adds a **classification head** on top of `TransformerModel`.
- Simplified by assuming the input is **already patch-embedded** (shape `[seq_len, embed_dim]`).
- The first token (**CLS token**) is used for classification output.

---

### 6. ResNet (Example CNN Model)

ResNet typically consists of:

1. **Initial Convolution Layer**  
   - Convolution + BatchNorm + ReLU + Pooling.
2. **Residual Blocks**  
   - Each block contains 2–3 convolutions, BatchNorm, and ReLU.
3. **Global Average Pooling and Fully Connected Layer**  
   - Reduces feature maps to a single vector for classification.

**Implementation in MyDL:**  
- A `class ResNet : public Module` can be defined, which:
  - Creates residual blocks internally.
  - Connects them sequentially in the `forward` function.

---

### 7. YOLO (Example Object Detection Model)

YOLO (**You Only Look Once**) consists of:

1. **Backbone**  
   - Darknet, CSP, or other CNN-based feature extractor.
   - Repeated convolutional + downsampling layers.
2. **Detection Head**  
   - Outputs:
     - Bounding box coordinates.
     - Class probabilities.
     - Objectness score.

**Implementation in MyDL:**  
- Define convolutional blocks following YOLO’s structure:
  - `Conv2d`, `BatchNorm2d`, `LeakyReLU`.
- Implement a **detection layer** that:
  - Processes the feature map into bounding box predictions.
- Handle **post-processing** externally:
  - **Non-Maximum Suppression (NMS)**.
  - **Anchor box adjustments**.

---

## 8. Weights Loader

### MyDL::loadWeights

```cpp
MyDL::loadWeights(const std::string& filePath, Module& model);
```

- Reads parameters from a file (e.g., `.bin`, `.txt`, `.onnx`) and sets them in the model’s layers.
- The exact format is up to the implementation. For a more advanced solution, you might parse ONNX or a custom binary format.

---

## Build and Run Instructions

### 1. CMake Build

```bash
mkdir build && cd build
cmake ..
make
```

This generates an executable (e.g., `MyDLApp`).

### 2. Run

```bash
./MyDLApp
```

---

## Example Usage

A simplified `main.cpp` might demonstrate:

1. **Creating a model** (e.g., VisionTransformer, ResNet, YOLO).  
2. **Loading pre-trained weights** from a file (`"model_weights.bin"`).  
3. **Preparing a dummy or real input** (e.g., an image tensor).  
4. **Running inference** (`forward`) and **printing or processing the output**.  

```cpp
#include "MyDL/Module.h"
#include "MyDL/VisionTransformer.h"
#include "MyDL/WeightsLoader.h"
#include <iostream>

int main() {
    // 1. Create a VisionTransformer (ViT) instance
    MyDL::VisionTransformer vit( /* embed_dim, num_heads, etc. */ );

    // 2. Load pre-trained weights
    MyDL::loadWeights("vit_weights.bin", vit);

    // 3. Prepare input tensor (example: sequence length 4, embedding dim 16)
    MyDL::Tensor input({4, 16});
    // ... Fill input data with actual values

    // 4. Run inference
    MyDL::Tensor output = vit.forward(input);

    // 5. Print results
    std::cout << "Output size: ";
    for (auto dim : output.shape()) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

---

## Mobile Optimization

1. **ARM NEON**  
   - Use NEON intrinsics for matrix multiplication, convolution loops, etc.  
2. **OpenMP or Thread Pooling**  
   - Distribute operations across multiple CPU cores to boost performance.  
3. **Vulkan / Metal**  
   - Consider writing GPU kernels for convolutions, matrix multiplication, etc., on mobile.  
4. **NPU / DSP Acceleration**  
   - If hardware supports specialized ML accelerators, integrate the driver or vendor library.  

These optimizations usually come after verifying correctness in a single-threaded CPU implementation.

---

## Future Improvements

1. **LayerNorm, Dropout** for Transformer-based models.  
2. **Patch Embedding** for VisionTransformer to handle raw images directly.  
3. **ONNX or Other Format Parsing** to load weights and model structure automatically.  
4. **Additional CNN Layers** (e.g., pooling, advanced blocks for ResNet/YOLO).  
5. **Advanced Post-Processing** for YOLO (NMS, multi-scale testing).  
6. **More Model Architectures** (e.g., MobileNet, UNet, etc.).  

---

## Conclusion

MyDL offers a simple yet extensible framework for running deep learning inference in C++. By organizing layers as `Module` subclasses, providing a flexible `Tensor` class, and separating basic operations into a `Functional` namespace, it mimics the ease of use of frameworks like PyTorch.

Whether you want to run ResNet, YOLO, or Vision Transformer on desktop or mobile, you can build upon MyDL’s modular structure. As you extend it with new layers, weight loaders, and mobile optimizations (NEON, GPU), MyDL can evolve into a robust solution for real-time on-device inference.

Feel free to explore, add new modules, and optimize for specific hardware to make MyDL truly powerful and suitable for production-level applications.

