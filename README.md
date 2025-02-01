# MyDL Framework

MyDL is a lightweight framework for implementing inference-only deep learning models (especially Vision Transformer) in C++, similar to PyTorch.  
This framework provides **Tensor** class, **Module** (the base class for all layers/models), fundamental operations (matrix multiplication, ReLU, Softmax, etc.),  
and includes modules such as Linear, MultiHeadAttention, FeedForward, TransformerBlock, TransformerModel, and VisionTransformer.

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
│       └── WeightsLoader.h
├── src/
│   ├── Tensor.cpp
│   ├── Module.cpp
│   ├── Functional.cpp
│   ├── Linear.cpp
│   ├── Transformer.cpp
│   ├── VisionTransformer.cpp
│   ├── WeightsLoader.cpp
│   └── main.cpp
```

- **include/MyDL/**:  
  - `Tensor.h`: Defines the tensor data structure.
  - `Module.h`: Defines the abstract class (interface) that all layers and models inherit from.
  - `Functional.h`: Provides basic operations like matrix multiplication (`matmul`), element-wise addition, ReLU, and Softmax under the `MyDL::F` namespace.
  - `Linear.h`: Implements a fully connected layer.
  - `Transformer.h`: Defines Transformer-related modules such as MultiHeadAttention, FeedForward, TransformerBlock, and TransformerModel.
  - `VisionTransformer.h`: Implements a simple Vision Transformer (ViT) model.  
    (PatchEmbedding is omitted, and the input tensor is assumed to be in the shape `[seq_len, embed_dim]`.)
  - `WeightsLoader.h`: Provides a function to load pre-trained model parameters from a file.

- **src/**:  
  - `Tensor.cpp`: Implements `Tensor` class functions.
  - `Module.cpp`: Implements `Module` class methods.
  - `Functional.cpp`: Implements functions such as `matmul`, `add`, `relu`, and `softmax`.
  - `Linear.cpp`: Implements the `Linear` layer operations.
  - `Transformer.cpp`: Implements Transformer-related modules.
  - `VisionTransformer.cpp`: Implements the `VisionTransformer` model.
  - `WeightsLoader.cpp`: Implements functions for loading pre-trained weights.
  - `main.cpp`: Demonstrates how to use the framework by creating a VisionTransformer model, loading weights, and performing inference.

## Key Features and Usage

### 1. Header Guards and Namespaces
- Each header file applies **header guards** using `#ifndef`, `#define` statements.
- All code (classes, functions) resides under the `MyDL` namespace, with fundamental functions grouped under `MyDL::F`.
- Example: The function `MyDL::F::matmul` performs matrix multiplication on two 2D tensors.

### 2. Tensor Class
- The `Tensor` class manages **data** (a 1D float array) and **shape** (a `std::vector<int>`).
- The constructor accepts a shape parameter and allocates the required buffer size.

### 3. Module (Abstract Class)
- All layers/models inherit from `MyDL::Module` and must implement `Tensor forward(const Tensor& x)`.
- This structure mimics PyTorch's `nn.Module`.

### 4. Functional Module
- The `MyDL::F` namespace includes functions for:
  - `matmul`: Matrix multiplication for 2D tensors.
  - `add`: Element-wise addition.
  - `relu`: ReLU activation function.
  - `softmax`: Row-wise Softmax for 2D tensors.

### 5. Linear Layer
- `MyDL::Linear` performs the linear transformation `y = xW + b`.
  - `setWeights` and `setBias` allow setting pre-trained parameters.

### 6. Transformer Modules
- **MultiHeadAttention**: Implements self-attention using a single head (simplified).
- **FeedForward**: Implements a fully connected feedforward network (Linear → ReLU → Linear).
- **TransformerBlock**: Connects MultiHeadAttention and FeedForward with residual connections.
- **TransformerModel**: Stacks multiple TransformerBlocks sequentially.

### 7. VisionTransformer
- `MyDL::VisionTransformer` combines TransformerModel with a classification head.
- The input tensor is `[seq_len, embed_dim]`, and the first token (CLS token) is passed to the classifier to output class scores (logits).

### 8. Weights Loader
- `MyDL::loadWeights` loads pre-trained model parameters from a file.
- Currently, it only prints the file path as a placeholder; actual implementation should parse the file and set layer parameters.

## Build and Run Instructions

### Build with CMake

```bash
mkdir build && cd build
cmake ..
make
```

### Run
```bash
./MyDLApp
```

## Example Usage (main.cpp)
- `main.cpp` demonstrates:
  1. Creating a VisionTransformer model.
  2. Loading pre-trained weights (e.g., "vit_weights.bin").
  3. Generating a dummy input tensor (e.g., 4 patches, each of length 16).
  4. Running inference using the `forward` function and printing the output logits.

## Future Improvements and Optimizations
- **Patch Embedding**: Implement patch extraction from images and apply linear transformation to obtain embeddings.
- **LayerNorm, Dropout**: Add normalization and dropout techniques for better generalization.
- **Mobile Optimization**: Implement optimizations using ARM NEON, Vulkan, or NPU/DSP for efficient inference on mobile devices.
- **Additional Models**: Extend the framework to include models for image segmentation (e.g., adding a decoder with upsampling layers).

## Conclusion
This framework provides a simple C++ implementation for inference-only models.  
Similar to PyTorch, it allows defining models by inheriting from `Module`, loading pre-trained weights, and calling `forward` for inference.  
It is extendable for mobile inference, and additional optimizations or model architectures can be integrated.

