LLM Training Framework in Metal:

Papers referenced:
FlashAttention-2: https://arxiv.org/pdf/2312.11918
FlashAttention-3: https://arxiv.org/abs/2407.08608
Shader Documentation: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf?utm_source=chatgpt.com

Kernels in this repo:
Embedding 
Elementwise Operation (used for residual adding)
Linear layer (Out = X @ A + B)
Matmul (no bias and bias - not used but in here as the base reference for how matmul2d_op works)
Relu
Softmax
Multi-layer perceptron/feedforward. Sibling dW and dA operations are unfused for threadgroup occupancy
Attention - Includes computing qkv, flashattention implementation, and backwards loop split into computeD, flashattnbackward, computeGrad (dX and dW) 
Optimizer step
Layernorm

Performance on my M4 Macbook Air: 
on these dimensions: static const float LR = 0.003f;
static const int BLOCK_SIZE = 256;
static const int N_HEADS = 6;
static const int HEAD_SIZE = 64;
static const int N_LAYERS = 6;
static const int MLP_SCALE = 2;
Iter speed ~= 23ms/iter

