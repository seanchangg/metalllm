#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <iostream>
#include <random>
#include <algorithm>
#include <Accelerate/Accelerate.h>
#include <chrono>
#include <string>
#include <fstream>
#include <numeric>
#include "tokenprocessor.h"

std::random_device rd{};
std::mt19937 generator{2};
std::ifstream file("drake_lyrics.txt");
std::string TEXT((std::istreambuf_iterator<char>(file)),
                    std::istreambuf_iterator<char>());
std::unordered_map<char, uint32_t> encoder;
std::unordered_map<uint32_t, char> decoder;

static NS::String* makeNSString(const char* s) {
    return NS::String::string(s, NS::UTF8StringEncoding);
}

template <typename T>
struct Matrix {
    int rows;
    int cols;
    std::vector<T> matrix;
    Matrix(int r, int c) : rows(r), cols(c), matrix(r*c) {};
    T& operator() (int r_idx, int c_idx) {return matrix[r_idx*cols + c_idx];}
    const T& operator() (int r_idx, int c_idx) const {return matrix[r_idx * cols + c_idx];}

    void fillNormal(float stdev = 1.0f) {
        std::normal_distribution<float> dist(0.0f, stdev);
        for (T& v : matrix) {
            v = (T)dist(generator);
        }
    }
    void fillXavier() {
        fillNormal(std::sqrt(2.0f / (float)(rows + cols)));
    }
    void fillKaiming() {
        fillNormal(std::sqrt(2.0f / (float)rows));
    }
    void fillUniform(float lo, float hi) {
        std::uniform_real_distribution<float> dist(lo, hi);
        for (T& v : matrix) {
            v = (T)dist(generator);
        }
    }
    void fillZeros() {
        std::fill(matrix.begin(), matrix.end(), (T)0.0f);
    }
    void fillOnes() {
        std::fill(matrix.begin(), matrix.end(), (T)1.0f);
    }
};

struct MatMulParams {
    std::uint32_t M; //output rows
    std::uint32_t N; //output cols
    std::uint32_t K;
    MatMulParams(int r, int c, int k) : M(r), N(c), K(k) {};
};

struct ReluInputParams {
    uint32_t count;
};

struct SoftMaxParams {
    uint32_t rows;
    uint32_t cols;
    uint32_t group_size;
};

struct LossParams {
    uint32_t rows;
    uint32_t cols;
    uint32_t group_size;
};

struct LayernormParams {
    uint32_t rows;
    uint32_t cols;
    uint32_t group_size;
    float eps;
    LayernormParams(int r, int c, float e = 1e-5f)
        : rows(r), cols(c), group_size(std::min(c, 1024)), eps(e) {}
};

struct StepParams {
    const uint32_t paramCount;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    float lr_t = 0; //= lr * sqrt(1-beta2^t) / (1-beta1^t)
    const float one_minus_beta1 = 0.1f;; //1-beta1
    const float one_minus_beta2 = 0.001f; //1-beta2
    const float eps = 1e-8;
};

struct EmbedParams {
    uint32_t t;
    uint32_t e; //embedding dimension
    uint32_t group_size;
};

struct AttentionParams {
    uint M; //input matrix dimensions
    uint K; //n_embed
    const uint D; //Head_size
    uint NH; //n-heads
};

struct mlpParams {
    uint M;
    uint N;
    uint S;
};

static const float LR = 0.003f;
static const int BLOCK_SIZE = 256;
static const int N_EMBED = 384;
//HEAD_SIZE must be 64: attention.metal hardcodes it (HEAD_SIZE constant, the
//64-wide matmul descriptors, and the dP-buffer reuse in flashAttentionBackward
//all assume it). other values silently compute wrong attention.
static const int N_HEADS = 6;
static const int HEAD_SIZE = 64;
static const int N_LAYERS = 6;
static const int MLP_SCALE = 2;

template <typename T>
void print(Matrix<T>& in, int n) {
    for (int i =0; i<n; ++i) {
        std::cout << static_cast<float>(in.matrix[i]) << " ";
    }
}

template <typename In, typename Out> struct MPPMatmulKernel;
template <> struct MPPMatmulKernel<float, float> {static constexpr const char* name = "MPPMatMul";};
template <> struct MPPMatmulKernel<__bf16, float> {static constexpr const char* name = "MPPMatMulBfloat";};
template <> struct MPPMatmulKernel<__bf16, __bf16> {static constexpr const char* name = "MPPMatMulBfloatBfloat";};

template <typename In, typename Out> struct MPPMatmulBiasKernel;
template <> struct MPPMatmulBiasKernel<float, float> {static constexpr const char* name = "MPPMatMulBias";};
template <> struct MPPMatmulBiasKernel<__bf16, __bf16> {static constexpr const char* name = "MPPMatMulBiasBfloat";};

template <typename In, typename Out>
struct MPPMatMul {
    NS::AutoreleasePool* pool;
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::ComputePipelineState* pipelineState;

    MTL::Buffer* ABuffer;
    MTL::Buffer* BBuffer;
    MTL::Buffer* outBuffer;
    MTL::Buffer* paramsBuffer;
    size_t ABytes;
    size_t BBytes;
    size_t outBytes;
    MatMulParams params;
    uint32_t num_groups_x;
    uint32_t num_groups_y;
    uint32_t threadgroup_x =64;
    uint32_t threadgroup_y = 64;
    uint32_t simd_groups = 4;


    MPPMatMul(int A_r, int A_c, int B_r, int B_c) :ABytes(A_r * A_c * sizeof(In)), BBytes(B_r * B_c * sizeof(In)), outBytes(A_r * B_c * sizeof(Out)), params(A_r, B_c, A_c) {
        pool = NS::AutoreleasePool::alloc()->init();
        device = MTL::CreateSystemDefaultDevice();
        commandQueue = device->newCommandQueue();
        MTL::Library* library = device->newDefaultLibrary();
        MTL::Function* function = library->newFunction(makeNSString(MPPMatmulKernel<In,Out>::name));

        NS::Error* error = nullptr;
        pipelineState = device->newComputePipelineState(function, &error);

        num_groups_x = B_c / threadgroup_x;   // tgid.x -> rows (M), per kernel contract
        num_groups_y = A_r / threadgroup_y;   // tgid.y -> cols (N)

        ABuffer = device->newBuffer(ABytes, MTL::ResourceStorageModeShared);
        BBuffer = device->newBuffer(BBytes, MTL::ResourceStorageModeShared);
        outBuffer = device->newBuffer(outBytes, MTL::ResourceStorageModeShared);
        paramsBuffer = device->newBuffer(sizeof(MatMulParams), MTL::ResourceStorageModeShared);

        std::memcpy(paramsBuffer->contents(), &params, sizeof(MatMulParams));


        function->release();
        library->release();
    }
    ~MPPMatMul() {
            paramsBuffer->release();
            outBuffer->release();
            ABuffer->release();
            BBuffer->release();
            pipelineState->release();
            commandQueue->release();
            pool->release();
            device->release();
        }
    void run(const Matrix<In>& a, const Matrix<In>& b, Matrix<Out>& out) {
        std::memcpy(ABuffer->contents(), a.matrix.data(), ABytes);
        std::memcpy(BBuffer->contents(), b.matrix.data(), BBytes);

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

        encoder->setComputePipelineState(pipelineState);
        encoder->setBuffer(ABuffer, 0, 0);
        encoder->setBuffer(BBuffer, 0, 1);
        encoder->setBuffer(outBuffer, 0, 2);
        encoder->setBuffer(paramsBuffer, 0, 3);

        encoder->dispatchThreadgroups(MTL::Size(num_groups_y, num_groups_x, 1), MTL::Size(32*simd_groups, 1, 1));
        encoder->endEncoding();
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
        auto* gpuOut = static_cast<Out*>(outBuffer->contents());
        std::copy(gpuOut, gpuOut + out.matrix.size(), out.matrix.begin());
    }
};

template <typename In, typename Out>
struct MPPMatMulBias {
    NS::AutoreleasePool* pool;
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::ComputePipelineState* pipelineState;

    MTL::Buffer* ABuffer;
    MTL::Buffer* BBuffer;
    MTL::Buffer* biasBuffer;
    MTL::Buffer* outBuffer;
    MTL::Buffer* paramsBuffer;
    size_t ABytes;
    size_t BBytes;
    size_t biasBytes;
    size_t outBytes;
    MatMulParams params;
    uint32_t num_groups_x;
    uint32_t num_groups_y;
    uint32_t threadgroup_x =64;
    uint32_t threadgroup_y = 64;
    uint32_t simd_groups = 4;


    MPPMatMulBias(int A_r, int A_c, int B_r, int B_c) :ABytes(A_r * A_c * sizeof(In)), BBytes(B_r * B_c * sizeof(In)), biasBytes(B_c * sizeof(float)), outBytes(A_r * B_c * sizeof(Out)), params(A_r, B_c, A_c) {
        pool = NS::AutoreleasePool::alloc()->init();
        device = MTL::CreateSystemDefaultDevice();
        commandQueue = device->newCommandQueue();
        MTL::Library* library = device->newDefaultLibrary();
        MTL::Function* function = library->newFunction(makeNSString(MPPMatmulBiasKernel<In,Out>::name));

        NS::Error* error = nullptr;
        pipelineState = device->newComputePipelineState(function, &error);

        num_groups_y = A_r / threadgroup_y;
        num_groups_x = B_c / threadgroup_x;

        ABuffer = device->newBuffer(ABytes, MTL::ResourceStorageModeShared);
        BBuffer = device->newBuffer(BBytes, MTL::ResourceStorageModeShared);
        biasBuffer = device->newBuffer(biasBytes, MTL::ResourceStorageModeShared);
        outBuffer = device->newBuffer(outBytes, MTL::ResourceStorageModeShared);
        paramsBuffer = device->newBuffer(sizeof(MatMulParams), MTL::ResourceStorageModeShared);

        std::memcpy(paramsBuffer->contents(), &params, sizeof(MatMulParams));


        function->release();
        library->release();
    }
    ~MPPMatMulBias() {
            paramsBuffer->release();
            outBuffer->release();
            ABuffer->release();
            BBuffer->release();
            biasBuffer->release();
            pipelineState->release();
            commandQueue->release();
            pool->release();
            device->release();
        }
    void run(const Matrix<In>& a, const Matrix<In>& b, const Matrix<float>& bias, Matrix<Out>& out) {
        std::memcpy(ABuffer->contents(), a.matrix.data(), ABytes);
        std::memcpy(BBuffer->contents(), b.matrix.data(), BBytes);
        std::memcpy(biasBuffer->contents(), bias.matrix.data(), biasBytes);

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

        encoder->setComputePipelineState(pipelineState);
        encoder->setBuffer(ABuffer, 0, 0);
        encoder->setBuffer(BBuffer, 0, 1);
        encoder->setBuffer(biasBuffer, 0, 2);
        encoder->setBuffer(outBuffer, 0, 3);
        encoder->setBuffer(paramsBuffer, 0, 4);

        encoder->dispatchThreadgroups(MTL::Size(num_groups_y, num_groups_x, 1), MTL::Size(32*simd_groups, 1, 1));
        encoder->endEncoding();
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
        auto* gpuOut = static_cast<Out*>(outBuffer->contents());
        std::copy(gpuOut, gpuOut + out.matrix.size(), out.matrix.begin());
    }
};

struct MetalRelu {
    NS::AutoreleasePool* pool;
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::ComputePipelineState* fwdPipeline;
    MTL::ComputePipelineState* bwdPipeline;

    MTL::Buffer* xBuffer;
    MTL::Buffer* maskBuffer;
    MTL::Buffer* paramsBuffer;
    size_t xBytes;
    size_t maskBytes;
    ReluInputParams params;
    uint32_t groupSize;

    MetalRelu(int r, int c) :xBytes(r * c * sizeof(__bf16)), maskBytes(r*c*sizeof(uint8_t)), params(r * c) {
        pool = NS::AutoreleasePool::alloc()->init();
        device = MTL::CreateSystemDefaultDevice();
        commandQueue = device->newCommandQueue();
        MTL::Library* library = device->newDefaultLibrary();
        NS::Error* error = nullptr;
        fwdPipeline = device->newComputePipelineState(library->newFunction(makeNSString("reluForward")), &error);
        bwdPipeline = device->newComputePipelineState(library->newFunction(makeNSString("reluBackward")), &error);

        xBuffer = device->newBuffer(xBytes, MTL::ResourceStorageModeShared);
        maskBuffer = device->newBuffer(maskBytes, MTL::ResourceStorageModeShared);
        paramsBuffer = device->newBuffer(sizeof(ReluInputParams), MTL::ResourceStorageModeShared);


        std::memcpy(paramsBuffer->contents(), &params, sizeof(ReluInputParams));

        library->release();
    }
    ~MetalRelu() {
        paramsBuffer->release();
        maskBuffer->release();
        xBuffer->release();
        fwdPipeline->release();
        bwdPipeline->release();
        commandQueue->release();
        pool->release();
        device->release();
        }
    void forward(Matrix<__bf16>& x) {
        std::memcpy(xBuffer->contents(), x.matrix.data(), xBytes);

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

        encoder->setComputePipelineState(fwdPipeline);
        encoder->setBuffer(xBuffer, 0, 0);
        encoder->setBuffer(maskBuffer, 0, 1);
        encoder->setBuffer(paramsBuffer, 0, 3);

        encoder->dispatchThreadgroups(MTL::Size(params.count/1024, 1, 1), MTL::Size(1024, 1, 1));
        encoder->endEncoding();

        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();

        auto* gpuOut = static_cast<__bf16*>(xBuffer->contents());
        std::copy(gpuOut, gpuOut + x.matrix.size(), x.matrix.begin());
    }
    void backward(Matrix<__bf16>& dZ) {
        std::memcpy(xBuffer->contents(), dZ.matrix.data(), xBytes);

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

        encoder->setComputePipelineState(bwdPipeline);
        encoder->setBuffer(xBuffer, 0, 0);
        encoder->setBuffer(maskBuffer, 0, 1);
        encoder->setBuffer(paramsBuffer, 0, 2);

        encoder->dispatchThreadgroups(MTL::Size(params.count/1024, 1, 1), MTL::Size(1024, 1, 1));
        encoder->endEncoding();

        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();

        auto* gpuOut = static_cast<__bf16*>(xBuffer->contents());
        std::copy(gpuOut, gpuOut + dZ.matrix.size(), dZ.matrix.begin());
    }
};

struct MetalSoftmax {
    NS::AutoreleasePool* pool;
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::ComputePipelineState* fwdPipeline;
    MTL::ComputePipelineState* bwdPipeline;


    MTL::Buffer* xBuffer;
    MTL::Buffer* outBuffer;
    MTL::Buffer* paramsBuffer;
    size_t xBytes;
    SoftMaxParams params;

    MetalSoftmax(int r, int c) :xBytes(r * c * sizeof(__bf16)), params(r, c, std::min(c, 1024)) {
        pool = NS::AutoreleasePool::alloc()->init();
        device = MTL::CreateSystemDefaultDevice();
        commandQueue = device->newCommandQueue();
        MTL::Library* library = device->newDefaultLibrary();

        NS::Error* error = nullptr;
        fwdPipeline = device->newComputePipelineState(library->newFunction(makeNSString("softmaxForward")), &error);
        bwdPipeline = device->newComputePipelineState(library->newFunction(makeNSString("softmaxBackward")), &error);

        xBuffer = device->newBuffer(xBytes, MTL::ResourceStorageModeShared);
        outBuffer = device->newBuffer(xBytes, MTL::ResourceStorageModeShared);
        paramsBuffer = device->newBuffer(sizeof(SoftMaxParams), MTL::ResourceStorageModeShared);


        std::memcpy(paramsBuffer->contents(), &params, sizeof(SoftMaxParams));

        library->release();
    }
    ~MetalSoftmax() {
        paramsBuffer->release();
        outBuffer->release();
        xBuffer->release();
        fwdPipeline->release();
        bwdPipeline->release();
        commandQueue->release();
        pool->release();
        device->release();
        }
    void forward(const Matrix<__bf16>& x, Matrix<__bf16>& out) {
        std::memcpy(xBuffer->contents(), x.matrix.data(), xBytes);

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

        encoder->setComputePipelineState(fwdPipeline);
        encoder->setBuffer(xBuffer, 0, 0);
        encoder->setBuffer(outBuffer, 0, 1);
        encoder->setBuffer(paramsBuffer, 0, 2);

        encoder->dispatchThreadgroups(MTL::Size(params.rows, 1, 1), MTL::Size(params.group_size, 1, 1));
        encoder->endEncoding();

        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();

        auto* gpuOut = static_cast<__bf16*>(outBuffer->contents());
        std::copy(gpuOut, gpuOut + out.matrix.size(), out.matrix.begin());
    }
    void backward(Matrix<__bf16>& dZ) {
        std::memcpy(xBuffer->contents(), dZ.matrix.data(), xBytes);

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

        encoder->setComputePipelineState(bwdPipeline);
        encoder->setBuffer(outBuffer, 0, 0); //cached output
        encoder->setBuffer(xBuffer, 0, 1); //reused for dZ
        encoder->setBuffer(paramsBuffer, 0, 2);

        encoder->dispatchThreadgroups(MTL::Size(params.rows, 1, 1), MTL::Size(params.group_size, 1, 1));
        encoder->endEncoding();

        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();

        auto* gpuOut = static_cast<__bf16*>(xBuffer->contents());
        std::copy(gpuOut, gpuOut + dZ.matrix.size(), dZ.matrix.begin());
    }
};

struct MetalLoss {
    NS::AutoreleasePool* pool;
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::ComputePipelineState* fwdPipeline;
    MTL::ComputePipelineState* bwdPipeline;

    MTL::Buffer* xBuffer;
    MTL::Buffer* outBuffer;
    MTL::Buffer* yBuffer;
    MTL::Buffer* lseBuffer;
    MTL::Buffer* dZBuffer;
    MTL::Buffer* paramsBuffer;
    size_t xBytes;
    size_t yBytes;
    size_t outBytes;
    size_t lseBytes;
    LossParams params;

    MetalLoss(int r, int c) : xBytes(r * c * sizeof(__bf16)), yBytes(r * sizeof(uint32_t)), outBytes(r*sizeof(float)), lseBytes(r*sizeof(float)), params(r, c, std::min(c, 1024)) {
        pool = NS::AutoreleasePool::alloc()->init();
        device = MTL::CreateSystemDefaultDevice();
        commandQueue = device->newCommandQueue();
        MTL::Library* library = device->newDefaultLibrary();

        NS::Error* error = nullptr;
        fwdPipeline = device->newComputePipelineState(library->newFunction(makeNSString("lossForward")), &error);
        bwdPipeline = device->newComputePipelineState(library->newFunction(makeNSString("lossBackward")), &error);

        xBuffer = device->newBuffer(xBytes, MTL::ResourceStorageModeShared);
        yBuffer = device->newBuffer(yBytes, MTL::ResourceStorageModeShared);
        outBuffer = device->newBuffer(outBytes, MTL::ResourceStorageModeShared);
        lseBuffer = device->newBuffer(lseBytes, MTL::ResourceStorageModeShared);
        dZBuffer = device->newBuffer(xBytes, MTL::ResourceStorageModeShared);

        paramsBuffer = device->newBuffer(sizeof(LossParams), MTL::ResourceStorageModeShared);

        std::memcpy(paramsBuffer->contents(), &params, sizeof(LossParams));

        library->release();
    }
    ~MetalLoss() {
        paramsBuffer->release();
        outBuffer->release();
        xBuffer->release();
        yBuffer->release();
        lseBuffer->release();
        dZBuffer->release();
        fwdPipeline->release();
        bwdPipeline->release();
        commandQueue->release();
        pool->release();
        device->release();
        }
    void forward(const Matrix<__bf16>& x, const Matrix<uint32_t>& y, Matrix<float>& out) {
        std::memcpy(xBuffer->contents(), x.matrix.data(), xBytes);
        std::memcpy(yBuffer->contents(), y.matrix.data(), yBytes);


        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

        encoder->setComputePipelineState(fwdPipeline);
        encoder->setBuffer(xBuffer, 0, 0);
        encoder->setBuffer(yBuffer, 0, 1);
        encoder->setBuffer(outBuffer, 0, 2);
        encoder->setBuffer(lseBuffer, 0, 3);
        encoder->setBuffer(paramsBuffer, 0, 4);

        encoder->dispatchThreadgroups(MTL::Size(params.rows, 1, 1), MTL::Size(params.group_size, 1, 1));
        encoder->endEncoding();

        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();

        auto* gpuOut = static_cast<float*>(outBuffer->contents());
        std::copy(gpuOut, gpuOut + out.matrix.size(), out.matrix.begin());
    }
    void backward(Matrix<__bf16>& dZ) {
        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

        encoder->setComputePipelineState(bwdPipeline);
        encoder->setBuffer(xBuffer, 0, 0);
        encoder->setBuffer(yBuffer, 0, 1);
        encoder->setBuffer(lseBuffer, 0, 2);
        encoder->setBuffer(dZBuffer, 0, 3);
        encoder->setBuffer(paramsBuffer, 0, 4);

        encoder->dispatchThreadgroups(MTL::Size(params.rows, 1, 1), MTL::Size(params.group_size, 1, 1));
        encoder->endEncoding();

        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();

        auto* gpuOut = static_cast<__bf16*>(dZBuffer->contents());
        std::copy(gpuOut, gpuOut + dZ.matrix.size(), dZ.matrix.begin());
    }
};

struct MetalOptimStep {
    NS::AutoreleasePool* pool;
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::ComputePipelineState* pipelineState;

    MTL::Buffer* parameterBuffer;
    MTL::Buffer* gradientBuffer;
    MTL::Buffer* mBuffer;
    MTL::Buffer* vBuffer;

    MTL::Buffer* paramsBuffer;
    size_t parameterBytes;
    size_t gradientBytes;
    size_t mBytes;
    size_t vBytes;
    StepParams params;

    MetalOptimStep(int p) : parameterBytes(p * sizeof(float)), gradientBytes(p * sizeof(__bf16)), mBytes(p*sizeof(__bf16)), vBytes(p*sizeof(__bf16)), params(p) {
        pool = NS::AutoreleasePool::alloc()->init();
        device = MTL::CreateSystemDefaultDevice();
        commandQueue = device->newCommandQueue();
        MTL::Library* library = device->newDefaultLibrary();
        MTL::Function* function = library->newFunction(makeNSString("MetalStep"));

        NS::Error* error = nullptr;
        pipelineState = device->newComputePipelineState(function, &error);

        parameterBuffer = device->newBuffer(parameterBytes, MTL::ResourceStorageModeShared);
        gradientBuffer = device->newBuffer(gradientBytes, MTL::ResourceStorageModeShared);
        mBuffer = device->newBuffer(mBytes, MTL::ResourceStorageModeShared);
        vBuffer = device->newBuffer(vBytes, MTL::ResourceStorageModeShared);

        paramsBuffer = device->newBuffer(sizeof(StepParams), MTL::ResourceStorageModeShared);

        function->release();
        library->release();
    }
    ~MetalOptimStep() {
            paramsBuffer->release();
            parameterBuffer->release();
            gradientBuffer->release();
            mBuffer->release();
            vBuffer->release();
            pipelineState->release();
            commandQueue->release();
            pool->release();
            device->release();
        }
    void run(std::vector<float>& p, std::vector<__bf16>& g, std::vector<__bf16>& m, std::vector<__bf16>& v, float& t) {
        std::memcpy(parameterBuffer->contents(), p.data(), parameterBytes);
        std::memcpy(gradientBuffer->contents(), g.data(), gradientBytes);
        std::memcpy(mBuffer->contents(), m.data(), mBytes);
        std::memcpy(vBuffer->contents(), v.data(), vBytes);
        params.lr_t = LR * std::sqrt(1.0f-std::pow(params.beta2, t)) / (1.0f - std::pow(params.beta1, t));

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

        encoder->setComputePipelineState(pipelineState);
        encoder->setBuffer(parameterBuffer, 0, 0);
        encoder->setBuffer(gradientBuffer, 0, 1);
        encoder->setBuffer(mBuffer, 0, 2);
        encoder->setBuffer(vBuffer, 0, 3);
        encoder->setBytes(&params, sizeof(params), 4);

        encoder->dispatchThreadgroups(MTL::Size(params.paramCount/16384, 1, 1), MTL::Size(256, 1, 1));
        encoder->endEncoding();

        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();

        auto* parameterOut = static_cast<float*>(parameterBuffer->contents());
        std::copy(parameterOut, parameterOut + p.size(), p.begin());
        auto* gradientOut = static_cast<__bf16*>(gradientBuffer->contents());
        std::copy(gradientOut, gradientOut + g.size(), g.begin());
        auto* mOut = static_cast<__bf16*>(mBuffer->contents());
        std::copy(mOut, mOut + m.size(), m.begin());
        auto* vOut = static_cast<__bf16*>(vBuffer->contents());
        std::copy(vOut, vOut + v.size(), v.begin());
    }
};

struct AdamOptimizer {
    std::vector<std::vector<__bf16>*> weights;
    std::vector<std::vector<__bf16>*> grads;
    std::vector<size_t> offsets;
    std::vector<float> master;
    std::vector<__bf16> g;
    std::vector<__bf16> m;
    std::vector<__bf16> v;
    float t = 0.0f;
    size_t count = 0;
    MetalOptimStep* stepper = nullptr;

    void add(std::vector<__bf16>& w, std::vector<__bf16>& dw) {
        weights.push_back(&w);
        grads.push_back(&dw);
    }
    void add(Matrix<__bf16>& w, Matrix<__bf16>& dw) {
        add(w.matrix, dw.matrix);
    }
    void build() {
        size_t total = 0;
        offsets.clear();
        for (std::vector<__bf16>* w : weights) {
            offsets.push_back(total);
            total += w->size();
        }
        count = ((total + 16383) / 16384) * 16384;
        master.assign(count, 0.0f);
        g.assign(count, (__bf16)0.0f);
        m.assign(count, (__bf16)0.0f);
        v.assign(count, (__bf16)0.0f);
        for (size_t i = 0; i < weights.size(); ++i) {
            std::transform(weights[i]->begin(), weights[i]->end(), master.begin() + offsets[i],
                           [](__bf16 x) { return (float)x; });
        }
        delete stepper;
        stepper = new MetalOptimStep((int)count);
    }
    void step() {
        for (size_t i = 0; i < grads.size(); ++i) {
            std::copy(grads[i]->begin(), grads[i]->end(), g.begin() + offsets[i]);
        }
        t += 1.0f;
        stepper->run(master, g, m, v, t);
        for (size_t i = 0; i < weights.size(); ++i) {
            std::transform(master.begin() + offsets[i], master.begin() + offsets[i] + weights[i]->size(),
                           weights[i]->begin(), [](float x) { return (__bf16)x; });
        }
    }
    ~AdamOptimizer() {
        delete stepper;
    }
};

struct MetalLayernorm {
    NS::AutoreleasePool* pool;
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::ComputePipelineState* fwdPipeline;
    MTL::ComputePipelineState* bwdPipeline;

    MTL::Buffer* xBuffer;
    MTL::Buffer* xnormBuffer;
    MTL::Buffer* outBuffer;
    MTL::Buffer* gammaBuffer;
    MTL::Buffer* betaBuffer;
    MTL::Buffer* stdevBuffer;
    MTL::Buffer* paramsBuffer;

    MTL::Buffer* d_xnormBuffer;
    MTL::Buffer* d_gammaBuffer;
    MTL::Buffer* d_betaBuffer;

    size_t xBytes;
    size_t xnormBytes;
    size_t outBytes;
    size_t gammaBytes;  //bf16: the gamma/beta weight inputs
    size_t betaBytes;
    size_t dGammaBytes; //float: the d_gamma/d_beta atomic accumulators
    size_t dBetaBytes;
    size_t stdevBytes;
    LayernormParams params;

    MetalLayernorm(int r, int c) : xBytes(r * c * sizeof(__bf16)), xnormBytes(r * c * sizeof(__bf16)), outBytes(r*c*sizeof(__bf16)), gammaBytes(c*sizeof(__bf16)), betaBytes(c*sizeof(__bf16)), dGammaBytes(c*sizeof(float)), dBetaBytes(c*sizeof(float)), stdevBytes(r*sizeof(float)), params(r, c) {
        pool = NS::AutoreleasePool::alloc()->init();
        device = MTL::CreateSystemDefaultDevice();
        commandQueue = device->newCommandQueue();
        MTL::Library* library = device->newDefaultLibrary();

        NS::Error* error = nullptr;
        fwdPipeline = device->newComputePipelineState(library->newFunction(makeNSString("layernormForward")), &error);
        bwdPipeline = device->newComputePipelineState(library->newFunction(makeNSString("layernormBackward")), &error);

        xBuffer = device->newBuffer(xBytes, MTL::ResourceStorageModeShared);
        xnormBuffer = device->newBuffer(xnormBytes, MTL::ResourceStorageModeShared);
        outBuffer = device->newBuffer(outBytes, MTL::ResourceStorageModeShared);
        gammaBuffer = device->newBuffer(gammaBytes, MTL::ResourceStorageModeShared);
        betaBuffer = device->newBuffer(betaBytes, MTL::ResourceStorageModeShared);
        stdevBuffer = device->newBuffer(stdevBytes, MTL::ResourceStorageModeShared);

        d_gammaBuffer = device->newBuffer(dGammaBytes, MTL::ResourceStorageModeShared);
        d_betaBuffer = device->newBuffer(dBetaBytes, MTL::ResourceStorageModeShared);
        d_xnormBuffer = device->newBuffer(xnormBytes, MTL::ResourceStorageModeShared);

        paramsBuffer = device->newBuffer(sizeof(LayernormParams), MTL::ResourceStorageModeShared);

        library->release();
    }
    ~MetalLayernorm() {
        xBuffer->release();
        xnormBuffer->release();
        outBuffer->release();
        gammaBuffer->release();
        betaBuffer->release();
        stdevBuffer->release();
        d_xnormBuffer->release();
        d_gammaBuffer->release();
        d_betaBuffer->release();
        paramsBuffer->release();
        fwdPipeline->release();
        bwdPipeline->release();
        commandQueue->release();
        pool->release();
        device->release();
    }
    void forward(const Matrix<__bf16>& x, Matrix<__bf16>& out, const std::vector<__bf16>& gamma, const std::vector<__bf16>& beta) {
        std::memcpy(xBuffer->contents(), x.matrix.data(), xBytes);
        std::memcpy(gammaBuffer->contents(), gamma.data(), gammaBytes);
        std::memcpy(betaBuffer->contents(), beta.data(), betaBytes);

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

        encoder->setComputePipelineState(fwdPipeline);
        encoder->setBuffer(xBuffer, 0, 0);
        encoder->setBuffer(xnormBuffer, 0, 1);
        encoder->setBuffer(outBuffer, 0, 2);
        encoder->setBuffer(gammaBuffer, 0, 3);
        encoder->setBuffer(betaBuffer, 0, 4);
        encoder->setBuffer(stdevBuffer, 0, 5);

        encoder->setBytes(&params, sizeof(params), 6);

        encoder->dispatchThreadgroups(MTL::Size(params.rows, 1, 1), MTL::Size(params.group_size, 1, 1));
        encoder->endEncoding();

        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();

        auto* gpuOut = static_cast<__bf16*>(outBuffer->contents());
        std::copy(gpuOut, gpuOut + out.matrix.size(), out.matrix.begin());
    }
    void backward(Matrix<__bf16>& dZ, Matrix<__bf16>& d_beta,  Matrix<__bf16>& d_gamma) {
        std::memcpy(outBuffer->contents(), dZ.matrix.data(), outBytes);
        std::memset(d_gammaBuffer->contents(), 0, dGammaBytes);
        std::memset(d_betaBuffer->contents(), 0, dBetaBytes);

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

        encoder->setComputePipelineState(bwdPipeline);
        encoder->setBuffer(outBuffer, 0, 0);
        encoder->setBuffer(xnormBuffer, 0, 1);
        encoder->setBuffer(gammaBuffer, 0, 2);
        encoder->setBuffer(stdevBuffer, 0, 3);
        encoder->setBuffer(d_xnormBuffer, 0, 5);
        encoder->setBuffer(d_gammaBuffer, 0, 6);
        encoder->setBuffer(d_betaBuffer, 0, 7);

        encoder->setBytes(&params, sizeof(params), 8);

        encoder->dispatchThreadgroups(MTL::Size(params.rows, 1, 1), MTL::Size(params.group_size, 1, 1));
        encoder->endEncoding();

        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();

        auto* gpuOut = static_cast<__bf16*>(outBuffer->contents());
        std::copy(gpuOut, gpuOut + dZ.matrix.size(), dZ.matrix.begin());
        auto* gammaOut = static_cast<float*>(d_gammaBuffer->contents());
        std::copy(gammaOut, gammaOut + d_gamma.matrix.size(), d_gamma.matrix.begin());
        auto* betaOut = static_cast<float*>(d_betaBuffer->contents());
        std::copy(betaOut, betaOut + d_beta.matrix.size(), d_beta.matrix.begin());
    }
};

struct MetalEmbed {
    NS::AutoreleasePool* pool;
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::ComputePipelineState* fwdPipeline;
    MTL::ComputePipelineState* bwdPipeline;

    MTL::Buffer* inputBuffer;
    MTL::Buffer* outBuffer;
    MTL::Buffer* weightBuffer;
    MTL::Buffer* paramsBuffer;

    MTL::Buffer* d_weightBuffer;

    size_t inputBytes;
    size_t outBytes;
    size_t weightBytes;
    EmbedParams params;

    MetalEmbed(int t, int v, int e) : inputBytes(t* sizeof(uint32_t)), outBytes(t*e * sizeof(__bf16)), weightBytes(v*e*sizeof(__bf16)), params(t, e, std::min(e, 1024)) {
        pool = NS::AutoreleasePool::alloc()->init();
        device = MTL::CreateSystemDefaultDevice();
        commandQueue = device->newCommandQueue();
        MTL::Library* library = device->newDefaultLibrary();

        NS::Error* error = nullptr;
        fwdPipeline = device->newComputePipelineState(library->newFunction(makeNSString("embedForward")), &error);
        bwdPipeline = device->newComputePipelineState(library->newFunction(makeNSString("embedBackward")), &error);

        inputBuffer = device->newBuffer(inputBytes, MTL::ResourceStorageModeShared);
        outBuffer = device->newBuffer(outBytes, MTL::ResourceStorageModeShared);
        weightBuffer = device->newBuffer(weightBytes, MTL::ResourceStorageModeShared);
        d_weightBuffer = device->newBuffer(v * e * sizeof(float), MTL::ResourceStorageModeShared);


        paramsBuffer = device->newBuffer(sizeof(EmbedParams), MTL::ResourceStorageModeShared);

        library->release();
    }
    ~MetalEmbed() {
        inputBuffer->release();
        outBuffer->release();
        weightBuffer->release();
        paramsBuffer->release();

        fwdPipeline->release();
        bwdPipeline->release();
        commandQueue->release();
        pool->release();
        device->release();
    }
    void forward(const std::vector<uint32_t>& input, Matrix<__bf16>& output, const Matrix<__bf16>& embed_weights) {
        std::memcpy(inputBuffer->contents(), input.data(), inputBytes);
        std::memcpy(weightBuffer->contents(), embed_weights.matrix.data(), weightBytes);

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

        encoder->setComputePipelineState(fwdPipeline);
        encoder->setBuffer(inputBuffer, 0, 0);
        encoder->setBuffer(weightBuffer, 0, 1);
        encoder->setBuffer(outBuffer, 0, 2);

        encoder->setBytes(&params, sizeof(params), 3);

        encoder->dispatchThreadgroups(MTL::Size(params.t, 1, 1), MTL::Size(params.group_size, 1, 1));
        encoder->endEncoding();

        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();

        auto* gpuOut = static_cast<__bf16*>(outBuffer->contents());
        std::copy(gpuOut, gpuOut + output.matrix.size(), output.matrix.begin());
    }
    void backward(Matrix<__bf16>& dZ, Matrix<__bf16>& d_weights) {
        std::memcpy(outBuffer->contents(), dZ.matrix.data(), outBytes);
        std::memset(d_weightBuffer->contents(), 0, d_weights.matrix.size() * sizeof(float));

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

        encoder->setComputePipelineState(bwdPipeline);
        encoder->setBuffer(inputBuffer, 0, 0);
        encoder->setBuffer(outBuffer, 0, 1);
        encoder->setBuffer(d_weightBuffer, 0, 2);

        encoder->setBytes(&params, sizeof(params), 3);

        encoder->dispatchThreadgroups(MTL::Size(params.t, 1, 1), MTL::Size(params.group_size, 1, 1));
        encoder->endEncoding();

        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();

        auto* d_weightOut = static_cast<float*>(d_weightBuffer->contents());
        std::copy(d_weightOut, d_weightOut + d_weights.matrix.size(), d_weights.matrix.begin());
    }
};


struct MetalAttention {
    NS::AutoreleasePool* pool;
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::ComputePipelineState* qkvPipeline;
    MTL::ComputePipelineState* dPipeline;
    MTL::ComputePipelineState* flashattnfwdPipeline;
    MTL::ComputePipelineState* flashattnbwdPipeline;
    MTL::ComputePipelineState* graddXPipeline;
    MTL::ComputePipelineState* graddWPipeline;
    MTL::ComputePipelineState* projfwdPipeline;
    MTL::ComputePipelineState* projbwddWPipeline;
    MTL::ComputePipelineState* projbwddXPipeline;

    MTL::Buffer* XBuffer;
    MTL::Buffer* QBuffer;
    MTL::Buffer* KBuffer;
    MTL::Buffer* VBuffer;
    MTL::Buffer* qwBuffer;
    MTL::Buffer* kwBuffer;
    MTL::Buffer* vwBuffer;
    MTL::Buffer* OutBuffer;
    MTL::Buffer* projwBuffer;
    MTL::Buffer* projOutBuffer;
    MTL::Buffer* dResBuffer;
    MTL::Buffer* dProjwBuffer;
    MTL::Buffer* paramsBuffer;

    MTL::Buffer* dBuffer;
    MTL::Buffer* lBuffer;
    MTL::Buffer* dZBuffer;
    MTL::Buffer* dQBuffer;
    MTL::Buffer* dKBuffer;
    MTL::Buffer* dVBuffer;
    MTL::Buffer* dQwBuffer;
    MTL::Buffer* dKwBuffer;
    MTL::Buffer* dVwBuffer;
    MTL::Buffer* dXBuffer;

    size_t xBytes;
    size_t qBytes;
    size_t kvBytes;
    size_t qkv_weightBytes;
    size_t dBytes;
    size_t outBytes;
    size_t lBytes; 

    AttentionParams params;
    uint32_t attn_tiles_y;
    uint32_t attn_tiles_x;
    uint32_t tile_m = 64;
    uint32_t tile_n = 64;
    uint32_t simd_groups = 4;


    MetalAttention(int m, int n, int d) : xBytes(m*n*sizeof(__bf16)), qBytes(m*n*sizeof(float)), kvBytes(m*n*sizeof(__bf16)), qkv_weightBytes(n*n*sizeof(__bf16)), dBytes(m*n/d*sizeof(__bf16)), lBytes(m*n/d *sizeof(float)), outBytes(m*n*sizeof(__bf16)), params(m, n, d, n/d) {
        pool = NS::AutoreleasePool::alloc()->init();
        device = MTL::CreateSystemDefaultDevice();
        commandQueue = device->newCommandQueue();
        MTL::Library* library = device->newDefaultLibrary();

        NS::Error* error = nullptr;
        qkvPipeline = device->newComputePipelineState(library->newFunction(makeNSString("computeQKV")), &error);
        dPipeline = device->newComputePipelineState(library->newFunction(makeNSString("computeD")), &error);
        flashattnfwdPipeline = device->newComputePipelineState(library->newFunction(makeNSString("flashAttentionForward")), &error);
        flashattnbwdPipeline = device->newComputePipelineState(library->newFunction(makeNSString("flashAttentionBackward")), &error);
        graddXPipeline = device->newComputePipelineState(library->newFunction(makeNSString("computeGradDx")), &error);
        graddWPipeline = device->newComputePipelineState(library->newFunction(makeNSString("computeGradDw")), &error);
        projfwdPipeline = device->newComputePipelineState(library->newFunction(makeNSString("projForward")), &error);
        projbwddWPipeline = device->newComputePipelineState(library->newFunction(makeNSString("projBackwarddW")), &error);
        projbwddXPipeline = device->newComputePipelineState(library->newFunction(makeNSString("projBackwarddX")), &error);

        if (!qkvPipeline || !flashattnfwdPipeline || !projfwdPipeline || !projbwddWPipeline || !projbwddXPipeline
            || !flashattnbwdPipeline || !graddXPipeline || !graddWPipeline || !dPipeline) {
            std::cerr << "pipeline: " << error->localizedDescription()->utf8String() << "\n";
            std::exit(1);
        }

        attn_tiles_y = m / tile_m;
        attn_tiles_x = n / tile_n;

        XBuffer = device->newBuffer(xBytes, MTL::ResourceStorageModeShared);
        QBuffer = device->newBuffer(qBytes, MTL::ResourceStorageModeShared);
        KBuffer = device->newBuffer(kvBytes, MTL::ResourceStorageModeShared);
        VBuffer = device->newBuffer(kvBytes, MTL::ResourceStorageModeShared);
        qwBuffer = device->newBuffer(qkv_weightBytes, MTL::ResourceStorageModeShared);
        kwBuffer = device->newBuffer(qkv_weightBytes, MTL::ResourceStorageModeShared);
        vwBuffer = device->newBuffer(qkv_weightBytes, MTL::ResourceStorageModeShared);
        OutBuffer = device->newBuffer(outBytes, MTL::ResourceStorageModeShared);
        projwBuffer = device->newBuffer(qkv_weightBytes, MTL::ResourceStorageModeShared);
        projOutBuffer = device->newBuffer(outBytes, MTL::ResourceStorageModeShared);
        dResBuffer = device->newBuffer(outBytes, MTL::ResourceStorageModeShared);
        dProjwBuffer = device->newBuffer(qkv_weightBytes, MTL::ResourceStorageModeShared);

        dBuffer = device->newBuffer(dBytes, MTL::ResourceStorageModeShared);
        lBuffer = device->newBuffer(lBytes, MTL::ResourceStorageModeShared);
        dZBuffer = device->newBuffer(outBytes, MTL::ResourceStorageModeShared);
        dQBuffer = device->newBuffer(qBytes, MTL::ResourceStorageModeShared);
        dKBuffer = device->newBuffer(kvBytes, MTL::ResourceStorageModeShared);
        dVBuffer = device->newBuffer(kvBytes, MTL::ResourceStorageModeShared);
        dQwBuffer = device->newBuffer(n*n*sizeof(float), MTL::ResourceStorageModeShared); //float: matmul2d needs a float destination to mix the float dQ with bfloat operands
        dKwBuffer = device->newBuffer(qkv_weightBytes, MTL::ResourceStorageModeShared);
        dVwBuffer = device->newBuffer(qkv_weightBytes, MTL::ResourceStorageModeShared);
        dXBuffer = device->newBuffer(qBytes, MTL::ResourceStorageModeShared); //float, same reason

        paramsBuffer = device->newBuffer(sizeof(AttentionParams), MTL::ResourceStorageModeShared);

        std::memcpy(paramsBuffer->contents(), &params, sizeof(AttentionParams));

        library->release();
    }
    void forward(const Matrix<__bf16>& x, const Matrix<__bf16>& qw, const Matrix<__bf16>& kw, const Matrix<__bf16>& vw, const Matrix<__bf16>& projw, Matrix<__bf16>& out) {
        std::memcpy(XBuffer->contents(), x.matrix.data(), xBytes);
        std::memcpy(qwBuffer->contents(), qw.matrix.data(), qkv_weightBytes);
        std::memcpy(kwBuffer->contents(), kw.matrix.data(), qkv_weightBytes);
        std::memcpy(vwBuffer->contents(), vw.matrix.data(), qkv_weightBytes);
        std::memcpy(projwBuffer->contents(), projw.matrix.data(), qkv_weightBytes);

        std::memset(QBuffer->contents(), 0, qBytes);
        std::memset(KBuffer->contents(), 0, kvBytes);
        std::memset(VBuffer->contents(), 0, kvBytes);
        
        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        
        //compute QKV
        MTL::ComputeCommandEncoder* qkvEncoder = commandBuffer->computeCommandEncoder();
        qkvEncoder->setComputePipelineState(qkvPipeline);
        qkvEncoder->setBuffer(XBuffer, 0, 0);
        qkvEncoder->setBuffer(QBuffer, 0, 1);
        qkvEncoder->setBuffer(KBuffer, 0, 2);
        qkvEncoder->setBuffer(VBuffer, 0, 3);
        qkvEncoder->setBuffer(qwBuffer, 0, 4);
        qkvEncoder->setBuffer(kwBuffer, 0, 5);
        qkvEncoder->setBuffer(vwBuffer, 0, 6);
        qkvEncoder->setBuffer(paramsBuffer, 0, 7);

        qkvEncoder->dispatchThreadgroups(MTL::Size(attn_tiles_y, attn_tiles_x, 1), MTL::Size(32*simd_groups, 1, 1));
        qkvEncoder->endEncoding();
        
        //flash attn 
        MTL::ComputeCommandEncoder* flashattnfwdEncoder = commandBuffer->computeCommandEncoder();
        flashattnfwdEncoder->setComputePipelineState(flashattnfwdPipeline);
        flashattnfwdEncoder->setThreadgroupMemoryLength(tile_m * params.D * sizeof(float), 0); //Accumulator
        flashattnfwdEncoder->setThreadgroupMemoryLength(tile_n*tile_m*sizeof(__bf16), 1);//softmax_out tile buffer
        flashattnfwdEncoder->setThreadgroupMemoryLength(tile_m * sizeof(float), 2);
        flashattnfwdEncoder->setThreadgroupMemoryLength(tile_m * sizeof(float), 3);
        flashattnfwdEncoder->setBuffer(QBuffer, 0, 0);
        flashattnfwdEncoder->setBuffer(KBuffer, 0, 1);
        flashattnfwdEncoder->setBuffer(VBuffer, 0, 2);
        flashattnfwdEncoder->setBuffer(OutBuffer, 0, 3);
        flashattnfwdEncoder->setBuffer(lBuffer, 0, 4);
        flashattnfwdEncoder->setBuffer(paramsBuffer, 0, 5);

        flashattnfwdEncoder->dispatchThreadgroups(MTL::Size(attn_tiles_y * params.NH, 1, 1), MTL::Size(32*simd_groups, 1, 1));
        flashattnfwdEncoder->endEncoding();
        
        //projection

        MTL::ComputeCommandEncoder* projfwdEncoder = commandBuffer->computeCommandEncoder();
        projfwdEncoder->setComputePipelineState(projfwdPipeline);
        projfwdEncoder->setBuffer(OutBuffer, 0, 0);
        projfwdEncoder->setBuffer(projwBuffer, 0, 1);
        projfwdEncoder->setBuffer(projOutBuffer, 0, 2);
        projfwdEncoder->setBuffer(paramsBuffer, 0, 3);

        projfwdEncoder->dispatchThreadgroups(MTL::Size(attn_tiles_y, attn_tiles_x, 1), MTL::Size(32*simd_groups, 1, 1));
        projfwdEncoder->endEncoding();

        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();

        auto* gpuOut = static_cast<__bf16*>(projOutBuffer->contents());
        std::copy(gpuOut, gpuOut + out.matrix.size(), out.matrix.begin());
    }
    void backward (const Matrix<__bf16>& dRes, Matrix<__bf16>& dprojw, Matrix<__bf16>& dX, Matrix<__bf16>& dqw, Matrix<__bf16>& dkw, Matrix<__bf16>& dvw) {
        std::memcpy(dResBuffer->contents(), dRes.matrix.data(), outBytes);
        std::memset(dProjwBuffer->contents(), 0, qkv_weightBytes);
        std::memset(dQBuffer->contents(), 0, qBytes);
        std::memset(dKBuffer->contents(), 0, kvBytes);
        std::memset(dVBuffer->contents(), 0, kvBytes);
        std::memset(dQwBuffer->contents(), 0, params.K*params.K*sizeof(float));
        std::memset(dKwBuffer->contents(), 0, qkv_weightBytes);
        std::memset(dVwBuffer->contents(), 0, qkv_weightBytes);
        std::memset(dXBuffer->contents(), 0, qBytes);

        //proj backward
        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* projbwddWEncoder = commandBuffer->computeCommandEncoder();

        projbwddWEncoder->setComputePipelineState(projbwddWPipeline);
        projbwddWEncoder->setBuffer(OutBuffer, 0, 0);
        projbwddWEncoder->setBuffer(dResBuffer, 0, 1);
        projbwddWEncoder->setBuffer(dProjwBuffer, 0, 2);
        projbwddWEncoder->setBuffer(paramsBuffer, 0, 3);

        projbwddWEncoder->dispatchThreadgroups(MTL::Size(attn_tiles_x, attn_tiles_x, 1), MTL::Size(32*simd_groups, 1, 1));
        projbwddWEncoder->endEncoding();

        MTL::ComputeCommandEncoder* projbwddXEncoder = commandBuffer->computeCommandEncoder();

        projbwddXEncoder->setComputePipelineState(projbwddXPipeline);
        projbwddXEncoder->setBuffer(dResBuffer, 0, 0);
        projbwddXEncoder->setBuffer(projwBuffer, 0, 1);
        projbwddXEncoder->setBuffer(dZBuffer, 0, 2);
        projbwddXEncoder->setBuffer(paramsBuffer, 0, 3);        
        projbwddXEncoder->dispatchThreadgroups(MTL::Size(attn_tiles_x, attn_tiles_y, 1), MTL::Size(32*simd_groups, 1, 1));
        projbwddXEncoder->endEncoding();
        
        //compute D
        MTL::ComputeCommandEncoder* dEncoder = commandBuffer->computeCommandEncoder();

        dEncoder->setComputePipelineState(dPipeline);
        dEncoder->setBuffer(dZBuffer, 0, 0);
        dEncoder->setBuffer(OutBuffer, 0, 1);
        dEncoder->setBuffer(dBuffer, 0, 2);
        dEncoder->setBuffer(paramsBuffer, 0, 3);

        dEncoder->dispatchThreadgroups(MTL::Size(attn_tiles_y * params.NH, 1, 1), MTL::Size(32*simd_groups, 1, 1));
        dEncoder->endEncoding();
        
        //flashattnbwd
        MTL::ComputeCommandEncoder* flashattnbwdEncoder = commandBuffer->computeCommandEncoder();
        
        
        flashattnbwdEncoder->setComputePipelineState(flashattnbwdPipeline);
        flashattnbwdEncoder->setThreadgroupMemoryLength(tile_m * tile_n * sizeof(__bf16), 0); //S
        flashattnbwdEncoder->setThreadgroupMemoryLength(tile_m * tile_n * sizeof(__bf16), 1); //dP
        flashattnbwdEncoder->setThreadgroupMemoryLength(tile_m * tile_n * sizeof(__bf16), 2); //dS
        flashattnbwdEncoder->setBuffer(dZBuffer, 0, 0);
        flashattnbwdEncoder->setBuffer(dBuffer, 0, 1);
        flashattnbwdEncoder->setBuffer(lBuffer, 0, 2);
        flashattnbwdEncoder->setBuffer(QBuffer, 0, 3);
        flashattnbwdEncoder->setBuffer(KBuffer, 0, 4);
        flashattnbwdEncoder->setBuffer(VBuffer, 0, 5);
        flashattnbwdEncoder->setBuffer(dQBuffer, 0, 6);
        flashattnbwdEncoder->setBuffer(dKBuffer, 0, 7);
        flashattnbwdEncoder->setBuffer(dVBuffer, 0, 8);
        flashattnbwdEncoder->setBuffer(paramsBuffer, 0, 9);
        
        flashattnbwdEncoder->dispatchThreadgroups(MTL::Size(attn_tiles_y * params.NH, 1, 1), MTL::Size(32*simd_groups, 1, 1));
        flashattnbwdEncoder->endEncoding();
        
        //compute grad
        
        //dX portion
        MTL::ComputeCommandEncoder* graddXEncoder = commandBuffer->computeCommandEncoder();

        graddXEncoder->setComputePipelineState(graddXPipeline);
        graddXEncoder->setBuffer(dXBuffer, 0, 0);
        graddXEncoder->setBuffer(qwBuffer, 0, 1);
        graddXEncoder->setBuffer(kwBuffer, 0, 2);
        graddXEncoder->setBuffer(vwBuffer, 0, 3);
        graddXEncoder->setBuffer(dQBuffer, 0, 4);
        graddXEncoder->setBuffer(dKBuffer, 0, 5);
        graddXEncoder->setBuffer(dVBuffer, 0, 6);
        graddXEncoder->setBuffer(paramsBuffer, 0, 7);

        graddXEncoder->dispatchThreadgroups(MTL::Size(attn_tiles_x, attn_tiles_y, 1), MTL::Size(32*simd_groups, 1, 1));
        graddXEncoder->endEncoding();
        
        //dW portion
        MTL::ComputeCommandEncoder* graddWEncoder = commandBuffer->computeCommandEncoder();

        graddWEncoder->setComputePipelineState(graddWPipeline);
        graddWEncoder->setBuffer(XBuffer, 0, 0);
        graddWEncoder->setBuffer(dQBuffer, 0, 1);
        graddWEncoder->setBuffer(dKBuffer, 0, 2);
        graddWEncoder->setBuffer(dVBuffer, 0, 3);
        graddWEncoder->setBuffer(dQwBuffer, 0, 4);
        graddWEncoder->setBuffer(dKwBuffer, 0, 5);
        graddWEncoder->setBuffer(dVwBuffer, 0, 6);
        graddWEncoder->setBuffer(paramsBuffer, 0, 7);

        graddWEncoder->dispatchThreadgroups(MTL::Size(attn_tiles_x, attn_tiles_x, 1), MTL::Size(32*simd_groups, 1, 1));
        graddWEncoder->endEncoding();
        
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
        
        auto* dprojOut = static_cast<__bf16*>(dProjwBuffer->contents());
        std::copy(dprojOut, dprojOut + dprojw.matrix.size(), dprojw.matrix.begin());
        auto* dXOut = static_cast<float*>(dXBuffer->contents());
        for (size_t i = 0; i < dX.matrix.size(); ++i) dX.matrix[i] = __bf16(dXOut[i]);
        auto* dqwOut = static_cast<float*>(dQwBuffer->contents());
        for (size_t i = 0; i < dqw.matrix.size(); ++i) dqw.matrix[i] = __bf16(dqwOut[i]);
        auto* dkwOut = static_cast<__bf16*>(dKwBuffer->contents());
        std::copy(dkwOut, dkwOut + dkw.matrix.size(), dkw.matrix.begin());
        auto* dvwOut = static_cast<__bf16*>(dVwBuffer->contents());
        std::copy(dvwOut, dvwOut + dvw.matrix.size(), dvw.matrix.begin());
    }
    ~MetalAttention() {
        paramsBuffer->release();
        XBuffer->release();
        QBuffer->release();
        KBuffer->release();
        VBuffer->release();
        qwBuffer->release();
        kwBuffer->release();
        vwBuffer->release();
        OutBuffer->release();
        projwBuffer->release();
        projOutBuffer->release();
        dResBuffer->release();
        dProjwBuffer->release();
        dBuffer->release();
        lBuffer->release();
        dZBuffer->release();
        dQBuffer->release();
        dKBuffer->release();
        dVBuffer->release();
        dQwBuffer->release();
        dKwBuffer->release();
        dVwBuffer->release();
        dXBuffer->release();
        qkvPipeline->release();
        dPipeline->release();
        flashattnfwdPipeline->release();
        flashattnbwdPipeline->release();
        graddXPipeline->release();
        graddWPipeline->release();
        projfwdPipeline->release();
        projbwddWPipeline->release();
        projbwddXPipeline->release();
        commandQueue->release();
        pool->release();
        device->release();

    }

};

struct MetalMLP {
    NS::AutoreleasePool* pool;
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::ComputePipelineState* fwdOnePipeline;
    MTL::ComputePipelineState* fwdTwoPipeline;
    MTL::ComputePipelineState* bwdOnePipeline;
    MTL::ComputePipelineState* bwdTwoPipeline;

    MTL::Buffer* xBuffer;
    MTL::Buffer* vBuffer;
    MTL::Buffer* outBuffer;
    MTL::Buffer* maskBuffer;
    MTL::Buffer* upprojBuffer;
    MTL::Buffer* downprojBuffer;
    
    MTL::Buffer* dXBuffer;
    MTL::Buffer* dZBuffer;
    MTL::Buffer* dUpBuffer;
    MTL::Buffer* dDpBuffer;
    MTL::Buffer* dVBuffer; //device scratch for dV: too big for threadgroup memory

    MTL::Buffer* paramsBuffer;
    mlpParams p;
    
    size_t xBytes;
    size_t vBytes;
    size_t maskBytes;
    size_t upBytes;
    size_t dpBytes;
    uint32_t tile_m = 64;
    uint32_t tile_n = 64;
    uint32_t simd_groups = 4;
    uint32_t tile_y;
    uint32_t tile_x;

    MetalMLP(int m, int n, int s) : xBytes(m*n*sizeof(__bf16)), vBytes(m*n*s*sizeof(__bf16)), maskBytes(m*n*s*sizeof(uint8_t)), upBytes(n*n*s*sizeof(__bf16)), dpBytes(n*s*n*sizeof(__bf16)), p(m,n,s) {
        pool = NS::AutoreleasePool::alloc()->init();
        device = MTL::CreateSystemDefaultDevice();
        commandQueue = device->newCommandQueue();
        MTL::Library* library = device->newDefaultLibrary();

        NS::Error* error = nullptr;
        fwdOnePipeline = device->newComputePipelineState(library->newFunction(makeNSString("mlpForwardOne")), &error);
        fwdTwoPipeline = device->newComputePipelineState(library->newFunction(makeNSString("mlpForwardTwo")), &error);
        bwdOnePipeline = device->newComputePipelineState(library->newFunction(makeNSString("mlpBackwardOne")), &error);
        bwdTwoPipeline = device->newComputePipelineState(library->newFunction(makeNSString("mlpBackwardTwo")), &error);

        if (!fwdOnePipeline || !fwdTwoPipeline || !bwdOnePipeline || !bwdTwoPipeline) {
            std::cerr << "pipeline: " << error->localizedDescription()->utf8String() << "\n";
            std::exit(1);
        }

        tile_y = m / tile_m;
        tile_x = n / tile_n;
        

        xBuffer = device->newBuffer(xBytes, MTL::ResourceStorageModeShared);
        vBuffer = device->newBuffer(vBytes, MTL::ResourceStorageModeShared);
        outBuffer = device->newBuffer(xBytes, MTL::ResourceStorageModeShared);
        maskBuffer = device->newBuffer(maskBytes, MTL::ResourceStorageModeShared);
        upprojBuffer = device->newBuffer(upBytes, MTL::ResourceStorageModeShared);
        downprojBuffer = device->newBuffer(dpBytes, MTL::ResourceStorageModeShared);
        
        dXBuffer = device->newBuffer(xBytes, MTL::ResourceStorageModeShared);
        dZBuffer = device->newBuffer(xBytes, MTL::ResourceStorageModeShared);
        dUpBuffer = device->newBuffer(upBytes, MTL::ResourceStorageModeShared);
        dDpBuffer = device->newBuffer(dpBytes, MTL::ResourceStorageModeShared);
        dVBuffer = device->newBuffer(vBytes, MTL::ResourceStorageModeShared);
                
        paramsBuffer = device->newBuffer(sizeof(mlpParams), MTL::ResourceStorageModeShared);

        std::memcpy(paramsBuffer->contents(), &p, sizeof(mlpParams));

        library->release();
    }
    void forward(const Matrix<__bf16>& x, const Matrix<__bf16>& upProj, const Matrix<__bf16>& downProj, Matrix<__bf16>& out) {
        std::memcpy(xBuffer->contents(), x.matrix.data(), xBytes);
        std::memcpy(upprojBuffer->contents(), upProj.matrix.data(), upBytes);
        std::memcpy(downprojBuffer->contents(), downProj.matrix.data(), dpBytes);
        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

        encoder->setComputePipelineState(fwdOnePipeline);
        encoder->setBuffer(xBuffer, 0, 0);
        encoder->setBuffer(upprojBuffer, 0, 1);
        encoder->setBuffer(vBuffer, 0, 2);
        encoder->setBuffer(maskBuffer, 0, 3);
        encoder->setBuffer(paramsBuffer, 0, 4);
        
        encoder->dispatchThreadgroups(MTL::Size(tile_y, p.N*p.S/tile_n, 1), MTL::Size(32*simd_groups, 1, 1));
        encoder->endEncoding();

        MTL::ComputeCommandEncoder* twoEncoder = commandBuffer->computeCommandEncoder();
        twoEncoder->setComputePipelineState(fwdTwoPipeline);
        twoEncoder->setBuffer(downprojBuffer, 0, 0);
        twoEncoder->setBuffer(vBuffer, 0, 1);
        twoEncoder->setBuffer(outBuffer, 0, 2);
        twoEncoder->setBuffer(paramsBuffer, 0, 3);
        twoEncoder->dispatchThreadgroups(MTL::Size(tile_y, tile_x, 1), MTL::Size(32*simd_groups, 1, 1));
        twoEncoder->endEncoding();

        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();

        auto* gpuOut = static_cast<__bf16*>(outBuffer->contents());
        std::copy(gpuOut, gpuOut + out.matrix.size(), out.matrix.begin());
     }
     void backward(const Matrix<__bf16>& dZ, Matrix<__bf16>& dX, Matrix<__bf16>& dUp, Matrix<__bf16>& dDp) {
        std::memcpy(dZBuffer->contents(), dZ.matrix.data(), xBytes);
        std::memset(dUpBuffer->contents(), 0, upBytes);
        std::memset(dDpBuffer->contents(), 0, dpBytes);
        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

        encoder->setComputePipelineState(bwdOnePipeline);
        encoder->setBuffer(downprojBuffer, 0, 1);
        encoder->setBuffer(maskBuffer, 0, 2);
        encoder->setBuffer(dZBuffer, 0, 3);
        encoder->setBuffer(dVBuffer, 0, 5);
        encoder->setBuffer(paramsBuffer, 0, 6);

        encoder->dispatchThreadgroups(MTL::Size(tile_y, p.N*p.S/tile_n, 1), MTL::Size(32*simd_groups, 1, 1));
        encoder->endEncoding();


        MTL::ComputeCommandEncoder* upEncoder = commandBuffer->computeCommandEncoder();
        upEncoder->setComputePipelineState(bwdTwoPipeline);
        upEncoder->setBuffer(xBuffer, 0, 0);
        upEncoder->setBuffer(dXBuffer, 0, 1);
        upEncoder->setBuffer(dVBuffer, 0, 2);
        upEncoder->setBuffer(upprojBuffer, 0, 3);
        upEncoder->setBuffer(dUpBuffer, 0, 4);
        upEncoder->setBuffer(vBuffer, 0, 5);
        upEncoder->setBuffer(dZBuffer, 0, 6);
        upEncoder->setBuffer(dDpBuffer, 0, 7);
        upEncoder->setBuffer(paramsBuffer, 0, 8);

        upEncoder->dispatchThreadgroups(MTL::Size(std::max(p.N*p.S/tile_m, (p.M/tile_m)*(p.N/tile_n)), 1, 1), MTL::Size(32*simd_groups, 1, 1));
        upEncoder->endEncoding();

        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
        if (commandBuffer->status() == MTL::CommandBufferStatusError) {
            auto* e = commandBuffer->error();
            std::cerr << "mlp bwd cmdbuf: " << (e ? e->localizedDescription()->utf8String() : "unknown") << "\n";
        }

        auto* gpuOut = static_cast<__bf16*>(dXBuffer->contents());
        std::copy(gpuOut, gpuOut + dX.matrix.size(), dX.matrix.begin());
        auto* dUpOut = static_cast<__bf16*>(dUpBuffer->contents());
        std::copy(dUpOut, dUpOut + dUp.matrix.size(), dUp.matrix.begin());
        auto* dDpOut = static_cast<__bf16*>(dDpBuffer->contents());
        std::copy(dDpOut, dDpOut + dDp.matrix.size(), dDp.matrix.begin());
    }
    ~MetalMLP() {
        paramsBuffer->release();
        xBuffer->release();
        vBuffer->release();
        upprojBuffer->release();
        downprojBuffer->release();
        maskBuffer->release();
        outBuffer->release();
        dZBuffer->release();
        dDpBuffer->release();
        dUpBuffer->release();
        dVBuffer->release();
        fwdOnePipeline->release();
        fwdTwoPipeline->release();
        bwdOnePipeline->release();
        bwdTwoPipeline->release();
        commandQueue->release();
        pool->release();
        device->release();

    }
};

struct MetalLinear {
    NS::AutoreleasePool* pool;
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::ComputePipelineState* fwdPipeline;
    MTL::ComputePipelineState* bwdPipeline;

    MTL::Buffer* xBuffer;
    MTL::Buffer* wBuffer;
    MTL::Buffer* outBuffer;
    MTL::Buffer* dZBuffer;
    MTL::Buffer* dXBuffer;
    MTL::Buffer* dWBuffer;
    MTL::Buffer* paramsBuffer;

    size_t xBytes;
    size_t wBytes;
    size_t outBytes;
    MatMulParams params;
    uint32_t tiles_m;
    uint32_t tiles_n;
    uint32_t simd_groups = 4;

    MetalLinear(int m, int k, int n) : xBytes(m*k*sizeof(__bf16)), wBytes(k*n*sizeof(__bf16)), outBytes(m*n*sizeof(__bf16)), params(m, n, k) {
        pool = NS::AutoreleasePool::alloc()->init();
        device = MTL::CreateSystemDefaultDevice();
        commandQueue = device->newCommandQueue();
        MTL::Library* library = device->newDefaultLibrary();

        NS::Error* error = nullptr;
        fwdPipeline = device->newComputePipelineState(library->newFunction(makeNSString("linearForward")), &error);
        bwdPipeline = device->newComputePipelineState(library->newFunction(makeNSString("linearBackward")), &error);

        if (!fwdPipeline || !bwdPipeline) {
            std::cerr << "pipeline: " << error->localizedDescription()->utf8String() << "\n";
            std::exit(1);
        }

        tiles_m = m / 64;
        tiles_n = n / 64;

        xBuffer = device->newBuffer(xBytes, MTL::ResourceStorageModeShared);
        wBuffer = device->newBuffer(wBytes, MTL::ResourceStorageModeShared);
        outBuffer = device->newBuffer(outBytes, MTL::ResourceStorageModeShared);
        dZBuffer = device->newBuffer(outBytes, MTL::ResourceStorageModeShared);
        dXBuffer = device->newBuffer(xBytes, MTL::ResourceStorageModeShared);
        dWBuffer = device->newBuffer(wBytes, MTL::ResourceStorageModeShared);
        paramsBuffer = device->newBuffer(sizeof(MatMulParams), MTL::ResourceStorageModeShared);

        std::memcpy(paramsBuffer->contents(), &params, sizeof(MatMulParams));

        library->release();
    }
    ~MetalLinear() {
        paramsBuffer->release();
        xBuffer->release();
        wBuffer->release();
        outBuffer->release();
        dZBuffer->release();
        dXBuffer->release();
        dWBuffer->release();
        fwdPipeline->release();
        bwdPipeline->release();
        commandQueue->release();
        pool->release();
        device->release();
    }
    void forward(const Matrix<__bf16>& x, const Matrix<__bf16>& w, Matrix<__bf16>& out) {
        std::memcpy(xBuffer->contents(), x.matrix.data(), xBytes);
        std::memcpy(wBuffer->contents(), w.matrix.data(), wBytes);

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

        encoder->setComputePipelineState(fwdPipeline);
        encoder->setBuffer(xBuffer, 0, 0);
        encoder->setBuffer(wBuffer, 0, 1);
        encoder->setBuffer(outBuffer, 0, 2);
        encoder->setBuffer(paramsBuffer, 0, 3);

        encoder->dispatchThreadgroups(MTL::Size(tiles_m, tiles_n, 1), MTL::Size(32*simd_groups, 1, 1));
        encoder->endEncoding();

        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
        if (commandBuffer->status() == MTL::CommandBufferStatusError) {
            auto* e = commandBuffer->error();
            std::cerr << "linear fwd cmdbuf: " << (e ? e->localizedDescription()->utf8String() : "unknown") << "\n";
        }

        auto* gpuOut = static_cast<__bf16*>(outBuffer->contents());
        std::copy(gpuOut, gpuOut + out.matrix.size(), out.matrix.begin());
    }
    void backward(const Matrix<__bf16>& dZ, Matrix<__bf16>& dX, Matrix<__bf16>& dW) {
        std::memcpy(dZBuffer->contents(), dZ.matrix.data(), outBytes);
        std::memset(dWBuffer->contents(), 0, wBytes);

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

        encoder->setComputePipelineState(bwdPipeline);
        encoder->setBuffer(xBuffer, 0, 0);
        encoder->setBuffer(wBuffer, 0, 1);
        encoder->setBuffer(dZBuffer, 0, 2);
        encoder->setBuffer(dXBuffer, 0, 3);
        encoder->setBuffer(dWBuffer, 0, 4);
        encoder->setBuffer(paramsBuffer, 0, 5);

        encoder->dispatchThreadgroups(MTL::Size(tiles_m, 1, 1), MTL::Size(32*simd_groups, 1, 1));
        encoder->endEncoding();

        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
        if (commandBuffer->status() == MTL::CommandBufferStatusError) {
            auto* e = commandBuffer->error();
            std::cerr << "linear bwd cmdbuf: " << (e ? e->localizedDescription()->utf8String() : "unknown") << "\n";
        }

        auto* dXOut = static_cast<__bf16*>(dXBuffer->contents());
        std::copy(dXOut, dXOut + dX.matrix.size(), dX.matrix.begin());
        auto* dWOut = static_cast<__bf16*>(dWBuffer->contents());
        std::copy(dWOut, dWOut + dW.matrix.size(), dW.matrix.begin());
    }
};

struct BlockWeights {
    std::vector<__bf16> ln1_gamma, ln1_beta;
    Matrix<__bf16> qw, kw, vw, proj_weights;
    std::vector<__bf16> ln2_gamma, ln2_beta;
    Matrix<__bf16> mlp_up, mlp_down;
    BlockWeights() : ln1_gamma(N_EMBED, (__bf16)1.0f), ln1_beta(N_EMBED, (__bf16)0.0f),
        qw(N_EMBED, N_HEADS*HEAD_SIZE), kw(N_EMBED, N_HEADS*HEAD_SIZE), vw(N_EMBED, N_HEADS*HEAD_SIZE),
        proj_weights(N_EMBED, N_EMBED),
        ln2_gamma(N_EMBED, (__bf16)1.0f), ln2_beta(N_EMBED, (__bf16)0.0f),
        mlp_up(N_EMBED, MLP_SCALE*N_EMBED), mlp_down(MLP_SCALE*N_EMBED, N_EMBED) {
        qw.fillXavier();
        kw.fillXavier();
        vw.fillXavier();
        proj_weights.fillNormal(0.02f / std::sqrt(2.0f * N_LAYERS));
        mlp_up.fillKaiming();
        mlp_down.fillNormal(0.02f / std::sqrt(2.0f * N_LAYERS));
    }
};

struct ModelWeights {
    Matrix<__bf16> embed_weights, pos_weights;
    std::vector<BlockWeights> blocks;
    std::vector<__bf16> ln_final_gamma, ln_final_beta;
    Matrix<__bf16> output_proj;
    ModelWeights(int vocab_size) : embed_weights(vocab_size, N_EMBED), pos_weights(BLOCK_SIZE, N_EMBED),
        blocks(N_LAYERS),
        ln_final_gamma(N_EMBED, (__bf16)1.0f), ln_final_beta(N_EMBED, (__bf16)0.0f),
        output_proj(N_EMBED, vocab_size) {
        embed_weights.fillNormal(0.02f);
        pos_weights.fillNormal(0.02f);
        output_proj.fillNormal(0.02f);
    }
};

struct BlockGrads {
    Matrix<__bf16> d_ln1_gamma, d_ln1_beta;
    Matrix<__bf16> d_qw, d_kw, d_vw, d_proj_weights;
    Matrix<__bf16> d_ln2_gamma, d_ln2_beta;
    Matrix<__bf16> d_mlp_up, d_mlp_down;
    BlockGrads() : d_ln1_gamma(1, N_EMBED), d_ln1_beta(1, N_EMBED),
        d_qw(N_EMBED, N_HEADS*HEAD_SIZE), d_kw(N_EMBED, N_HEADS*HEAD_SIZE), d_vw(N_EMBED, N_HEADS*HEAD_SIZE),
        d_proj_weights(N_EMBED, N_EMBED),
        d_ln2_gamma(1, N_EMBED), d_ln2_beta(1, N_EMBED),
        d_mlp_up(N_EMBED, MLP_SCALE*N_EMBED), d_mlp_down(MLP_SCALE*N_EMBED, N_EMBED) {}
};

struct ModelGrads {
    Matrix<__bf16> d_embed_weights, d_pos_weights;
    std::vector<BlockGrads> blocks;
    Matrix<__bf16> d_ln_final_gamma, d_ln_final_beta;
    Matrix<__bf16> d_output_proj;
    ModelGrads(int vocab_size) : d_embed_weights(vocab_size, N_EMBED), d_pos_weights(BLOCK_SIZE, N_EMBED),
        blocks(N_LAYERS),
        d_ln_final_gamma(1, N_EMBED), d_ln_final_beta(1, N_EMBED),
        d_output_proj(N_EMBED, vocab_size) {}
};

struct BlockCache {
    Matrix<__bf16> ln1_out, attn_proj_out, residual1;
    Matrix<__bf16> ln2_out, mlp_out, residual2;
    Matrix<__bf16> d_ln1, d_ln2;
    BlockCache() : ln1_out(BLOCK_SIZE, N_EMBED),
        attn_proj_out(BLOCK_SIZE, N_EMBED), residual1(BLOCK_SIZE, N_EMBED),
        ln2_out(BLOCK_SIZE, N_EMBED), mlp_out(BLOCK_SIZE, N_EMBED), residual2(BLOCK_SIZE, N_EMBED),
        d_ln1(BLOCK_SIZE, N_EMBED), d_ln2(BLOCK_SIZE, N_EMBED) {}
};

struct ModelCache {
    std::vector<uint32_t> x;
    Matrix<uint32_t> y;
    Matrix<__bf16> embed_out, d_embed_out;
    std::vector<BlockCache> blocks;
    Matrix<__bf16> ln_final_out, d_ln_final;
    Matrix<__bf16> logits, d_logits;
    Matrix<float> loss_out;
    float loss = 0.0f;
    ModelCache(int vocab_size) : x(BLOCK_SIZE, 0), y(BLOCK_SIZE, 1),
        embed_out(BLOCK_SIZE, N_EMBED), d_embed_out(BLOCK_SIZE, N_EMBED),
        blocks(N_LAYERS),
        ln_final_out(BLOCK_SIZE, N_EMBED), d_ln_final(BLOCK_SIZE, N_EMBED),
        logits(BLOCK_SIZE, vocab_size), d_logits(BLOCK_SIZE, vocab_size),
        loss_out(BLOCK_SIZE, 1) {}
};

void registerParams(AdamOptimizer& opt, ModelWeights& w, ModelGrads& g) {
    opt.add(w.embed_weights, g.d_embed_weights);
    opt.add(w.pos_weights, g.d_pos_weights);
    for (int n = 0; n < N_LAYERS; ++n) {
        opt.add(w.blocks[n].ln1_gamma, g.blocks[n].d_ln1_gamma.matrix);
        opt.add(w.blocks[n].ln1_beta, g.blocks[n].d_ln1_beta.matrix);
        opt.add(w.blocks[n].qw, g.blocks[n].d_qw);
        opt.add(w.blocks[n].kw, g.blocks[n].d_kw);
        opt.add(w.blocks[n].vw, g.blocks[n].d_vw);
        opt.add(w.blocks[n].proj_weights, g.blocks[n].d_proj_weights);
        opt.add(w.blocks[n].ln2_gamma, g.blocks[n].d_ln2_gamma.matrix);
        opt.add(w.blocks[n].ln2_beta, g.blocks[n].d_ln2_beta.matrix);
        opt.add(w.blocks[n].mlp_up, g.blocks[n].d_mlp_up);
        opt.add(w.blocks[n].mlp_down, g.blocks[n].d_mlp_down);
    }
    opt.add(w.ln_final_gamma, g.d_ln_final_gamma.matrix);
    opt.add(w.ln_final_beta, g.d_ln_final_beta.matrix);
    opt.add(w.output_proj, g.d_output_proj);
    opt.build();
}

void getBatch(const std::vector<uint32_t>& encoded_text, std::vector<uint32_t>& x, Matrix<uint32_t>& y) {
    std::uniform_int_distribution<size_t> dist(0, encoded_text.size() - BLOCK_SIZE - 2);
    size_t idx = dist(generator);
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        x[i] = encoded_text[idx + i];
        y.matrix[i] = encoded_text[idx + i + 1];
    }
}

struct MetalBlock {
    MetalLayernorm ln1;
    MetalAttention attn;
    MetalLayernorm ln2;
    MetalMLP mlp;
    MetalBlock() : ln1(BLOCK_SIZE, N_EMBED), attn(BLOCK_SIZE, N_EMBED, HEAD_SIZE),
        ln2(BLOCK_SIZE, N_EMBED), mlp(BLOCK_SIZE, N_EMBED, MLP_SCALE) {}
};

void addInto(Matrix<__bf16>& acc, const Matrix<__bf16>& other) {
    for (size_t i = 0; i < acc.matrix.size(); ++i) {
        acc.matrix[i] = (__bf16)((float)acc.matrix[i] + (float)other.matrix[i]);
    }
}

//Model depends on the structs and constants above, so it gets included here
//rather than with the headers at the top
#include "model.h"

int main() {
    if (HEAD_SIZE * N_HEADS != N_EMBED) {
        throw std::invalid_argument ("Head size x n_heads must equal n_embed (for my sanity)");
    }
    uint32_t vocab_size = 0;
    build_mapping(TEXT, encoder, decoder, vocab_size);
    std::vector<uint32_t> data;
    encode(TEXT, data, encoder);
    int padded_vocab = (int)(vocab_size + 63) / 64 * 64;

    ModelWeights weights(padded_vocab);
    ModelGrads grads(padded_vocab);

    Model model(BLOCK_SIZE, padded_vocab, N_EMBED, MLP_SCALE, HEAD_SIZE, weights, grads);

    const int max_iters = 300;
    for (int iter = 1; iter <= max_iters; ++iter) {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<uint32_t> x(BLOCK_SIZE, 0);
        Matrix<uint32_t> y(BLOCK_SIZE, 1);
        
        getBatch(data, x, y);
        
        float loss = model.forward(x, y, weights);
        model.backward(grads);
        model.step();
        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        std::cout << "iter " << iter << " | loss " << loss << " | " << ms << " ms" << std::endl;
        // Jade telemetry: feeds the Loss chart + persistent run comparison
        // (__FORGE_SCALAR|name|step|value|ts — swallowed from OUTPUT).
        std::cout << "__FORGE_SCALAR|loss|" << iter << "|" << loss << "|0" << std::endl;
    }
    return 0;
}