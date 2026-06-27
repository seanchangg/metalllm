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

std::random_device rd{};
std::mt19937 generator{2};

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

static const float LR = 0.003f;

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
template <> struct MPPMatmulBiasKernel<__bf16, float> {static constexpr const char* name = "MPPMatMulBiasBfloat";};

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
        auto t0 = std::chrono::high_resolution_clock::now();
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
        auto t1 = std::chrono::high_resolution_clock::now();
        auto* gpuOut = static_cast<Out*>(outBuffer->contents());
        std::copy(gpuOut, gpuOut + out.matrix.size(), out.matrix.begin());
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
        std::cout << "Elapsed time: " << elapsed.count() << " ms" << std::endl;
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


    MPPMatMulBias(int A_r, int A_c, int B_r, int B_c) :ABytes(A_r * A_c * sizeof(In)), BBytes(B_r * B_c * sizeof(In)), biasBytes(B_c * sizeof(In)), outBytes(A_r * B_c * sizeof(Out)), params(A_r, B_c, A_c) {
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
    void run(const Matrix<In>& a, const Matrix<In>& b, const Matrix<In>& bias, Matrix<Out>& out) {
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
        auto t0 = std::chrono::high_resolution_clock::now();
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
        auto t1 = std::chrono::high_resolution_clock::now();
        auto* gpuOut = static_cast<Out*>(outBuffer->contents());
        std::copy(gpuOut, gpuOut + out.matrix.size(), out.matrix.begin());
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
        std::cout << "Elapsed time: " << elapsed.count() << " ms" << std::endl;
    }
};

struct MetalReluForward {
    NS::AutoreleasePool* pool;
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::ComputePipelineState* pipelineState;

    MTL::Buffer* xBuffer;
    MTL::Buffer* maskBuffer;
    MTL::Buffer* outBuffer;
    MTL::Buffer* paramsBuffer;
    size_t xBytes;
    size_t maskBytes;
    ReluInputParams params;
    uint32_t groupSize;

    MetalReluForward(int r, int c) :xBytes(r * c * sizeof(__bf16)), maskBytes(r*c*sizeof(uint8_t)), params(r * c) {
        pool = NS::AutoreleasePool::alloc()->init();
        device = MTL::CreateSystemDefaultDevice();
        commandQueue = device->newCommandQueue();
        MTL::Library* library = device->newDefaultLibrary();
        MTL::Function* function = library->newFunction(makeNSString("reluForward"));

        NS::Error* error = nullptr;
        pipelineState = device->newComputePipelineState(function, &error);

        xBuffer = device->newBuffer(xBytes, MTL::ResourceStorageModeShared);
        maskBuffer = device->newBuffer(maskBytes, MTL::ResourceStorageModeShared);
        outBuffer = device->newBuffer(xBytes, MTL::ResourceStorageModeShared);
        paramsBuffer = device->newBuffer(sizeof(ReluInputParams), MTL::ResourceStorageModeShared);


        std::memcpy(paramsBuffer->contents(), &params, sizeof(ReluInputParams));

        function->release();
        library->release();
    }
    ~MetalReluForward() {
            paramsBuffer->release();
            outBuffer->release();
            maskBuffer->release();
            xBuffer->release();
            pipelineState->release();
            commandQueue->release();
            pool->release();
            device->release();
        }
    void run(const Matrix<__bf16>& x, Matrix<uint8_t>& mask, Matrix<__bf16>& out) {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::memcpy(xBuffer->contents(), x.matrix.data(), xBytes);

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

        encoder->setComputePipelineState(pipelineState);
        encoder->setBuffer(xBuffer, 0, 0);
        encoder->setBuffer(maskBuffer, 0, 1);
        encoder->setBuffer(outBuffer, 0, 2);
        encoder->setBuffer(paramsBuffer, 0, 3);

        encoder->dispatchThreadgroups(MTL::Size(params.count/1024, 1, 1), MTL::Size(1024, 1, 1));
        encoder->endEncoding();

        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();

        auto* gpuOut = static_cast<__bf16*>(outBuffer->contents());
        std::copy(gpuOut, gpuOut + out.matrix.size(), out.matrix.begin());
        auto* maskOut = static_cast<uint8_t*>(maskBuffer->contents());
        std::copy(maskOut, maskOut + mask.matrix.size(), mask.matrix.begin());
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = t1 - t0;
        std::cout << "Metal time: " << elapsed.count() << " ms\n";
    }
};

struct MetalSoftmaxForward {
    NS::AutoreleasePool* pool;
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::ComputePipelineState* pipelineState;

    MTL::Buffer* xBuffer;
    MTL::Buffer* maskBuffer;
    MTL::Buffer* outBuffer;
    MTL::Buffer* paramsBuffer;
    size_t xBytes;
    SoftMaxParams params;

    MetalSoftmaxForward(int r, int c) :xBytes(r * c * sizeof(__bf16)), params(r, c, std::min(c, 1024)) {
        pool = NS::AutoreleasePool::alloc()->init();
        device = MTL::CreateSystemDefaultDevice();
        commandQueue = device->newCommandQueue();
        MTL::Library* library = device->newDefaultLibrary();
        MTL::Function* function = library->newFunction(makeNSString("softmaxForward"));

        NS::Error* error = nullptr;
        pipelineState = device->newComputePipelineState(function, &error);

        xBuffer = device->newBuffer(xBytes, MTL::ResourceStorageModeShared);
        outBuffer = device->newBuffer(xBytes, MTL::ResourceStorageModeShared);
        paramsBuffer = device->newBuffer(sizeof(SoftMaxParams), MTL::ResourceStorageModeShared);


        std::memcpy(paramsBuffer->contents(), &params, sizeof(SoftMaxParams));

        function->release();
        library->release();
    }
    ~MetalSoftmaxForward() {
            paramsBuffer->release();
            outBuffer->release();
            xBuffer->release();
            pipelineState->release();
            commandQueue->release();
            pool->release();
            device->release();
        }
    void run(const Matrix<__bf16>& x, Matrix<__bf16>& out) {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::memcpy(xBuffer->contents(), x.matrix.data(), xBytes);

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

        encoder->setComputePipelineState(pipelineState);
        encoder->setBuffer(xBuffer, 0, 0);
        encoder->setBuffer(outBuffer, 0, 1);
        encoder->setBuffer(paramsBuffer, 0, 2);

        encoder->dispatchThreadgroups(MTL::Size(params.rows, 1, 1), MTL::Size(params.group_size, 1, 1));
        encoder->endEncoding();

        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();

        auto* gpuOut = static_cast<__bf16*>(outBuffer->contents());
        std::copy(gpuOut, gpuOut + out.matrix.size(), out.matrix.begin());
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = t1 - t0;
        std::cout << "Metal time: " << elapsed.count() << " ms\n";
    }
};

struct MetalLossForward {
    NS::AutoreleasePool* pool;
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::ComputePipelineState* pipelineState;

    MTL::Buffer* xBuffer;
    MTL::Buffer* outBuffer;
    MTL::Buffer* yBuffer;
    MTL::Buffer* lseBuffer;
    MTL::Buffer* paramsBuffer;
    size_t xBytes;
    size_t yBytes;
    size_t outBytes;
    size_t lseBytes;
    LossParams params;

    MetalLossForward(int r, int c) : xBytes(r * c * sizeof(__bf16)), yBytes(r * sizeof(uint32_t)), outBytes(r*sizeof(float)), lseBytes(r*sizeof(float)), params(r, c, std::min(c, 1024)) {
        pool = NS::AutoreleasePool::alloc()->init();
        device = MTL::CreateSystemDefaultDevice();
        commandQueue = device->newCommandQueue();
        MTL::Library* library = device->newDefaultLibrary();
        MTL::Function* function = library->newFunction(makeNSString("lossForward"));

        NS::Error* error = nullptr;
        pipelineState = device->newComputePipelineState(function, &error);

        xBuffer = device->newBuffer(xBytes, MTL::ResourceStorageModeShared);
        yBuffer = device->newBuffer(yBytes, MTL::ResourceStorageModeShared);
        outBuffer = device->newBuffer(outBytes, MTL::ResourceStorageModeShared);
        lseBuffer = device->newBuffer(lseBytes, MTL::ResourceStorageModeShared);

        paramsBuffer = device->newBuffer(sizeof(LossParams), MTL::ResourceStorageModeShared);

        std::memcpy(paramsBuffer->contents(), &params, sizeof(LossParams));

        function->release();
        library->release();
    }
    ~MetalLossForward() {
            paramsBuffer->release();
            outBuffer->release();
            xBuffer->release();
            yBuffer->release();
            lseBuffer->release();
            pipelineState->release();
            commandQueue->release();
            pool->release();
            device->release();
        }
    void run(const Matrix<__bf16>& x, const Matrix<uint32_t>& y, Matrix<float>& lse_cache, Matrix<float>& out) {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::memcpy(xBuffer->contents(), x.matrix.data(), xBytes);
        std::memcpy(yBuffer->contents(), y.matrix.data(), yBytes);


        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

        encoder->setComputePipelineState(pipelineState);
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
        auto* lseOut = static_cast<float*>(lseBuffer->contents());
        std::copy(lseOut, lseOut + lse_cache.matrix.size(), lse_cache.matrix.begin());
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = t1 - t0;
        std::cout << "Metal time: " << elapsed.count() << " ms\n";
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
            gradientBuffer->release();
            mBuffer->release();
            vBuffer->release();
            pipelineState->release();
            commandQueue->release();
            pool->release();
            device->release();
        }
    void run(std::vector<float>& p, std::vector<__bf16>& g, std::vector<__bf16>& m, std::vector<__bf16>& v, float& t) {
        auto t0 = std::chrono::high_resolution_clock::now();
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
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = t1 - t0;
        std::cout << "Metal time: " << elapsed.count() << " ms\n";
    }
};

struct MetalLayernormForward {
    NS::AutoreleasePool* pool;
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::ComputePipelineState* pipelineState;

    MTL::Buffer* xBuffer;
    MTL::Buffer* xnormBuffer;
    MTL::Buffer* outBuffer;
    MTL::Buffer* gammaBuffer;
    MTL::Buffer* betaBuffer;
    MTL::Buffer* stdevBuffer;
    MTL::Buffer* paramsBuffer;

    size_t xBytes;
    size_t xnormBytes;
    size_t outBytes;
    size_t gammaBytes;
    size_t betaBytes;
    size_t stdevBytes;
    LayernormParams params;

    MetalLayernormForward(int r, int c) : xBytes(r * c * sizeof(__bf16)), xnormBytes(r * c * sizeof(__bf16)), outBytes(r*c*sizeof(__bf16)), gammaBytes(c*sizeof(__bf16)), betaBytes(c*sizeof(__bf16)), stdevBytes(r*sizeof(float)), params(r, c) {
        pool = NS::AutoreleasePool::alloc()->init();
        device = MTL::CreateSystemDefaultDevice();
        commandQueue = device->newCommandQueue();
        MTL::Library* library = device->newDefaultLibrary();
        MTL::Function* function = library->newFunction(makeNSString("layernormForward"));

        NS::Error* error = nullptr;
        pipelineState = device->newComputePipelineState(function, &error);

        xBuffer = device->newBuffer(xBytes, MTL::ResourceStorageModeShared);
        xnormBuffer = device->newBuffer(xnormBytes, MTL::ResourceStorageModeShared);
        outBuffer = device->newBuffer(outBytes, MTL::ResourceStorageModeShared);
        gammaBuffer = device->newBuffer(gammaBytes, MTL::ResourceStorageModeShared);
        betaBuffer = device->newBuffer(betaBytes, MTL::ResourceStorageModeShared);
        stdevBuffer = device->newBuffer(stdevBytes, MTL::ResourceStorageModeShared);

        paramsBuffer = device->newBuffer(sizeof(LayernormParams), MTL::ResourceStorageModeShared);

        function->release();
        library->release();
    }
    ~MetalLayernormForward() {
        xBuffer->release();
        xnormBuffer->release();
        outBuffer->release();
        gammaBuffer->release();
        betaBuffer->release();
        stdevBuffer->release();
        paramsBuffer->release();

        pipelineState->release();
        commandQueue->release();
        pool->release();
        device->release();
    }
    void run(const Matrix<__bf16>& x, Matrix<__bf16>& x_norm, Matrix<__bf16>& out, const std::vector<__bf16>& gamma, const std::vector<__bf16>& beta, std::vector<float>& stdev) {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::memcpy(xBuffer->contents(), x.matrix.data(), xBytes);
        std::memcpy(gammaBuffer->contents(), gamma.data(), gammaBytes);
        std::memcpy(betaBuffer->contents(), beta.data(), betaBytes);

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

        encoder->setComputePipelineState(pipelineState);
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

        auto* xnormOut = static_cast<__bf16*>(xnormBuffer->contents());
        std::copy(xnormOut, xnormOut + x_norm.matrix.size(), x_norm.matrix.begin());
        auto* gpuOut = static_cast<__bf16*>(outBuffer->contents());
        std::copy(gpuOut, gpuOut + out.matrix.size(), out.matrix.begin());
        auto* stdevOut = static_cast<float*>(stdevBuffer->contents());
        std::copy(stdevOut, stdevOut + stdev.size(), stdev.begin());
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = t1 - t0;
        std::cout << "Metal time: " << elapsed.count() << " ms\n";
    }
};

struct MetalEmbedForward {
    NS::AutoreleasePool* pool;
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::ComputePipelineState* pipelineState;

    MTL::Buffer* inputBuffer;
    MTL::Buffer* outputBuffer;
    MTL::Buffer* weightBuffer;
    MTL::Buffer* paramsBuffer;

    size_t inputBytes;
    size_t outputBytes;
    size_t weightBytes;
    EmbedParams params;

    MetalEmbedForward(int t, int v, int e) : inputBytes(t* sizeof(uint32_t)), outputBytes(t*e * sizeof(__bf16)), weightBytes(v*e*sizeof(__bf16)), params(t, e, std::min(e, 1024)) {
        pool = NS::AutoreleasePool::alloc()->init();
        device = MTL::CreateSystemDefaultDevice();
        commandQueue = device->newCommandQueue();
        MTL::Library* library = device->newDefaultLibrary();
        MTL::Function* function = library->newFunction(makeNSString("embedForward"));

        NS::Error* error = nullptr;
        pipelineState = device->newComputePipelineState(function, &error);

        inputBuffer = device->newBuffer(inputBytes, MTL::ResourceStorageModeShared);
        outputBuffer = device->newBuffer(outputBytes, MTL::ResourceStorageModeShared);
        weightBuffer = device->newBuffer(weightBytes, MTL::ResourceStorageModeShared);

        paramsBuffer = device->newBuffer(sizeof(LayernormParams), MTL::ResourceStorageModeShared);

        function->release();
        library->release();
    }
    ~MetalEmbedForward() {
        inputBuffer->release();
        outputBuffer->release();
        weightBuffer->release();
        paramsBuffer->release();

        pipelineState->release();
        commandQueue->release();
        pool->release();
        device->release();
    }
    void run(const std::vector<uint32_t>& input, Matrix<__bf16>& output, const Matrix<__bf16>& embed_weights) {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::memcpy(inputBuffer->contents(), input.data(), inputBytes);
        std::memcpy(weightBuffer->contents(), embed_weights.matrix.data(), weightBytes);

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

        encoder->setComputePipelineState(pipelineState);
        encoder->setBuffer(inputBuffer, 0, 0);
        encoder->setBuffer(outputBuffer, 0, 1);
        encoder->setBuffer(weightBuffer, 0, 2);

        encoder->setBytes(&params, sizeof(params), 3);

        encoder->dispatchThreadgroups(MTL::Size(params.t, 1, 1), MTL::Size(params.group_size, 1, 1));
        encoder->endEncoding();

        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();

        auto* gpuOut = static_cast<__bf16*>(outputBuffer->contents());
        std::copy(gpuOut, gpuOut + output.matrix.size(), output.matrix.begin());
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = t1 - t0;
        std::cout << "Metal time: " << elapsed.count() << " ms\n";
    }
};

struct ComputeQKV {
    NS::AutoreleasePool* pool;
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::ComputePipelineState* pipelineState;

    MTL::Buffer* XBuffer;
    MTL::Buffer* QBuffer;
    MTL::Buffer* KBuffer;
    MTL::Buffer* VBuffer;
    MTL::Buffer* Q_weightBuffer;
    MTL::Buffer* K_weightBuffer;
    MTL::Buffer* V_weightBuffer;
    MTL::Buffer* paramsBuffer;

    size_t xBytes;
    size_t qBytes;
    size_t kvBytes;
    size_t qkv_weightBytes;

    AttentionParams params;
    uint32_t num_groups_x;
    uint32_t num_groups_y;
    uint32_t tile_m = 64;
    uint32_t tile_n = 64;
    uint32_t tile_k = 64;
    uint32_t simd_groups = 4;


    ComputeQKV(int m, int n, int d) : xBytes(m*n*sizeof(__bf16)), qBytes(m*d*sizeof(__bf16)), kvBytes(m*d*sizeof(__bf16)), qkv_weightBytes(n*d*sizeof(__bf16)), params(m, n, d) {
        pool = NS::AutoreleasePool::alloc()->init();
        device = MTL::CreateSystemDefaultDevice();
        commandQueue = device->newCommandQueue();
        MTL::Library* library = device->newDefaultLibrary();
        MTL::Function* function = library->newFunction(makeNSString("computeQKV"));

        NS::Error* error = nullptr;
        pipelineState = device->newComputePipelineState(function, &error);

        if (!pipelineState) {
            std::cerr << "pipeline: " << error->localizedDescription()->utf8String() << "\n";
            std::exit(1);
        }

        num_groups_x = d / tile_n;
        num_groups_y = m / tile_m;

        XBuffer = device->newBuffer(xBytes, MTL::ResourceStorageModeShared);
        QBuffer = device->newBuffer(qBytes, MTL::ResourceStorageModeShared);
        KBuffer = device->newBuffer(kvBytes, MTL::ResourceStorageModeShared);
        VBuffer = device->newBuffer(kvBytes, MTL::ResourceStorageModeShared);
        Q_weightBuffer = device->newBuffer(qkv_weightBytes, MTL::ResourceStorageModeShared);
        K_weightBuffer = device->newBuffer(qkv_weightBytes, MTL::ResourceStorageModeShared);
        V_weightBuffer = device->newBuffer(qkv_weightBytes, MTL::ResourceStorageModeShared);

        paramsBuffer = device->newBuffer(sizeof(AttentionParams), MTL::ResourceStorageModeShared);

        std::memcpy(paramsBuffer->contents(), &params, sizeof(AttentionParams));

        function->release();
        library->release();
    }
    ~ComputeQKV() {
        paramsBuffer->release();
        QBuffer->release();
        KBuffer->release();
        VBuffer->release();
        Q_weightBuffer->release();
        K_weightBuffer->release();
        V_weightBuffer->release();
        pipelineState->release();
        commandQueue->release();
        pool->release();
        device->release();
        }
    void run(const Matrix<__bf16>& x, Matrix<__bf16>& q, Matrix<__bf16>& k, Matrix<__bf16>& v, const Matrix<__bf16>& qw, const Matrix<__bf16>& kw, const Matrix<__bf16>& vw) {
        std::memcpy(XBuffer->contents(), x.matrix.data(), xBytes);
        std::memcpy(Q_weightBuffer->contents(), qw.matrix.data(), qkv_weightBytes);
        std::memcpy(K_weightBuffer->contents(), kw.matrix.data(), qkv_weightBytes);
        std::memcpy(V_weightBuffer->contents(), vw.matrix.data(), qkv_weightBytes);
        std::memset(QBuffer->contents(), 0, qBytes);
        std::memset(KBuffer->contents(), 0, kvBytes);
        std::memset(VBuffer->contents(), 0, kvBytes);

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

        encoder->setComputePipelineState(pipelineState);
        encoder->setBuffer(XBuffer, 0, 0);
        encoder->setBuffer(QBuffer, 0, 1);
        encoder->setBuffer(KBuffer, 0, 2);
        encoder->setBuffer(VBuffer, 0, 3);
        encoder->setBuffer(Q_weightBuffer, 0, 4);
        encoder->setBuffer(K_weightBuffer, 0, 5);
        encoder->setBuffer(V_weightBuffer, 0, 6);
        encoder->setBuffer(paramsBuffer, 0, 7);

        encoder->dispatchThreadgroups(MTL::Size(num_groups_y, num_groups_x, 1), MTL::Size(32*simd_groups, 1, 1));
        encoder->endEncoding();
        auto t0 = std::chrono::high_resolution_clock::now();
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
        if (commandBuffer->status() == MTL::CommandBufferStatusError) {
            auto* e = commandBuffer->error();
            std::cerr << "cmdbuf: " << (e ? e->localizedDescription()->utf8String() : "unknown") << "\n";
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        auto* q_out = static_cast<__bf16*>(QBuffer->contents());
        std::copy(q_out, q_out + q.matrix.size(), q.matrix.begin());
        auto* k_out = static_cast<__bf16*>(KBuffer->contents());
        std::copy(k_out, k_out + k.matrix.size(), k.matrix.begin());
        auto* v_out = static_cast<__bf16*>(VBuffer->contents());
        std::copy(v_out, v_out + v.matrix.size(), v.matrix.begin());

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
        std::cout << "Elapsed time: " << elapsed.count() << " ms" << std::endl;
    }
};

struct MetalAttentionForward {
    NS::AutoreleasePool* pool;
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::ComputePipelineState* pipelineState;

    MTL::Buffer* QBuffer;
    MTL::Buffer* KBuffer;
    MTL::Buffer* VBuffer;
    MTL::Buffer* SoftmaxBuffer;
    MTL::Buffer* OutBuffer;
    MTL::Buffer* paramsBuffer;

    size_t qBytes;
    size_t kvBytes;
    size_t softmaxBytes;
    size_t outBytes;

    AttentionParams params;
    uint32_t num_groups;
    uint32_t tile_m = 64;
    uint32_t tile_n = 64;
    uint32_t simd_groups = 4;


    MetalAttentionForward(int m, int n, int d) : qBytes(m*n*sizeof(__bf16)), kvBytes(m*n*sizeof(__bf16)), softmaxBytes(m*m*sizeof(__bf16)), outBytes(m*n*sizeof(__bf16)), params(m, n, d, n/d) {
        pool = NS::AutoreleasePool::alloc()->init();
        device = MTL::CreateSystemDefaultDevice();
        commandQueue = device->newCommandQueue();
        MTL::Library* library = device->newDefaultLibrary();
        MTL::Function* function = library->newFunction(makeNSString("attentionForward"));

        NS::Error* error = nullptr;
        pipelineState = device->newComputePipelineState(function, &error);

        if (!pipelineState) {
            std::cerr << "pipeline: " << error->localizedDescription()->utf8String() << "\n";
            std::exit(1);
        }

        num_groups = m / tile_m;

        QBuffer = device->newBuffer(qBytes, MTL::ResourceStorageModeShared);
        KBuffer = device->newBuffer(kvBytes, MTL::ResourceStorageModeShared);
        VBuffer = device->newBuffer(kvBytes, MTL::ResourceStorageModeShared);
        SoftmaxBuffer = device->newBuffer(softmaxBytes, MTL::ResourceStorageModeShared);
        OutBuffer = device->newBuffer(outBytes, MTL::ResourceStorageModeShared);

        paramsBuffer = device->newBuffer(sizeof(AttentionParams), MTL::ResourceStorageModeShared);

        std::memcpy(paramsBuffer->contents(), &params, sizeof(AttentionParams));

        function->release();
        library->release();
    }
    ~MetalAttentionForward() {
            paramsBuffer->release();
            QBuffer->release();
            KBuffer->release();
        VBuffer->release();
        SoftmaxBuffer->release();
        OutBuffer->release();
            pipelineState->release();
            commandQueue->release();
            pool->release();
            device->release();
        }
    void run(Matrix<__bf16>& q, Matrix<__bf16>& k, Matrix<__bf16>& v, Matrix<__bf16>& softmax, Matrix<__bf16>& out) {

        std::memcpy(QBuffer->contents(), q.matrix.data(), qBytes);
        std::memcpy(KBuffer->contents(), k.matrix.data(), kvBytes);
        std::memcpy(VBuffer->contents(), v.matrix.data(), kvBytes);

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

        encoder->setComputePipelineState(pipelineState);
        encoder->setThreadgroupMemoryLength(tile_m * params.D * sizeof(float), 0); //Accumulator
        encoder->setThreadgroupMemoryLength(tile_n*tile_m*sizeof(__bf16), 1);//softmax_out tile buffer
        encoder->setThreadgroupMemoryLength(tile_m * sizeof(float), 2);
        encoder->setThreadgroupMemoryLength(tile_m * sizeof(float), 3);
        encoder->setBuffer(QBuffer, 0, 0);
        encoder->setBuffer(KBuffer, 0, 1);
        encoder->setBuffer(VBuffer, 0, 2);
        encoder->setBuffer(SoftmaxBuffer, 0, 3);
        encoder->setBuffer(OutBuffer, 0, 4);
        encoder->setBuffer(paramsBuffer, 0, 5);

        encoder->dispatchThreadgroups(MTL::Size(num_groups * params.NH, 1, 1), MTL::Size(32*simd_groups, 1, 1));
        encoder->endEncoding();
        auto t0 = std::chrono::high_resolution_clock::now();
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
        if (commandBuffer->status() == MTL::CommandBufferStatusError) {
            auto* e = commandBuffer->error();
            std::cerr << "cmdbuf: " << (e ? e->localizedDescription()->utf8String() : "unknown") << "\n";
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        auto* gpuOut = static_cast<__bf16*>(OutBuffer->contents());
        std::copy(gpuOut, gpuOut + out.matrix.size(), out.matrix.begin());

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
        std::cout << "Elapsed time: " << elapsed.count() << " ms" << std::endl;
    }
};

int main() {
    Matrix<__bf16> x(4096,4096);
    Matrix<__bf16> q(4096, 4096);
    Matrix<__bf16> k(4096,4096);
    Matrix<__bf16> v(4096,4096);
    Matrix<__bf16> qw(4096,4096);
    Matrix<__bf16> kw(4096,4096);
    Matrix<__bf16> vw(4096,4096);
    Matrix<__bf16> softmax_out(4096,4096);
    Matrix<__bf16> out(4096,4096);
    std::fill(x.matrix.begin(), x.matrix.end(), 1);
    std::fill(qw.matrix.begin(), qw.matrix.end(), 1);
    std::fill(kw.matrix.begin(), kw.matrix.end(), 1);
    std::fill(vw.matrix.begin(), vw.matrix.end(), 1);
    ComputeQKV c(4096,4096,4096);
    MetalAttentionForward attn(4096,4096, 64);

    c.run(x, q, k,v,qw,kw,vw);
    auto t0 = std::chrono::high_resolution_clock::now();
    attn.run(q,k,v,softmax_out,out);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Elapsed time: " << ms << " ms\n";
    std::cout << "Throughput: " << 275.0 / (ms / 1000.0) << " GFLOP/s\n";
    return 0;
}