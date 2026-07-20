//One host struct for the whole net: forward() encodes embed -> every block ->
//ln_final -> output projection -> loss into ONE command buffer, backward()
//encodes the mirror image (loss -> ... -> embed grads) into ONE command
//buffer, and step() is the fused Adam update. Residual adds and the
//float->bfloat handoff between attention backward and ln1 backward run on the
//GPU via the residualAdd/castFloatToBfloat kernels in elementwise.metal, so
//nothing breaks the pass into pieces. Activations live in per-block buffers
//between forward and backward. Weights and Adam state are GPU-resident:
//uploaded once at construction, gradients gathered and weights updated
//entirely on the GPU by step() (gatherGrad*/MetalStep/scatterWeight), so the
//host only uploads the batch each iteration and reads back the loss.
//
//Depends on Matrix, the kernel param structs, ModelWeights/ModelGrads, and the
//N_LAYERS/LR constants from main.cpp, so include this AFTER those definitions.
#pragma once

struct Model {
    struct LayernormBuffers {
        MTL::Buffer* gammaBuffer;
        MTL::Buffer* betaBuffer;
        MTL::Buffer* xnormBuffer;
        MTL::Buffer* stdevBuffer; //float
        MTL::Buffer* outBuffer;
        MTL::Buffer* dGammaBuffer; //float atomic accumulators
        MTL::Buffer* dBetaBuffer;
    };
    struct BlockBuffers {
        LayernormBuffers ln1;
        //attention weights + forward caches
        MTL::Buffer* qwBuffer;
        MTL::Buffer* kwBuffer;
        MTL::Buffer* vwBuffer;
        MTL::Buffer* projwBuffer;
        MTL::Buffer* qBuffer; //float
        MTL::Buffer* kBuffer;
        MTL::Buffer* vBuffer;
        MTL::Buffer* attnOutBuffer; //pre-projection
        MTL::Buffer* lBuffer; //flash attention row stats, float
        MTL::Buffer* projOutBuffer;
        MTL::Buffer* residual1Buffer;
        LayernormBuffers ln2;
        //mlp weights + forward caches
        MTL::Buffer* upprojBuffer;
        MTL::Buffer* downprojBuffer;
        MTL::Buffer* mlpVBuffer;
        MTL::Buffer* mlpMaskBuffer;
        MTL::Buffer* mlpOutBuffer;
        MTL::Buffer* residual2Buffer;
        //attention backward scratch (accumulated into, so per block)
        MTL::Buffer* dQBuffer; //float
        MTL::Buffer* dKBuffer;
        MTL::Buffer* dVBuffer;
        MTL::Buffer* dXAttnBuffer; //float
        //gradient outputs read by the host after backward
        MTL::Buffer* dQwBuffer; //float
        MTL::Buffer* dKwBuffer;
        MTL::Buffer* dVwBuffer;
        MTL::Buffer* dProjwBuffer;
        MTL::Buffer* dUpBuffer;
        MTL::Buffer* dDpBuffer;
    };

    NS::AutoreleasePool* pool;
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;

    //one pipeline per kernel, shared by every block
    MTL::ComputePipelineState* embedFwdPipeline;
    MTL::ComputePipelineState* embedBwdPipeline;
    MTL::ComputePipelineState* layernormFwdPipeline;
    MTL::ComputePipelineState* layernormBwdPipeline;
    MTL::ComputePipelineState* qkvPipeline;
    MTL::ComputePipelineState* flashAttnFwdPipeline;
    MTL::ComputePipelineState* computeDPipeline;
    MTL::ComputePipelineState* flashAttnBwdPipeline;
    MTL::ComputePipelineState* attnGradDxPipeline;
    MTL::ComputePipelineState* attnGradDwPipeline;
    MTL::ComputePipelineState* projFwdPipeline;
    MTL::ComputePipelineState* projBwdDwPipeline;
    MTL::ComputePipelineState* projBwdDxPipeline;
    MTL::ComputePipelineState* mlpFwdOnePipeline;
    MTL::ComputePipelineState* mlpFwdTwoPipeline;
    MTL::ComputePipelineState* mlpBwdOnePipeline;
    MTL::ComputePipelineState* mlpBwdTwoPipeline;
    MTL::ComputePipelineState* linearFwdPipeline;
    MTL::ComputePipelineState* linearBwdPipeline;
    MTL::ComputePipelineState* lossFwdPipeline;
    MTL::ComputePipelineState* lossBwdPipeline;
    MTL::ComputePipelineState* residualAddPipeline;
    MTL::ComputePipelineState* castPipeline;
    MTL::ComputePipelineState* stepPipeline;
    MTL::ComputePipelineState* gatherFloatPipeline;
    MTL::ComputePipelineState* gatherBfloatPipeline;
    MTL::ComputePipelineState* scatterPipeline;

    //dimensions
    uint32_t blockSize;
    uint32_t vocabSize; //padded
    uint32_t embedDim;
    uint32_t mlpScale;
    uint32_t headSize;
    uint32_t numHeads;
    uint32_t hiddenDim; //embedDim * mlpScale
    uint32_t tileRows;  //blockSize / 64
    uint32_t tileCols;  //embedDim / 64
    uint32_t activationCount; //blockSize * embedDim
    static constexpr uint32_t tileM = 64;
    static constexpr uint32_t tileN = 64;
    static constexpr uint32_t simdGroups = 4;

    //kernel params, one of each shared by every block
    EmbedParams embedParams;
    LayernormParams layernormParams;
    AttentionParams attnParams;
    mlpParams mlpParameters;
    MatMulParams linearParams;
    LossParams lossParams;

    //common byte sizes
    size_t activationBytes; //blockSize * embedDim * bf16
    size_t activationFloatBytes;
    size_t attnWeightBytes; //embedDim * embedDim * bf16
    size_t hiddenBytes;     //blockSize * hiddenDim * bf16
    size_t gammaFloatBytes; //embedDim * float

    //embed
    MTL::Buffer* inputTokensBuffer;
    MTL::Buffer* embedWeightBuffer;
    MTL::Buffer* posWeightBuffer;
    MTL::Buffer* embedOutBuffer;
    MTL::Buffer* dEmbedWeightBuffer; //float atomic accumulators

    //blocks
    std::vector<BlockBuffers> blocks;

    //final layernorm / output projection / loss
    LayernormBuffers lnFinal;
    MTL::Buffer* linearWBuffer;
    MTL::Buffer* logitsBuffer;
    MTL::Buffer* dLogitsBuffer;
    MTL::Buffer* dLinearWBuffer;
    MTL::Buffer* targetsBuffer;
    MTL::Buffer* lossOutBuffer; //float
    MTL::Buffer* lseBuffer;     //float

    //backward flow, shared by every block since the encoders run in order
    MTL::Buffer* dStreamBuffer; //gradient flowing down the residual stream
    MTL::Buffer* dBranchBuffer; //branch gradient inside a block
    MTL::Buffer* dXnormBuffer;  //layernorm backward scratch
    MTL::Buffer* dZAttnBuffer;  //post-projection attention gradient
    MTL::Buffer* dAttnDBuffer;  //computeD output
    MTL::Buffer* dMlpVBuffer;

    //optimizer: weights and Adam state are GPU-resident. The host vectors in
    //ModelWeights are read once at construction (initial upload + master
    //build) and never touched again; per-parameter bookkeeping below drives
    //the gather (grads -> flat) and scatter (master -> weight buffers) passes
    //that bracket the Adam kernel inside step()'s command buffer.
    struct FlatCopyParams { uint32_t count; uint32_t offset; };
    std::vector<std::vector<__bf16>*> optWeights; //host init values, used for sizes + one-time upload
    std::vector<MTL::Buffer*> optWeightBuffers;   //bf16 buffers the forward kernels read
    std::vector<MTL::Buffer*> optGradBuffers;     //buffers backward writes each parameter's grad into
    std::vector<bool> optGradIsFloat;             //float atomic accumulators vs bf16
    std::vector<size_t> optOffsets;
    size_t optCount = 0;
    float stepCounter = 0.0f;
    MTL::Buffer* parameterBuffer;
    MTL::Buffer* gradientBuffer;
    MTL::Buffer* momentumBuffer;
    MTL::Buffer* varianceBuffer;

    Model(int m, int v, int e, int s, int d, ModelWeights& weights, ModelGrads& grads)
        : blockSize(m), vocabSize(v), embedDim(e), mlpScale(s), headSize(d),
          numHeads(e / d), hiddenDim(e * s), tileRows(m / tileM), tileCols(e / tileN),
          activationCount((uint32_t)(m * e)),
          embedParams{(uint32_t)m, (uint32_t)e, (uint32_t)std::min(e, 1024)},
          layernormParams(m, e),
          attnParams{(uint)m, (uint)e, (uint)d, (uint)(e / d)},
          mlpParameters{(uint)m, (uint)e, (uint)s},
          linearParams(m, v, e),
          lossParams{(uint32_t)m, (uint32_t)v, (uint32_t)std::min(v, 1024)},
          activationBytes((size_t)m * e * sizeof(__bf16)),
          activationFloatBytes((size_t)m * e * sizeof(float)),
          attnWeightBytes((size_t)e * e * sizeof(__bf16)),
          hiddenBytes((size_t)m * e * s * sizeof(__bf16)),
          gammaFloatBytes((size_t)e * sizeof(float)) {
        pool = NS::AutoreleasePool::alloc()->init();
        device = MTL::CreateSystemDefaultDevice();
        commandQueue = device->newCommandQueue();
        MTL::Library* library = device->newDefaultLibrary();

        embedFwdPipeline = makePipeline(library, "embedForward");
        embedBwdPipeline = makePipeline(library, "embedBackward");
        layernormFwdPipeline = makePipeline(library, "layernormForward");
        layernormBwdPipeline = makePipeline(library, "layernormBackward");
        qkvPipeline = makePipeline(library, "computeQKV");
        flashAttnFwdPipeline = makePipeline(library, "flashAttentionForward");
        computeDPipeline = makePipeline(library, "computeD");
        flashAttnBwdPipeline = makePipeline(library, "flashAttentionBackward");
        attnGradDxPipeline = makePipeline(library, "computeGradDx");
        attnGradDwPipeline = makePipeline(library, "computeGradDw");
        projFwdPipeline = makePipeline(library, "projForward");
        projBwdDwPipeline = makePipeline(library, "projBackwarddW");
        projBwdDxPipeline = makePipeline(library, "projBackwarddX");
        mlpFwdOnePipeline = makePipeline(library, "mlpForwardOne");
        mlpFwdTwoPipeline = makePipeline(library, "mlpForwardTwo");
        mlpBwdOnePipeline = makePipeline(library, "mlpBackwardOne");
        mlpBwdTwoPipeline = makePipeline(library, "mlpBackwardTwo");
        linearFwdPipeline = makePipeline(library, "linearForward");
        linearBwdPipeline = makePipeline(library, "linearBackward");
        lossFwdPipeline = makePipeline(library, "lossForward");
        lossBwdPipeline = makePipeline(library, "lossBackward");
        residualAddPipeline = makePipeline(library, "residualAdd");
        castPipeline = makePipeline(library, "castFloatToBfloat");
        stepPipeline = makePipeline(library, "MetalStep");
        gatherFloatPipeline = makePipeline(library, "gatherGradFloat");
        gatherBfloatPipeline = makePipeline(library, "gatherGradBfloat");
        scatterPipeline = makePipeline(library, "scatterWeight");
        library->release();

        //embed
        inputTokensBuffer = newSharedBuffer((size_t)blockSize * sizeof(uint32_t));
        embedWeightBuffer = newSharedBuffer((size_t)vocabSize * embedDim * sizeof(__bf16));
        posWeightBuffer = newSharedBuffer(activationBytes);
        embedOutBuffer = newSharedBuffer(activationBytes);
        dEmbedWeightBuffer = newSharedBuffer((size_t)vocabSize * embedDim * sizeof(float));

        //blocks
        blocks.resize(N_LAYERS);
        for (BlockBuffers& block : blocks) {
            allocLayernorm(block.ln1);
            block.qwBuffer = newSharedBuffer(attnWeightBytes);
            block.kwBuffer = newSharedBuffer(attnWeightBytes);
            block.vwBuffer = newSharedBuffer(attnWeightBytes);
            block.projwBuffer = newSharedBuffer(attnWeightBytes);
            block.qBuffer = newSharedBuffer(activationFloatBytes);
            block.kBuffer = newSharedBuffer(activationBytes);
            block.vBuffer = newSharedBuffer(activationBytes);
            block.attnOutBuffer = newSharedBuffer(activationBytes);
            block.lBuffer = newSharedBuffer((size_t)blockSize * numHeads * sizeof(float));
            block.projOutBuffer = newSharedBuffer(activationBytes);
            block.residual1Buffer = newSharedBuffer(activationBytes);
            allocLayernorm(block.ln2);
            block.upprojBuffer = newSharedBuffer((size_t)embedDim * hiddenDim * sizeof(__bf16));
            block.downprojBuffer = newSharedBuffer((size_t)hiddenDim * embedDim * sizeof(__bf16));
            block.mlpVBuffer = newSharedBuffer(hiddenBytes);
            block.mlpMaskBuffer = newSharedBuffer((size_t)blockSize * hiddenDim * sizeof(uint8_t));
            block.mlpOutBuffer = newSharedBuffer(activationBytes);
            block.residual2Buffer = newSharedBuffer(activationBytes);
            block.dQBuffer = newSharedBuffer(activationFloatBytes);
            block.dKBuffer = newSharedBuffer(activationBytes);
            block.dVBuffer = newSharedBuffer(activationBytes);
            block.dXAttnBuffer = newSharedBuffer(activationFloatBytes);
            block.dQwBuffer = newSharedBuffer((size_t)embedDim * embedDim * sizeof(float));
            block.dKwBuffer = newSharedBuffer(attnWeightBytes);
            block.dVwBuffer = newSharedBuffer(attnWeightBytes);
            block.dProjwBuffer = newSharedBuffer(attnWeightBytes);
            block.dUpBuffer = newSharedBuffer((size_t)embedDim * hiddenDim * sizeof(__bf16));
            block.dDpBuffer = newSharedBuffer((size_t)hiddenDim * embedDim * sizeof(__bf16));
        }

        //final layernorm / output projection / loss
        allocLayernorm(lnFinal);
        linearWBuffer = newSharedBuffer((size_t)embedDim * vocabSize * sizeof(__bf16));
        logitsBuffer = newSharedBuffer((size_t)blockSize * vocabSize * sizeof(__bf16));
        dLogitsBuffer = newSharedBuffer((size_t)blockSize * vocabSize * sizeof(__bf16));
        dLinearWBuffer = newSharedBuffer((size_t)embedDim * vocabSize * sizeof(__bf16));
        targetsBuffer = newSharedBuffer((size_t)blockSize * sizeof(uint32_t));
        lossOutBuffer = newSharedBuffer((size_t)blockSize * sizeof(float));
        lseBuffer = newSharedBuffer((size_t)blockSize * sizeof(float));

        //backward flow
        dStreamBuffer = newSharedBuffer(activationBytes);
        dBranchBuffer = newSharedBuffer(activationBytes);
        dXnormBuffer = newSharedBuffer(activationBytes);
        dZAttnBuffer = newSharedBuffer(activationBytes);
        dAttnDBuffer = newSharedBuffer((size_t)blockSize * numHeads * sizeof(__bf16));
        dMlpVBuffer = newSharedBuffer(hiddenBytes);

        //optimizer: register every parameter with its GPU weight/grad buffer,
        //then build the flat master copy and Adam state once. Registration
        //order defines the flat layout, nothing else depends on it.
        addParam(weights.embed_weights, embedWeightBuffer, dEmbedWeightBuffer, true);
        addParam(weights.pos_weights, posWeightBuffer, dStreamBuffer, false); //dStream holds the pos grad after backward
        for (int n = 0; n < N_LAYERS; ++n) {
            BlockBuffers& block = blocks[n];
            addParam(weights.blocks[n].ln1_gamma, block.ln1.gammaBuffer, block.ln1.dGammaBuffer, true);
            addParam(weights.blocks[n].ln1_beta, block.ln1.betaBuffer, block.ln1.dBetaBuffer, true);
            addParam(weights.blocks[n].qw, block.qwBuffer, block.dQwBuffer, true);
            addParam(weights.blocks[n].kw, block.kwBuffer, block.dKwBuffer, false);
            addParam(weights.blocks[n].vw, block.vwBuffer, block.dVwBuffer, false);
            addParam(weights.blocks[n].proj_weights, block.projwBuffer, block.dProjwBuffer, false);
            addParam(weights.blocks[n].ln2_gamma, block.ln2.gammaBuffer, block.ln2.dGammaBuffer, true);
            addParam(weights.blocks[n].ln2_beta, block.ln2.betaBuffer, block.ln2.dBetaBuffer, true);
            addParam(weights.blocks[n].mlp_up, block.upprojBuffer, block.dUpBuffer, false);
            addParam(weights.blocks[n].mlp_down, block.downprojBuffer, block.dDpBuffer, false);
        }
        addParam(weights.ln_final_gamma, lnFinal.gammaBuffer, lnFinal.dGammaBuffer, true);
        addParam(weights.ln_final_beta, lnFinal.betaBuffer, lnFinal.dBetaBuffer, true);
        addParam(weights.output_proj, linearWBuffer, dLinearWBuffer, false);

        size_t total = 0;
        for (std::vector<__bf16>* weight : optWeights) {
            optOffsets.push_back(total);
            total += weight->size();
        }
        optCount = ((total + 16383) / 16384) * 16384;
        parameterBuffer = newSharedBuffer(optCount * sizeof(float));
        gradientBuffer = newSharedBuffer(optCount * sizeof(__bf16));
        momentumBuffer = newSharedBuffer(optCount * sizeof(__bf16));
        varianceBuffer = newSharedBuffer(optCount * sizeof(__bf16));

        //one-time upload: master floats + Adam state + the bf16 weight buffers.
        //the padded tail of the gradient buffer stays zero forever, so the
        //step kernel leaves the padded master/momentum/variance tail alone.
        auto* masterOut = static_cast<float*>(parameterBuffer->contents());
        std::fill(masterOut, masterOut + optCount, 0.0f);
        for (size_t i = 0; i < optWeights.size(); ++i) {
            std::transform(optWeights[i]->begin(), optWeights[i]->end(), masterOut + optOffsets[i],
                           [](__bf16 x) { return (float)x; });
            std::memcpy(optWeightBuffers[i]->contents(), optWeights[i]->data(), optWeights[i]->size() * sizeof(__bf16));
        }
        std::memset(gradientBuffer->contents(), 0, optCount * sizeof(__bf16));
        std::memset(momentumBuffer->contents(), 0, optCount * sizeof(__bf16));
        std::memset(varianceBuffer->contents(), 0, optCount * sizeof(__bf16));
    }

    //---------------------------------------------------------------- helpers
    MTL::ComputePipelineState* makePipeline(MTL::Library* library, const char* name) {
        MTL::Function* function = library->newFunction(makeNSString(name));
        NS::Error* error = nullptr;
        MTL::ComputePipelineState* pipeline = device->newComputePipelineState(function, &error);
        if (!pipeline) {
            std::cerr << name << " pipeline: " << error->localizedDescription()->utf8String() << "\n";
            std::exit(1);
        }
        function->release();
        return pipeline;
    }
    MTL::Buffer* newSharedBuffer(size_t bytes) {
        return device->newBuffer(bytes, MTL::ResourceStorageModeShared);
    }
    void allocLayernorm(LayernormBuffers& ln) {
        ln.gammaBuffer = newSharedBuffer((size_t)embedDim * sizeof(__bf16));
        ln.betaBuffer = newSharedBuffer((size_t)embedDim * sizeof(__bf16));
        ln.xnormBuffer = newSharedBuffer(activationBytes);
        ln.stdevBuffer = newSharedBuffer((size_t)blockSize * sizeof(float));
        ln.outBuffer = newSharedBuffer(activationBytes);
        ln.dGammaBuffer = newSharedBuffer(gammaFloatBytes);
        ln.dBetaBuffer = newSharedBuffer(gammaFloatBytes);
    }
    MTL::ComputeCommandEncoder* makeEncoder(MTL::CommandBuffer* commandBuffer,
                                            MTL::ComputePipelineState* pipeline) {
        MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
        encoder->setComputePipelineState(pipeline);
        return encoder;
    }
    static void commitWait(MTL::CommandBuffer* commandBuffer, const char* stage) {
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
        if (commandBuffer->status() == MTL::CommandBufferStatusError) {
            auto* error = commandBuffer->error();
            std::cerr << stage << " cmdbuf: "
                      << (error ? error->localizedDescription()->utf8String() : "unknown") << "\n";
        }
    }
    //out = a + b, elementwise over one activation-sized tensor
    void encodeResidualAdd(MTL::CommandBuffer* commandBuffer, MTL::Buffer* a, MTL::Buffer* b, MTL::Buffer* out) {
        MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, residualAddPipeline);
        encoder->setBuffer(a, 0, 0);
        encoder->setBuffer(b, 0, 1);
        encoder->setBuffer(out, 0, 2);
        encoder->setBytes(&activationCount, sizeof(activationCount), 3);
        encoder->dispatchThreadgroups(MTL::Size(activationCount / 1024, 1, 1), MTL::Size(1024, 1, 1));
        encoder->endEncoding();
    }
    void encodeCast(MTL::CommandBuffer* commandBuffer, MTL::Buffer* floatIn, MTL::Buffer* bfloatOut) {
        MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, castPipeline);
        encoder->setBuffer(floatIn, 0, 0);
        encoder->setBuffer(bfloatOut, 0, 1);
        encoder->setBytes(&activationCount, sizeof(activationCount), 2);
        encoder->dispatchThreadgroups(MTL::Size(activationCount / 1024, 1, 1), MTL::Size(1024, 1, 1));
        encoder->endEncoding();
    }
    void encodeLayernormForward(MTL::CommandBuffer* commandBuffer, LayernormBuffers& ln, MTL::Buffer* inputBuffer) {
        MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, layernormFwdPipeline);
        encoder->setBuffer(inputBuffer, 0, 0);
        encoder->setBuffer(ln.xnormBuffer, 0, 1);
        encoder->setBuffer(ln.outBuffer, 0, 2);
        encoder->setBuffer(ln.gammaBuffer, 0, 3);
        encoder->setBuffer(ln.betaBuffer, 0, 4);
        encoder->setBuffer(ln.stdevBuffer, 0, 5);
        encoder->setBytes(&layernormParams, sizeof(layernormParams), 6);
        encoder->dispatchThreadgroups(MTL::Size(layernormParams.rows, 1, 1), MTL::Size(layernormParams.group_size, 1, 1));
        encoder->endEncoding();
    }
    //dZBuffer is rewritten in place with dX
    void encodeLayernormBackward(MTL::CommandBuffer* commandBuffer, LayernormBuffers& ln, MTL::Buffer* dZBuffer) {
        MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, layernormBwdPipeline);
        encoder->setBuffer(dZBuffer, 0, 0);
        encoder->setBuffer(ln.xnormBuffer, 0, 1);
        encoder->setBuffer(ln.gammaBuffer, 0, 2);
        encoder->setBuffer(ln.stdevBuffer, 0, 3);
        encoder->setBuffer(dXnormBuffer, 0, 5);
        encoder->setBuffer(ln.dGammaBuffer, 0, 6);
        encoder->setBuffer(ln.dBetaBuffer, 0, 7);
        encoder->setBytes(&layernormParams, sizeof(layernormParams), 8);
        encoder->dispatchThreadgroups(MTL::Size(layernormParams.rows, 1, 1), MTL::Size(layernormParams.group_size, 1, 1));
        encoder->endEncoding();
    }
    //---------------------------------------------------------------- forward
    //weights are GPU-resident (uploaded once at construction, updated in
    //place by step()), so only the batch crosses the bus here
    float forward(const std::vector<uint32_t>& inputTokens, const Matrix<uint32_t>& targets,
                  const ModelWeights&) {
        std::memcpy(inputTokensBuffer->contents(), inputTokens.data(), (size_t)blockSize * sizeof(uint32_t));
        std::memcpy(targetsBuffer->contents(), targets.matrix.data(), (size_t)blockSize * sizeof(uint32_t));
        for (int n = 0; n < N_LAYERS; ++n) {
            BlockBuffers& block = blocks[n];
            //the qkv kernel accumulates, so its outputs start zeroed
            std::memset(block.qBuffer->contents(), 0, activationFloatBytes);
            std::memset(block.kBuffer->contents(), 0, activationBytes);
            std::memset(block.vBuffer->contents(), 0, activationBytes);
        }

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();

        //embedding + positional embedding
        {
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, embedFwdPipeline);
            encoder->setBuffer(inputTokensBuffer, 0, 0);
            encoder->setBuffer(embedWeightBuffer, 0, 1);
            encoder->setBuffer(embedOutBuffer, 0, 2);
            encoder->setBytes(&embedParams, sizeof(embedParams), 3);
            encoder->dispatchThreadgroups(MTL::Size(embedParams.t, 1, 1), MTL::Size(embedParams.group_size, 1, 1));
            encoder->endEncoding();
        }
        encodeResidualAdd(commandBuffer, embedOutBuffer, posWeightBuffer, embedOutBuffer);

        MTL::Buffer* hiddenBuffer = embedOutBuffer;
        for (int n = 0; n < N_LAYERS; ++n) {
            BlockBuffers& block = blocks[n];

            //ln1 -> qkv -> flash attention -> projection -> residual
            encodeLayernormForward(commandBuffer, block.ln1, hiddenBuffer);
            {
                MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, qkvPipeline);
                encoder->setBuffer(block.ln1.outBuffer, 0, 0);
                encoder->setBuffer(block.qBuffer, 0, 1);
                encoder->setBuffer(block.kBuffer, 0, 2);
                encoder->setBuffer(block.vBuffer, 0, 3);
                encoder->setBuffer(block.qwBuffer, 0, 4);
                encoder->setBuffer(block.kwBuffer, 0, 5);
                encoder->setBuffer(block.vwBuffer, 0, 6);
                encoder->setBytes(&attnParams, sizeof(attnParams), 7);
                encoder->dispatchThreadgroups(MTL::Size(tileRows, tileCols, 1), MTL::Size(32 * simdGroups, 1, 1));
                encoder->endEncoding();
            }
            {
                MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, flashAttnFwdPipeline);
                encoder->setThreadgroupMemoryLength(tileM * headSize * sizeof(float), 0); //accumulator
                encoder->setThreadgroupMemoryLength(tileN * tileM * sizeof(__bf16), 1); //softmax tile
                encoder->setThreadgroupMemoryLength(tileM * sizeof(float), 2);
                encoder->setThreadgroupMemoryLength(tileM * sizeof(float), 3);
                encoder->setBuffer(block.qBuffer, 0, 0);
                encoder->setBuffer(block.kBuffer, 0, 1);
                encoder->setBuffer(block.vBuffer, 0, 2);
                encoder->setBuffer(block.attnOutBuffer, 0, 3);
                encoder->setBuffer(block.lBuffer, 0, 4);
                encoder->setBytes(&attnParams, sizeof(attnParams), 5);
                encoder->dispatchThreadgroups(MTL::Size((size_t)tileRows * numHeads, 1, 1), MTL::Size(32 * simdGroups, 1, 1));
                encoder->endEncoding();
            }
            {
                MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, projFwdPipeline);
                encoder->setBuffer(block.attnOutBuffer, 0, 0);
                encoder->setBuffer(block.projwBuffer, 0, 1);
                encoder->setBuffer(block.projOutBuffer, 0, 2);
                encoder->setBytes(&attnParams, sizeof(attnParams), 3);
                encoder->dispatchThreadgroups(MTL::Size(tileRows, tileCols, 1), MTL::Size(32 * simdGroups, 1, 1));
                encoder->endEncoding();
            }
            encodeResidualAdd(commandBuffer, hiddenBuffer, block.projOutBuffer, block.residual1Buffer);

            //ln2 -> mlp (stage 1: V = relu(X @ Up), stage 2: out = V @ Dp) -> residual
            encodeLayernormForward(commandBuffer, block.ln2, block.residual1Buffer);
            {
                MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, mlpFwdOnePipeline);
                encoder->setBuffer(block.ln2.outBuffer, 0, 0);
                encoder->setBuffer(block.upprojBuffer, 0, 1);
                encoder->setBuffer(block.mlpVBuffer, 0, 2);
                encoder->setBuffer(block.mlpMaskBuffer, 0, 3);
                encoder->setBytes(&mlpParameters, sizeof(mlpParameters), 4);
                encoder->dispatchThreadgroups(MTL::Size(tileRows, hiddenDim / tileN, 1), MTL::Size(32 * simdGroups, 1, 1));
                encoder->endEncoding();
            }
            {
                MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, mlpFwdTwoPipeline);
                encoder->setBuffer(block.downprojBuffer, 0, 0);
                encoder->setBuffer(block.mlpVBuffer, 0, 1);
                encoder->setBuffer(block.mlpOutBuffer, 0, 2);
                encoder->setBytes(&mlpParameters, sizeof(mlpParameters), 3);
                encoder->dispatchThreadgroups(MTL::Size(tileRows, tileCols, 1), MTL::Size(32 * simdGroups, 1, 1));
                encoder->endEncoding();
            }
            encodeResidualAdd(commandBuffer, block.residual1Buffer, block.mlpOutBuffer, block.residual2Buffer);
            hiddenBuffer = block.residual2Buffer;
        }

        //final layernorm -> output projection -> loss
        encodeLayernormForward(commandBuffer, lnFinal, hiddenBuffer);
        {
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, linearFwdPipeline);
            encoder->setBuffer(lnFinal.outBuffer, 0, 0);
            encoder->setBuffer(linearWBuffer, 0, 1);
            encoder->setBuffer(logitsBuffer, 0, 2);
            encoder->setBytes(&linearParams, sizeof(linearParams), 3);
            encoder->dispatchThreadgroups(MTL::Size(tileRows, vocabSize / tileN, 1), MTL::Size(32 * simdGroups, 1, 1));
            encoder->endEncoding();
        }
        {
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, lossFwdPipeline);
            encoder->setBuffer(logitsBuffer, 0, 0);
            encoder->setBuffer(targetsBuffer, 0, 1);
            encoder->setBuffer(lossOutBuffer, 0, 2);
            encoder->setBuffer(lseBuffer, 0, 3);
            encoder->setBytes(&lossParams, sizeof(lossParams), 4);
            encoder->dispatchThreadgroups(MTL::Size(lossParams.rows, 1, 1), MTL::Size(lossParams.group_size, 1, 1));
            encoder->endEncoding();
        }
        commitWait(commandBuffer, "forward");

        auto* losses = static_cast<float*>(lossOutBuffer->contents());
        return std::accumulate(losses, losses + blockSize, 0.0f) / (float)blockSize;
    }

    //--------------------------------------------------------------- backward
    void backward(ModelGrads&) {
        //zero every accumulated gradient before the pass
        std::memset(dEmbedWeightBuffer->contents(), 0, (size_t)vocabSize * embedDim * sizeof(float));
        std::memset(dLinearWBuffer->contents(), 0, (size_t)embedDim * vocabSize * sizeof(__bf16));
        std::memset(lnFinal.dGammaBuffer->contents(), 0, gammaFloatBytes);
        std::memset(lnFinal.dBetaBuffer->contents(), 0, gammaFloatBytes);
        for (BlockBuffers& block : blocks) {
            std::memset(block.ln1.dGammaBuffer->contents(), 0, gammaFloatBytes);
            std::memset(block.ln1.dBetaBuffer->contents(), 0, gammaFloatBytes);
            std::memset(block.ln2.dGammaBuffer->contents(), 0, gammaFloatBytes);
            std::memset(block.ln2.dBetaBuffer->contents(), 0, gammaFloatBytes);
            std::memset(block.dQBuffer->contents(), 0, activationFloatBytes);
            std::memset(block.dKBuffer->contents(), 0, activationBytes);
            std::memset(block.dVBuffer->contents(), 0, activationBytes);
            std::memset(block.dXAttnBuffer->contents(), 0, activationFloatBytes);
            std::memset(block.dQwBuffer->contents(), 0, (size_t)embedDim * embedDim * sizeof(float));
            std::memset(block.dKwBuffer->contents(), 0, attnWeightBytes);
            std::memset(block.dVwBuffer->contents(), 0, attnWeightBytes);
            std::memset(block.dProjwBuffer->contents(), 0, attnWeightBytes);
            std::memset(block.dUpBuffer->contents(), 0, (size_t)embedDim * hiddenDim * sizeof(__bf16));
            std::memset(block.dDpBuffer->contents(), 0, (size_t)hiddenDim * embedDim * sizeof(__bf16));
        }

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();

        //loss -> output projection -> final layernorm
        {
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, lossBwdPipeline);
            encoder->setBuffer(logitsBuffer, 0, 0);
            encoder->setBuffer(targetsBuffer, 0, 1);
            encoder->setBuffer(lseBuffer, 0, 2);
            encoder->setBuffer(dLogitsBuffer, 0, 3);
            encoder->setBytes(&lossParams, sizeof(lossParams), 4);
            encoder->dispatchThreadgroups(MTL::Size(lossParams.rows, 1, 1), MTL::Size(lossParams.group_size, 1, 1));
            encoder->endEncoding();
        }
        {
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, linearBwdPipeline);
            encoder->setBuffer(lnFinal.outBuffer, 0, 0);
            encoder->setBuffer(linearWBuffer, 0, 1);
            encoder->setBuffer(dLogitsBuffer, 0, 2);
            encoder->setBuffer(dStreamBuffer, 0, 3);
            encoder->setBuffer(dLinearWBuffer, 0, 4);
            encoder->setBytes(&linearParams, sizeof(linearParams), 5);
            encoder->dispatchThreadgroups(MTL::Size(tileRows, 1, 1), MTL::Size(32 * simdGroups, 1, 1));
            encoder->endEncoding();
        }
        encodeLayernormBackward(commandBuffer, lnFinal, dStreamBuffer);

        for (int n = N_LAYERS - 1; n >= 0; --n) {
            BlockBuffers& block = blocks[n];

            //mlp backward: dStream is dZ; stage 1 makes the masked dV, stage 2
            //produces dUp/dDp and lands the branch gradient dX in dBranch
            {
                MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, mlpBwdOnePipeline);
                encoder->setBuffer(block.downprojBuffer, 0, 1);
                encoder->setBuffer(block.mlpMaskBuffer, 0, 2);
                encoder->setBuffer(dStreamBuffer, 0, 3);
                encoder->setBuffer(dMlpVBuffer, 0, 5);
                encoder->setBytes(&mlpParameters, sizeof(mlpParameters), 6);
                encoder->dispatchThreadgroups(MTL::Size(tileRows, hiddenDim / tileN, 1), MTL::Size(32 * simdGroups, 1, 1));
                encoder->endEncoding();
            }
            {
                MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, mlpBwdTwoPipeline);
                encoder->setBuffer(block.ln2.outBuffer, 0, 0);
                encoder->setBuffer(dBranchBuffer, 0, 1);
                encoder->setBuffer(dMlpVBuffer, 0, 2);
                encoder->setBuffer(block.upprojBuffer, 0, 3);
                encoder->setBuffer(block.dUpBuffer, 0, 4);
                encoder->setBuffer(block.mlpVBuffer, 0, 5);
                encoder->setBuffer(dStreamBuffer, 0, 6);
                encoder->setBuffer(block.dDpBuffer, 0, 7);
                encoder->setBytes(&mlpParameters, sizeof(mlpParameters), 8);
                //covers both tilings inside the kernel: hidden strips for dUp/dDp, M x N tiles for dX
                encoder->dispatchThreadgroups(MTL::Size(std::max(hiddenDim / tileM, tileRows * tileCols), 1, 1), MTL::Size(32 * simdGroups, 1, 1));
                encoder->endEncoding();
            }
            encodeLayernormBackward(commandBuffer, block.ln2, dBranchBuffer);
            encodeResidualAdd(commandBuffer, dBranchBuffer, dStreamBuffer, dBranchBuffer); //skip connection

            //attention backward: dBranch is the gradient into residual1
            {
                MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, projBwdDwPipeline);
                encoder->setBuffer(block.attnOutBuffer, 0, 0);
                encoder->setBuffer(dBranchBuffer, 0, 1);
                encoder->setBuffer(block.dProjwBuffer, 0, 2);
                encoder->setBytes(&attnParams, sizeof(attnParams), 3);
                encoder->dispatchThreadgroups(MTL::Size(tileCols, tileCols, 1), MTL::Size(32 * simdGroups, 1, 1));
                encoder->endEncoding();
            }
            {
                MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, projBwdDxPipeline);
                encoder->setBuffer(dBranchBuffer, 0, 0);
                encoder->setBuffer(block.projwBuffer, 0, 1);
                encoder->setBuffer(dZAttnBuffer, 0, 2);
                encoder->setBytes(&attnParams, sizeof(attnParams), 3);
                encoder->dispatchThreadgroups(MTL::Size(tileCols, tileRows, 1), MTL::Size(32 * simdGroups, 1, 1));
                encoder->endEncoding();
            }
            {
                MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, computeDPipeline);
                encoder->setBuffer(dZAttnBuffer, 0, 0);
                encoder->setBuffer(block.attnOutBuffer, 0, 1);
                encoder->setBuffer(dAttnDBuffer, 0, 2);
                encoder->setBytes(&attnParams, sizeof(attnParams), 3);
                encoder->dispatchThreadgroups(MTL::Size((size_t)tileRows * numHeads, 1, 1), MTL::Size(32 * simdGroups, 1, 1));
                encoder->endEncoding();
            }
            {
                MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, flashAttnBwdPipeline);
                encoder->setThreadgroupMemoryLength(tileM * tileN * sizeof(__bf16), 0); //S
                encoder->setThreadgroupMemoryLength(tileM * tileN * sizeof(__bf16), 1); //dP
                encoder->setThreadgroupMemoryLength(tileM * tileN * sizeof(__bf16), 2); //dS
                encoder->setBuffer(dZAttnBuffer, 0, 0);
                encoder->setBuffer(dAttnDBuffer, 0, 1);
                encoder->setBuffer(block.lBuffer, 0, 2);
                encoder->setBuffer(block.qBuffer, 0, 3);
                encoder->setBuffer(block.kBuffer, 0, 4);
                encoder->setBuffer(block.vBuffer, 0, 5);
                encoder->setBuffer(block.dQBuffer, 0, 6);
                encoder->setBuffer(block.dKBuffer, 0, 7);
                encoder->setBuffer(block.dVBuffer, 0, 8);
                encoder->setBytes(&attnParams, sizeof(attnParams), 9);
                encoder->dispatchThreadgroups(MTL::Size((size_t)tileRows * numHeads, 1, 1), MTL::Size(32 * simdGroups, 1, 1));
                encoder->endEncoding();
            }
            {
                MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, attnGradDxPipeline);
                encoder->setBuffer(block.dXAttnBuffer, 0, 0);
                encoder->setBuffer(block.qwBuffer, 0, 1);
                encoder->setBuffer(block.kwBuffer, 0, 2);
                encoder->setBuffer(block.vwBuffer, 0, 3);
                encoder->setBuffer(block.dQBuffer, 0, 4);
                encoder->setBuffer(block.dKBuffer, 0, 5);
                encoder->setBuffer(block.dVBuffer, 0, 6);
                encoder->setBytes(&attnParams, sizeof(attnParams), 7);
                encoder->dispatchThreadgroups(MTL::Size(tileCols, tileRows, 1), MTL::Size(32 * simdGroups, 1, 1));
                encoder->endEncoding();
            }
            {
                MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, attnGradDwPipeline);
                encoder->setBuffer(block.ln1.outBuffer, 0, 0);
                encoder->setBuffer(block.dQBuffer, 0, 1);
                encoder->setBuffer(block.dKBuffer, 0, 2);
                encoder->setBuffer(block.dVBuffer, 0, 3);
                encoder->setBuffer(block.dQwBuffer, 0, 4);
                encoder->setBuffer(block.dKwBuffer, 0, 5);
                encoder->setBuffer(block.dVwBuffer, 0, 6);
                encoder->setBytes(&attnParams, sizeof(attnParams), 7);
                encoder->dispatchThreadgroups(MTL::Size(tileCols, tileCols, 1), MTL::Size(32 * simdGroups, 1, 1));
                encoder->endEncoding();
            }
            //computeGradDx writes dX as float; ln1 backward wants bfloat in dStream
            encodeCast(commandBuffer, block.dXAttnBuffer, dStreamBuffer);
            encodeLayernormBackward(commandBuffer, block.ln1, dStreamBuffer);
            encodeResidualAdd(commandBuffer, dStreamBuffer, dBranchBuffer, dStreamBuffer); //skip connection
        }

        //embedding: dStream is now the gradient at the embedding output
        {
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, embedBwdPipeline);
            encoder->setBuffer(inputTokensBuffer, 0, 0);
            encoder->setBuffer(dStreamBuffer, 0, 1);
            encoder->setBuffer(dEmbedWeightBuffer, 0, 2);
            encoder->setBytes(&embedParams, sizeof(embedParams), 3);
            encoder->dispatchThreadgroups(MTL::Size(embedParams.t, 1, 1), MTL::Size(embedParams.group_size, 1, 1));
            encoder->endEncoding();
        }
        commitWait(commandBuffer, "backward");
        //gradients stay on the GPU; step() gathers them straight from the
        //per-parameter buffers
    }

    //------------------------------------------------------------------- step
    void addParam(std::vector<__bf16>& weight, MTL::Buffer* weightBuffer, MTL::Buffer* gradBuffer, bool gradIsFloat) {
        optWeights.push_back(&weight);
        optWeightBuffers.push_back(weightBuffer);
        optGradBuffers.push_back(gradBuffer);
        optGradIsFloat.push_back(gradIsFloat);
    }
    void addParam(Matrix<__bf16>& weight, MTL::Buffer* weightBuffer, MTL::Buffer* gradBuffer, bool gradIsFloat) {
        addParam(weight.matrix, weightBuffer, gradBuffer, gradIsFloat);
    }
    //everything stays on the GPU: gather each parameter's gradient into the
    //flat buffer, run Adam on the flat master params, scatter the updated
    //master back to the bf16 weight buffers the next forward reads
    void step() {
        stepCounter += 1.0f;
        StepParams stepParams{(uint32_t)optCount};
        stepParams.lr_t = LR * std::sqrt(1.0f - std::pow(stepParams.beta2, stepCounter))
                             / (1.0f - std::pow(stepParams.beta1, stepCounter));

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        for (size_t i = 0; i < optWeights.size(); ++i) {
            FlatCopyParams copyParams{(uint32_t)optWeights[i]->size(), (uint32_t)optOffsets[i]};
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer,
                optGradIsFloat[i] ? gatherFloatPipeline : gatherBfloatPipeline);
            encoder->setBuffer(optGradBuffers[i], 0, 0);
            encoder->setBuffer(gradientBuffer, 0, 1);
            encoder->setBytes(&copyParams, sizeof(copyParams), 2);
            encoder->dispatchThreadgroups(MTL::Size((copyParams.count + 1023) / 1024, 1, 1), MTL::Size(1024, 1, 1));
            encoder->endEncoding();
        }
        {
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, stepPipeline);
            encoder->setBuffer(parameterBuffer, 0, 0);
            encoder->setBuffer(gradientBuffer, 0, 1);
            encoder->setBuffer(momentumBuffer, 0, 2);
            encoder->setBuffer(varianceBuffer, 0, 3);
            encoder->setBytes(&stepParams, sizeof(stepParams), 4);
            encoder->dispatchThreadgroups(MTL::Size(optCount / 16384, 1, 1), MTL::Size(256, 1, 1));
            encoder->endEncoding();
        }
        for (size_t i = 0; i < optWeights.size(); ++i) {
            FlatCopyParams copyParams{(uint32_t)optWeights[i]->size(), (uint32_t)optOffsets[i]};
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, scatterPipeline);
            encoder->setBuffer(parameterBuffer, 0, 0);
            encoder->setBuffer(optWeightBuffers[i], 0, 1);
            encoder->setBytes(&copyParams, sizeof(copyParams), 2);
            encoder->dispatchThreadgroups(MTL::Size((copyParams.count + 1023) / 1024, 1, 1), MTL::Size(1024, 1, 1));
            encoder->endEncoding();
        }
        commitWait(commandBuffer, "adam step");
    }

    ~Model() {
        auto releaseLayernorm = [](LayernormBuffers& ln) {
            ln.gammaBuffer->release();
            ln.betaBuffer->release();
            ln.xnormBuffer->release();
            ln.stdevBuffer->release();
            ln.outBuffer->release();
            ln.dGammaBuffer->release();
            ln.dBetaBuffer->release();
        };
        for (BlockBuffers& block : blocks) {
            releaseLayernorm(block.ln1);
            releaseLayernorm(block.ln2);
            block.qwBuffer->release();
            block.kwBuffer->release();
            block.vwBuffer->release();
            block.projwBuffer->release();
            block.qBuffer->release();
            block.kBuffer->release();
            block.vBuffer->release();
            block.attnOutBuffer->release();
            block.lBuffer->release();
            block.projOutBuffer->release();
            block.residual1Buffer->release();
            block.upprojBuffer->release();
            block.downprojBuffer->release();
            block.mlpVBuffer->release();
            block.mlpMaskBuffer->release();
            block.mlpOutBuffer->release();
            block.residual2Buffer->release();
            block.dQBuffer->release();
            block.dKBuffer->release();
            block.dVBuffer->release();
            block.dXAttnBuffer->release();
            block.dQwBuffer->release();
            block.dKwBuffer->release();
            block.dVwBuffer->release();
            block.dProjwBuffer->release();
            block.dUpBuffer->release();
            block.dDpBuffer->release();
        }
        releaseLayernorm(lnFinal);
        inputTokensBuffer->release();
        embedWeightBuffer->release();
        posWeightBuffer->release();
        embedOutBuffer->release();
        dEmbedWeightBuffer->release();
        linearWBuffer->release();
        logitsBuffer->release();
        dLogitsBuffer->release();
        dLinearWBuffer->release();
        targetsBuffer->release();
        lossOutBuffer->release();
        lseBuffer->release();
        dStreamBuffer->release();
        dBranchBuffer->release();
        dXnormBuffer->release();
        dZAttnBuffer->release();
        dAttnDBuffer->release();
        dMlpVBuffer->release();
        parameterBuffer->release();
        gradientBuffer->release();
        momentumBuffer->release();
        varianceBuffer->release();
        embedFwdPipeline->release();
        embedBwdPipeline->release();
        layernormFwdPipeline->release();
        layernormBwdPipeline->release();
        qkvPipeline->release();
        flashAttnFwdPipeline->release();
        computeDPipeline->release();
        flashAttnBwdPipeline->release();
        attnGradDxPipeline->release();
        attnGradDwPipeline->release();
        projFwdPipeline->release();
        projBwdDwPipeline->release();
        projBwdDxPipeline->release();
        mlpFwdOnePipeline->release();
        mlpFwdTwoPipeline->release();
        mlpBwdOnePipeline->release();
        mlpBwdTwoPipeline->release();
        linearFwdPipeline->release();
        linearBwdPipeline->release();
        lossFwdPipeline->release();
        lossBwdPipeline->release();
        residualAddPipeline->release();
        castPipeline->release();
        stepPipeline->release();
        gatherFloatPipeline->release();
        gatherBfloatPipeline->release();
        scatterPipeline->release();
        commandQueue->release();
        pool->release();
        device->release();
    }
};
