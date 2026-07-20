# metalLLM

A from-scratch GPT-style LLM **training** framework written in C++ and Metal Shading Language — no PyTorch, no MLX, no MPSGraph. Every kernel in the forward pass, backward pass, and optimizer is hand-written, and the entire training step (forward → backward → Adam) runs on the GPU. The host only uploads the batch and reads back the loss each iteration.

It trains a character-level transformer on `drake_lyrics.txt` (included) and reaches **~23 ms/iter on an M4 MacBook Air** at the default config below.

## Model architecture

Standard pre-LN decoder-only transformer:

- Token embedding + learned positional embedding
- N × blocks of: `LayerNorm → causal multi-head attention (FlashAttention) → residual add → LayerNorm → MLP (Linear → ReLU → Linear) → residual add`
- Final LayerNorm → output projection to vocab → softmax cross-entropy loss
- AdamW-style Adam with bias correction, mixed-precision (bf16 compute, fp32 master weights)

Default configuration (`main.cpp`):

| Constant | Value |
|---|---|
| `BLOCK_SIZE` (sequence length) | 256 |
| `N_HEADS` × `HEAD_SIZE` | 6 × 64 (`N_EMBED` = 384) |
| `N_LAYERS` | 6 |
| `MLP_SCALE` | 2 |
| `LR` | 0.003 |
| Vocab | character-level, padded up to a multiple of 64 |

Matmul kernels are built on Metal 4's tensor ops (`metal_tensor` + MetalPerformancePrimitives `matmul2d`), running 64×64 output tiles with 4 cooperative simdgroups per threadgroup.

## Kernel inventory

| File | Kernels | Role |
|---|---|---|
| `embed.metal` | `embedForward`, `embedBackward` | Token embedding lookup; backward scatters into fp32 atomic accumulators |
| `attention.metal` | `computeQKV`, `flashAttentionForward`, `computeD`, `flashAttentionBackward`, `computeGradDx`, `computeGradDw`, `projForward`, `projBackwarddW`, `projBackwarddX` | Full attention forward/backward (FlashAttention-2 style) |
| `mlp.metal` | `mlpForwardOne`, `mlpForwardTwo`, `mlpBackwardOne`, `mlpBackwardTwo` | Feed-forward network, ReLU fused into the matmuls |
| `layernorm.metal` | `layernormForward`, `layernormBackward` | Row-parallel LayerNorm with cached `x_norm`/`stdev` |
| `linear.metal` | `linearForward`, `linearBackward` | Output projection to vocab (dX and dW fused into one backward kernel) |
| `loss.metal` | `lossForward`, `lossBackward` | Fused softmax + cross-entropy via logsumexp |
| `elementwise.metal` | `residualAdd`, `castFloatToBfloat`, `gatherGradFloat`, `gatherGradBfloat`, `scatterWeight` | Glue kernels that keep the whole pass on-GPU |
| `optimstep.metal` | `MetalStep` | Fused Adam over one flat parameter buffer, vectorized ×4 |
| `matmul.metal`, `matmulbias.metal`, `relu.metal`, `softmax.metal` | — | Legacy/reference kernels from before fusion; not dispatched by the model, kept as the base reference for how `matmul2d` tiling works |

## Design choices: where to fuse, where not to

The guiding rules that fell out of writing this:

1. **Fuse when ops share input tiles or an elementwise epilogue can ride on a matmul's output tile** while it's still resident in registers/threadgroup memory.
2. **Unfuse (split kernels) when the next op needs the *entire* previous tensor.** Metal has no cross-threadgroup barrier inside a kernel — the kernel boundary *is* the global barrier — so a full-tensor dependency forces a new dispatch.
3. **Unfuse when two ops want different grid shapes/tilings.** Cramming them into one kernel means divergent branches and idle threadgroups; separate dispatches keep threadgroup occupancy high.

### Fused

- **QKV projection (`computeQKV`)** — one kernel runs all three matmuls `X@Wq`, `X@Wk`, `X@Wv`. Each threadgroup loads its X tile once and reuses it for all three outputs, instead of three kernels each re-reading X from device memory.
- **FlashAttention forward (`flashAttentionForward`)** — the big one: `Q@Kᵀ`, 1/√d scaling, causal masking, *online* softmax (running row-max/row-sum with rescaling of the output accumulator), and the `P@V` accumulation all live in a single kernel. The `M×M` attention matrix is never materialized in device memory — only a 64×64 score tile in threadgroup memory. Per-row logsumexp is saved so the backward pass can rebuild `P` cheaply.
- **FlashAttention backward (`flashAttentionBackward`)** — recomputes `P` from Q/K and the cached logsumexp, then computes `dP`, `dS`, and accumulates `dQ`, `dK`, `dV` in one kernel. Each threadgroup *owns one key tile* and loops over query tiles, so its `dK`/`dV` accumulation is race-free by construction; only `dQ` (which every key tile contributes to) needs fp32 atomics.
- **ReLU into the MLP matmuls** — `mlpForwardOne` computes `V = X@Up` and applies ReLU + writes the activation mask as an epilogue on the same output tile; `mlpBackwardOne` computes `dV = dZ@Dpᵀ` and applies the cached mask in place. ReLU never exists as its own dispatch (the standalone `relu.metal` is the pre-fusion leftover).
- **Attention input gradients (`computeGradDx`)** — `dX = dQ@Wqᵀ + dK@Wkᵀ + dV@Wvᵀ` runs as three `multiply_accumulate` matmuls into the same output tile in one kernel, instead of three dispatches plus two adds. `computeGradDw` does the same for the three weight gradients.
- **Loss (`lossForward` / `lossBackward`)** — softmax and cross-entropy are fused through the logsumexp trick: forward caches one `lse` scalar per row, backward reconstructs probabilities from logits + `lse`, so the full softmax output is never stored.
- **Linear backward (`linearBackward`)** — dX and dW share one kernel: every threadgroup computes its row-tile of dX, then the dW column tiles (there are `vocab/64` of them, more than there are threadgroups) are strided across threadgroups so each tile is reduced exactly once over the full X/dZ.
- **Optimizer (`step()`)** — all parameters live in one flat fp32 master buffer, so Adam is a single `MetalStep` dispatch (float4/bfloat4 vectorized, 16 elements per thread, 16 384 per threadgroup) rather than one dispatch per tensor. Per-parameter `gather` (grad → flat bf16) and `scatter` (fp32 master → bf16 weight buffers) kernels bracket it inside the same command buffer.
- **The whole pass, at the command-buffer level** — forward is *one* command buffer, backward is *one* command buffer. The `residualAdd` and `castFloatToBfloat` glue kernels in `elementwise.metal` exist precisely so residual adds and the float→bfloat handoff (between attention backward and ln1 backward) never bounce through the CPU and split the pass into multiple submissions.

### Deliberately unfused

- **QKV projection vs. FlashAttention** — the attention kernel's first key tile needs *all* rows of K and V, not just the ones its own threadgroup would have projected. Full-tensor dependency → kernel boundary (rule 2).
- **`computeD` before FlashAttention backward** — the backward needs `D = rowsum(dO ∘ O)` per *query* row, but the backward kernel is parallelized over *key* tiles, each of which visits every query row. Precomputing D in its own tiny reduction kernel avoids every key-tile threadgroup redundantly recomputing it (and avoids a cross-threadgroup dependency). The rowsum(P∘dP) term from the FlashAttention paper is algebraically moved onto `dO∘O` so it can be computed without rebuilding P.
- **MLP stage 1 vs. stage 2 (`mlpForwardOne` / `mlpForwardTwo`, same split in backward)** — the down-projection `V@Dp` reduces over the *entire* hidden dimension, so every output tile needs every column tile of V finished. Same full-tensor barrier argument as QKV→attention.
- **`projBackwarddW` vs. `projBackwarddX`, and `computeGradDw` vs. `computeGradDx`** — the sibling dW and dX matmuls read the same gradients but want different grid shapes (dW tiles the `N×N` weight, dX tiles the `M×N` activation) and different transpose descriptors. Keeping them as separate dispatches keeps threadgroup occupancy high instead of one kernel where half the threadgroups branch idle (rule 3).
- **`mlpBackwardTwo` is the exception that proves the rule** — it *does* pack dUp, dDp, and dX into one kernel using two independent tilings over a 1-D grid, with each half guarded by its own bound and the host dispatching `max(hidden_tiles, dx_tiles)`. It works because the tile counts are close at this config; the attention-side equivalents were left split because their grids diverge more.
- **Residual adds and casts stay as their own trivial kernels** — they're memory-bound one-liners; fusing them into neighbors would save nothing measurable, and as standalone kernels they can be reused everywhere (both skip connections per block, both passes, the pos-embedding add).

## Numerics / precision policy

- **bf16** for weights, activations, and most gradients — halves bandwidth on what is largely a bandwidth-bound workload.
- **fp32 accumulation everywhere it matters**: the matmul2d accumulators, the attention output accumulator and softmax row statistics, logsumexp, LayerNorm mean/variance, and the loss.
- **fp32 atomic accumulators** for gradients that are scattered into concurrently: embedding table, LayerNorm dγ/dβ, attention dQ (and dQw — `matmul2d` requires a float destination when mixing a float operand with bfloat). There are no bf16 atomics on Metal, and these sums are exactly where bf16 accumulation would lose gradient signal. A `castFloatToBfloat` kernel converts back into the bf16 stream on-GPU.
- **fp32 master weights** in the Adam step (classic mixed-precision recipe): Adam updates the flat fp32 master copy, then `scatterWeight` re-quantizes to the bf16 buffers the next forward reads. Momentum/variance are stored bf16.

## Execution / memory model

- All weights, activations, gradients, and Adam state are **GPU-resident**, allocated once at model construction (`model.h`). Per iteration the host writes only the input tokens and targets (256 × 4 B each) and reads back 256 loss floats.
- Forward, backward, and step each encode into **a single command buffer** — one `commit`/`wait` per phase, three per training iteration.
- Activations are cached per-block between forward and backward (ln outputs, Q/K/V, attention output, logsumexp, MLP hidden + ReLU mask, residuals). Backward-only scratch that is consumed in order (`dStream`, `dBranch`, `dXnorm`, …) is shared across all blocks.
- One pipeline state per kernel, shared by every block.

## Building and running

Requirements:

- Apple silicon Mac with an SDK recent enough for Metal 4 tensor ops (`<metal_tensor>` + MetalPerformancePrimitives)
- [metal-cpp](https://developer.apple.com/metal/cpp/) headers — set `METAL_CPP_DIR`, or place them at `~/.local/include/metal-cpp`
- CMake ≥ 3.20

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cd build && ./metalLLM   # expects drake_lyrics.txt in the working directory
```

CMake compiles every sibling `.metal` file to AIR via `xcrun metal` and links them into `default.metallib`, which the binary loads as the default library at startup.

Training prints one line per iteration:

```
iter 42 | loss 2.13 | 23 ms
```

### Performance

~23 ms/iter on an M4 MacBook Air at the default config (256 tokens, 6 layers, 384 embed, 6 heads × 64, MLP scale 4). Finally beat PyTorch (by about 6 ms/iter)

## Repo layout notes

- `model.h` — the whole host-side model: buffer allocation, pipeline setup, and the `forward()` / `backward()` / `step()` encoders. Depends on structs in `main.cpp`, so it's included after them.
- `main.cpp` — kernel param structs, weight init (GPT-2-style: 0.02 std, residual projections scaled by 1/√(2·N_LAYERS)), char tokenizer plumbing, training loop.
- `apple-fb-repro/` — a self-contained repro filed with Apple Feedback: `flashAttentionForward` is miscompiled by the Metal compiler at default optimization (row 0 of a causal head must equal V row 0; it doesn't) but passes when the metallib is compiled with `-O0`.

## References

- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3](https://arxiv.org/abs/2407.08608)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
