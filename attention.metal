#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct AttentionParams {
    uint M; //input matrix dimensions
    uint K; //n_embed
    const uint D; //Head_size
    const uint NH;
};

kernel void computeQKV (
    device bfloat* X [[buffer(0)]],
    device bfloat* q [[buffer(1)]],
    device bfloat* k [[buffer(2)]],
    device bfloat* v [[buffer(3)]],
    device bfloat* q_w [[buffer(4)]],
    device bfloat* k_w [[buffer(5)]],
    device bfloat* v_w [[buffer(6)]],
    constant AttentionParams& p [[buffer(7)]],
    uint2 gid [[threadgroup_position_in_grid]] //which row of QKV this threadgroup is computing
    ) {
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Xt(X, dextents<int32_t, 2>(p.K, p.M)); // x=K, y=M
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> QWt(q_w, dextents<int32_t, 2>(p.D, p.K));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> KWt(k_w, dextents<int32_t, 2>(p.D, p.K));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> VWt(v_w, dextents<int32_t, 2>(p.D, p.K));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Qt(q, dextents<int32_t, 2>(p.D, p.M));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Kt(k, dextents<int32_t, 2>(p.D, p.M));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Vt(v, dextents<int32_t, 2>(p.D, p.M));
    constexpr auto desc = matmul2d_descriptor(
        64,                               // m: output tile rows
        64,                               // n: output tile cols
        static_cast<int>(dynamic_extent), // k: read full K from the tensors. So dynamic extent automatically reduces over K internally. No need to write the logic myself
        false, false, false); // no transpose, full precision
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = Xt.slice(0, gid.x * 64); // A rows [gid.x*64 ..], all K
    auto mQW = QWt.slice(gid.y * 64, 0);           // B cols [gid.y*64 ..], all K
    auto mQ = Qt.slice(gid.y * 64, gid.x * 64); // the 64x64 output tile
    auto mKW = KWt.slice(gid.y * 64, 0);           // B cols [gid.y*64 ..], all K
    auto mK = Kt.slice(gid.y * 64, gid.x * 64); // the 64x64 output tile
    auto mVW = VWt.slice(gid.y * 64, 0);           // B cols [gid.y*64 ..], all K
    auto mV = Vt.slice(gid.y * 64, gid.x * 64); // the 64x64 output tile
    op.run(mA,mQW,mQ);
    op.run(mA,mKW,mK);
    op.run(mA,mVW,mV);
}


kernel void attentionForward (
    threadgroup float* O [[threadgroup(0)]],
    threadgroup bfloat* S [[threadgroup(1)]], //[TILE_K][p.D]
    threadgroup float* row_max [[threadgroup(2)]],
    threadgroup float* row_sum [[threadgroup(3)]],
    device bfloat* q [[buffer(0)]],
    device bfloat* k [[buffer(1)]],
    device bfloat* v [[buffer(2)]],
    device bfloat* softmax_out [[buffer(3)]],
    device bfloat* out [[buffer(4)]], //M x D
    constant AttentionParams& p [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]], //the row of attention tiles this threadgroup is responsible for
    uint tptg [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]]
) {
    constexpr uint TILE_M = 64;
    constexpr uint TILE_N = 64;
    constexpr int HEAD_SIZE = 64;
    const uint num_tiles = p.M / TILE_M;
    const uint tile = gid % num_tiles;
    const uint head = gid / num_tiles;
    const uint head_offset = head * p.D;
    const uint global_row_start = tile * TILE_M;
    if (tile > p.M / TILE_M || head > p.NH) return;

    for (uint i = tid; i < TILE_M * p.D; i += tptg) O[i] = bfloat(0);
    for (uint i = tid; i < TILE_M; i += tptg) { row_max[i] = -INFINITY; row_sum[i] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);


    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Qt(q, dextents<int32_t, 2>(p.K, p.M));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Kt(k, dextents<int32_t, 2>(p.K, p.M));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Vt(v, dextents<int32_t, 2>(p.K, p.M));

    tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> St(S, dextents<int32_t, 2>(TILE_N, TILE_M));
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> Ot(O, dextents<int32_t, 2>(p.D, TILE_M));

    constexpr auto desc = matmul2d_descriptor(
        TILE_M,                               // m: output tile rows
        TILE_N,                               // n: output tile cols
        HEAD_SIZE, // k: read full K from the tensors
        false, true, false);            //transpose, full precision

    constexpr auto desc_nt = matmul2d_descriptor(
        TILE_M,                               // m: output tile rows
        TILE_N,                               // n: output tile cols
        HEAD_SIZE, // k: read full K from the tensors
        false, false, false, matmul2d_descriptor::mode::multiply_accumulate); //no tranpose, don't overwrite

    matmul2d<desc, execution_simdgroups<4>> op;
    matmul2d<desc_nt, execution_simdgroups<4>> op_nt;

    auto mQ = Qt.slice(head_offset, global_row_start);
    float d_sqrt = 1 / sqrt(float(p.D));

    for (int k_tile = 0; k_tile<=tile; ++k_tile) { //only loop for lower triangle
        auto mK = Kt.slice(head_offset, k_tile * TILE_N);
        op.run(mQ, mK, St); //automatically coordinates between simdgroups and individual threads
        for (int i = tid; i<(TILE_M*TILE_N); i+=tptg) { //scale and mask upper triangle if on diagonal
            float val = S[i] * d_sqrt;
            int sign = 0;
            if (k_tile == tile) {
                int row = i >> 6; //log2(TILE_N)
                int col = i & (TILE_N - 1);
                sign = (row-col) >> 31; //-1 if col > row
            }
            S[i] = bfloat(val + as_type<float>(sign & as_type<int>(-INFINITY)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // each simd group takes care of 1 row of the attention tile

        for (int row = sgid; row<TILE_M; row+=nsg) { //calculate new row_max and add to row_sum. Also adjust previous buffers
            uint row_start = row* TILE_N;
            float new_max = -INFINITY;
            for (int col = lane; col<TILE_N; col += 32) {
                new_max = max(new_max, S[row_start + col]);
            }
            new_max = simd_max(new_max);
            float old_max = row_max[row];
            new_max = max(new_max, old_max);
            float exp_diff = exp(old_max - new_max);
            float local_sum = 0.0f;
            for (int col = lane; col<TILE_N; col += 32) { //computing S - row_max
                float val = exp(S[row*TILE_N + col] - new_max);
                S[row * TILE_N + col] = bfloat(val);
                local_sum += val;
            }
            for (int o_col = lane; o_col<p.D; o_col+= 32) {
                O[row * p.D + o_col] = O[row*p.D + o_col] * exp_diff;
            }
            if (lane == 0) {
                row_max[row] = new_max;
            }
            //compute row_sum and add to running total
            local_sum = simd_sum(local_sum);
            if (lane == 0) {row_sum[row] = row_sum[row] * exp_diff + local_sum;}
        }
        auto mV = Vt.slice(head_offset, k_tile * TILE_N);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op_nt.run(St, mV, Ot);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int row = sgid; row < TILE_M; row += nsg) {
        float sum_repr = 1 / row_sum[row];
        for (int col = lane; col < p.D; col += 32) {
            out[(global_row_start + row) * p.K + head_offset+col] = bfloat(O[row*p.D + col] * sum_repr); //we don't need Ot anymore so we just write straight to HBM
        }
    }
}