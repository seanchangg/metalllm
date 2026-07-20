#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct AttentionParams {
    uint M; //input matrix dimensions
    uint N; //n_embed
    const uint D; //Head_size
    const uint NH;
};

constant uint TILE_M = 64;
constant uint TILE_N = 64;
constant int HEAD_SIZE = 64;
constant int N_EMBED = 384;

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
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Xt(X, dextents<int32_t, 2>(p.N, p.M)); // x=K, y=M
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> QWt(q_w, dextents<int32_t, 2>(p.N, p.N));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> KWt(k_w, dextents<int32_t, 2>(p.N, p.N));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> VWt(v_w, dextents<int32_t, 2>(p.N, p.N));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Qt(q, dextents<int32_t, 2>(p.N, p.M));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Kt(k, dextents<int32_t, 2>(p.N, p.M));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Vt(v, dextents<int32_t, 2>(p.N, p.M));
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


kernel void flashAttentionForward (
    threadgroup float* O [[threadgroup(0)]],
    threadgroup bfloat* S [[threadgroup(1)]], //[TILE_K][p.D]
    threadgroup float* row_max [[threadgroup(2)]],
    threadgroup float* row_sum [[threadgroup(3)]],
    device bfloat* q [[buffer(0)]],
    device bfloat* k [[buffer(1)]],
    device bfloat* v [[buffer(2)]],
    device bfloat* out [[buffer(3)]], //M x D
    device float* l [[buffer(4)]], //logsumexp vector
    constant AttentionParams& p [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]], //the row of attention tiles this threadgroup is responsible for
    uint tptg [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]]
) {
    const uint num_tiles = p.M / TILE_M;
    const uint tile = gid % num_tiles;
    const uint head = gid / num_tiles;
    const uint head_offset = head * p.D;
    const uint global_row_start = tile * TILE_M;
    if (tile > p.M / TILE_M || head > p.NH) return;

    for (uint i = tid; i < TILE_M * p.D; i += tptg) O[i] = bfloat(0);
    for (uint i = tid; i < TILE_M; i += tptg) { row_max[i] = -INFINITY; row_sum[i] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);


    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Qt(q, dextents<int32_t, 2>(p.N, p.M));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Kt(k, dextents<int32_t, 2>(p.N, p.M));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Vt(v, dextents<int32_t, 2>(p.N, p.M));

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
        threadgroup_barrier(mem_flags::mem_threadgroup); 
        for (int i = tid; i<(TILE_M*TILE_N); i+=tptg) { 
            float val = S[i] * d_sqrt;
            int grow = int(global_row_start) + (i >> 6);          //global query index
            int gcol = int(k_tile * TILE_N) + (i & (TILE_N - 1)); //global key index
            int sign = (grow - gcol) >> 31; //-1 if the key is in the future
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
        l[(global_row_start + row) * p.NH + head] = row_max[row] + log(row_sum[row]); //logsumexp, needed to rebuild P in backward

        for (int col = lane; col < p.D; col += 32) {
            out[(global_row_start + row) * p.N + head_offset+col] = bfloat(O[row*p.D + col] * sum_repr); //we don't need Ot anymore so we just write straight to HBM
        }
    }
}

kernel void computeD( //the sum(P @ dP) simplifies to rowsum(dO @ O) since dP = dO @ VT and O = P @ V and due to the commutative property we can transfer the V from dP to O and use O.
//This allows us to simplify the term into a pre-computed vector that we'll use in the backwards pass
    device bfloat* dO [[buffer(0)]], //M x N
    device bfloat* O [[buffer(1)]], //M x N
    device bfloat* D [[buffer(2)]], //M x NH
    constant AttentionParams& p [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint tptg [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]]
) {
    const uint num_tiles = p.M / TILE_M;
    const uint tile = gid % num_tiles;
    const uint head = gid / num_tiles;
    const uint head_offset = head * p.D;
    const uint global_row_start = tile * TILE_M;

    threadgroup float scratch[32];
    for (int row = sgid; row<TILE_M; row+=nsg) {
            uint row_start = row* p.N;
            float local_sum = 0.0f;
            for (int col = lane; col<HEAD_SIZE; col += 32) {
                local_sum += dO[global_row_start * p.N + row_start + head_offset+ col] * O[global_row_start * p.N + row_start +head_offset+ col];
            }
            local_sum = simd_sum(local_sum);
            if (lane == 0) {D[(global_row_start+row) * p.NH + head] = bfloat(local_sum);}
    }
}


kernel void flashAttentionBackward(
    threadgroup bfloat* S [[threadgroup(0)]], //TILE_M x TILE_N
    threadgroup bfloat* dP [[threadgroup(1)]], //TILE_M x TILE_N
    threadgroup bfloat* dS [[threadgroup(2)]],
    device bfloat* dZ [[buffer(0)]], //M x N
    device bfloat* d [[buffer(1)]],
    device float* l [[buffer(2)]],
    device bfloat* q [[buffer(3)]],
    device bfloat* k [[buffer(4)]],
    device bfloat* v [[buffer(5)]],
    device atomic_float* dQ [[buffer(6)]],
    device bfloat* dK [[buffer(7)]],
    device bfloat* dV [[buffer(8)]],
    constant AttentionParams& p [[buffer(9)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]], //the row of attention tiles this threadgroup is responsible for
    uint tptg [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]]
) {
    const uint num_tiles = p.M / TILE_M;
    const uint k_tile = gid % num_tiles; //this threadgroup OWNS key tile k_tile; dK/dV accumulate race-free
    const uint head = gid / num_tiles;
    const uint head_offset = head * p.D;
    if (k_tile >= num_tiles || head >= p.NH) return;

    for (uint i = tid; i < TILE_M * TILE_N; i += tptg) {dP[i] = bfloat(0); dS[i] = bfloat(0);}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> dZt(dZ, dextents<int32_t, 2>(p.N, p.M));

    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Qt(q, dextents<int32_t, 2>(p.N, p.M));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Kt(k, dextents<int32_t, 2>(p.N, p.M));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Vt(v, dextents<int32_t, 2>(p.N, p.M));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> dKt(dK, dextents<int32_t, 2>(p.N, p.M));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> dVt(dV, dextents<int32_t, 2>(p.N, p.M));

    tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> dPt(dP, dextents<int32_t, 2>(TILE_N, TILE_M));
    tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> dSt(dS, dextents<int32_t, 2>(TILE_N, TILE_M));
    tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> St(S, dextents<int32_t, 2>(TILE_N, TILE_M));

    constexpr auto desc = matmul2d_descriptor(
        TILE_M,                               // m: output tile rows
        TILE_N,                               // n: output tile cols
        HEAD_SIZE,
        false, true, false);            //transpose, full precision

    constexpr auto desc_dQ = matmul2d_descriptor(
        TILE_M,                               // m: output tile rows
        TILE_N,                               // n: output tile cols
        HEAD_SIZE, // k: read full K from the tensors
        false, false, false); //no tranpose, don't overwrite

    constexpr auto desc_dK = matmul2d_descriptor( //also the one we use for dV since P^T @ dO = dV
        TILE_M,                               // m: output tile rows
        TILE_N,                               // n: output tile cols
        HEAD_SIZE, // k: read full K from the tensors
        true, false, false, matmul2d_descriptor::mode::multiply_accumulate); //tranpose dS, don't overwrite

    matmul2d<desc, execution_simdgroups<4>> op;
    matmul2d<desc_dQ, execution_simdgroups<4>> op_dQ;
    matmul2d<desc_dK, execution_simdgroups<4>> op_dK;

    auto mdK = dKt.slice(head_offset, k_tile*TILE_N); //key rows: offset by k_tile only, not the query tile
    auto mdV = dVt.slice(head_offset, k_tile*TILE_N);

    auto mK = Kt.slice(head_offset, k_tile*TILE_N);
    auto mV = Vt.slice(head_offset, k_tile * TILE_N);

    float d_sqrt = 1 / sqrt(float(p.D));

    for (uint q_tile = k_tile; q_tile < num_tiles; ++q_tile) { //causal: only queries at or after this key tile attend to it
        uint q_row_start = q_tile*TILE_M;
        auto mQ = Qt.slice(head_offset, q_row_start);
        auto mdZ = dZt.slice(head_offset, q_row_start);
        op.run(mdZ, mV, dPt);
        op.run(mQ, mK, St);
        threadgroup_barrier(mem_flags::mem_threadgroup); //matmul outputs must land before the elementwise pass
        for (uint row = sgid; row < TILE_M; row+=nsg) {
            float l_val = l[(q_row_start + row) * p.NH + head]; //l and d are per QUERY row
            float d_val = float(d[(q_row_start + row) * p.NH + head]);
            for (uint col = lane; col < TILE_N; col += 32) {
                float val = exp(float(S[row*TILE_N + col]) * d_sqrt - l_val); //P = exp(score - logsumexp)
                if (k_tile == q_tile && col > row) val = 0.0f; //causal mask on the diagonal tile: P is a probability here, masked entries are 0 not -inf
                S[row*TILE_N + col] = bfloat(val); //now it's P
                dS[row * TILE_N + col] = bfloat(val * (float(dP[row*TILE_N + col]) - d_val) * d_sqrt);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op_dQ.run(dSt, mK, dPt); //reuse the dP buffer since we've already calculated dS. This only works because p.D = 64, which also happens to be TILE_N
        op_dK.run(dSt, mQ, mdK);
        op_dK.run(St, mdZ, mdV);
        threadgroup_barrier(mem_flags::mem_threadgroup); //S/dP/dS are reused next k_tile iteration
        for (uint row = sgid; row < TILE_M; row+= nsg) {
            for (uint col = lane; col < uint(p.D); col += 32) {
                atomic_fetch_add_explicit(&dQ[(q_row_start+row)*p.N + head_offset+col], float(dP[row*TILE_N + col]), memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void computeGradDx(
    device float* dX [[buffer(0)]],
    device bfloat* qw [[buffer(1)]],
    device bfloat* kw [[buffer(2)]],
    device bfloat* vw [[buffer(3)]],
    device float* dQ [[buffer(4)]],
    device bfloat* dK [[buffer(5)]],
    device bfloat* dV [[buffer(6)]],
    constant AttentionParams& p [[buffer(7)]],
    uint2 gid [[threadgroup_position_in_grid]] //the row of attention tiles this threadgroup is responsible for
) {
    const uint global_row_start = gid.y * TILE_M;
    const uint global_col_start = gid.x * TILE_N;
    constexpr auto desc_dx = matmul2d_descriptor(
        TILE_M,                               // m: output tile rows
        TILE_N,
        static_cast<int>(dynamic_extent), // k: read full K from the tensors. So dynamic extent automatically reduces over K internally. No need to write the logic myself
        false, true, false, matmul2d_descriptor::mode::multiply_accumulate); //dZ @ Wt
    
    matmul2d<desc_dx, execution_simdgroups<4>> dX_op;
        
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> qwt (qw, dextents<int32_t, 2>{p.D*p.NH, p.N});
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> kwt (kw, dextents<int32_t, 2>{p.D*p.NH, p.N});
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> vwt (vw, dextents<int32_t, 2>{p.D*p.NH, p.N});
    tensor<device float, dextents<int32_t, 2>, tensor_inline> dQt (dQ, dextents<int32_t, 2>{p.D*p.NH, p.M});
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> dKt (dK, dextents<int32_t, 2>{p.D*p.NH, p.M});
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> dVt(dV, dextents<int32_t, 2>{p.D*p.NH, p.M});
    tensor<device float, dextents<int32_t, 2>, tensor_inline> dXt (dX, dextents<int32_t, 2>{p.N, p.M});
    
    auto mdQ = dQt.slice(0, global_row_start);
    auto mdK = dKt.slice(0, global_row_start);
    auto mdV = dVt.slice(0, global_row_start);
    auto mdX = dXt.slice(global_col_start, global_row_start);
    auto mQw = qwt.slice(0, global_col_start);
    auto mKw = kwt.slice(0, global_col_start);
    auto mVw = vwt.slice(0, global_col_start);
    
    dX_op.run(mdQ, mQw, mdX);
    dX_op.run(mdK, mKw, mdX);
    dX_op.run(mdV, mVw, mdX);
}

kernel void computeGradDw(
    device bfloat* x [[buffer(0)]],
    device float* dQ [[buffer(1)]],
    device bfloat* dK [[buffer(2)]],
    device bfloat* dV [[buffer(3)]],
    device float* dQw [[buffer(4)]], //float dest: matmul2d only allows mixing bfloat operands with the float dQ if the destination is float
    device bfloat* dKw [[buffer(5)]],
    device bfloat* dVw [[buffer(6)]],
    constant AttentionParams& p [[buffer(7)]],
    uint2 gid [[threadgroup_position_in_grid]] //the row of attention tiles this threadgroup is responsible for
) {
    const uint global_row_start = gid.y * TILE_M;
    const uint global_col_start = gid.x * TILE_N;

    constexpr auto desc_dw = matmul2d_descriptor(
        TILE_N, // m: output tile rows
        TILE_M,
        static_cast<int>(dynamic_extent), // k: read full K from the tensors. So dynamic extent automatically reduces over K internally. No need to write the logic myself
        true, false, false, matmul2d_descriptor::mode::multiply_accumulate); //input^T @ dZ
    
    matmul2d<desc_dw, execution_simdgroups<4>> dW_op;
        
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Xt (x, dextents<int32_t, 2>{p.N, p.M});
    tensor<device float, dextents<int32_t, 2>, tensor_inline> dqwt (dQw, dextents<int32_t, 2>{p.D*p.NH, p.N});
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> dkwt (dKw, dextents<int32_t, 2>{p.D*p.NH, p.N});
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> dvwt (dVw, dextents<int32_t, 2>{p.D*p.NH, p.N});
    tensor<device float, dextents<int32_t, 2>, tensor_inline> dQt (dQ, dextents<int32_t, 2>{p.D*p.NH, p.M});
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> dKt (dK, dextents<int32_t, 2>{p.D*p.NH, p.M});
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> dVt(dV, dextents<int32_t, 2>{p.D*p.NH, p.M});
    
    auto mX = Xt.slice(global_row_start, 0);
    auto mdqw = dqwt.slice(global_col_start, global_row_start);
    auto mdkw = dkwt.slice(global_col_start, global_row_start);
    auto mdvw = dvwt.slice(global_col_start, global_row_start);
    auto mdQ = dQt.slice(global_col_start, 0);
    auto mdK = dKt.slice(global_col_start, 0);
    auto mdV = dVt.slice(global_col_start, 0);
    dW_op.run(mX, mdQ, mdqw);
    dW_op.run(mX, mdK, mdkw);
    dW_op.run(mX, mdV, mdvw);
}
    
kernel void projForward (
    device bfloat* attn_out [[buffer(0)]],
    device bfloat* proj_w [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant AttentionParams& p [[buffer(3)]],
    uint2 gid [[threadgroup_position_in_grid]]
    ) {
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Ot(attn_out, dextents<int32_t, 2>(p.N, p.M));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Wt(proj_w, dextents<int32_t, 2>(p.N, p.N));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Outt(out, dextents<int32_t, 2>(p.N, p.M));
    constexpr auto desc = matmul2d_descriptor(
        64,
        64,
        static_cast<int>(dynamic_extent),
        false, false, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = Ot.slice(0, gid.x * 64);
    auto mW = Wt.slice(gid.y * 64, 0);
    auto mO = Outt.slice(gid.y * 64, gid.x * 64);
    op.run(mA, mW, mO);
}

kernel void projBackwarddW (
    device bfloat* attn_out [[buffer(0)]],
    device bfloat* dZ [[buffer(1)]],
    device bfloat* dProjw [[buffer(2)]],
    constant AttentionParams& p [[buffer(3)]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    uint global_row_start = gid.y * TILE_M;
    uint global_col_start = gid.x * TILE_N;
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Ot(attn_out, dextents<int32_t, 2>(p.N, p.M));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> dZt(dZ, dextents<int32_t, 2>(p.N, p.M));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> dProjwt(dProjw, dextents<int32_t, 2>(p.N, p.N));
    constexpr auto desc_dw = matmul2d_descriptor(
        TILE_N,
        TILE_M,
        static_cast<int>(dynamic_extent),
        true, false, false, matmul2d_descriptor::mode::multiply_accumulate); //O^T @ dZ
    
    matmul2d<desc_dw, execution_simdgroups<4>> dW_op;
    auto mO = Ot.slice(global_row_start, 0);    
    auto mdZ = dZt.slice(global_col_start, 0);
    auto mdProjw = dProjwt.slice(global_col_start, global_row_start);
    dW_op.run(mO, mdZ, mdProjw);
}

kernel void projBackwarddX (
    device bfloat* dZ [[buffer(0)]],
    device bfloat* proj_w [[buffer(1)]],
    device bfloat* dAttn [[buffer(2)]],
    constant AttentionParams& p [[buffer(3)]],
    uint2 gid [[threadgroup_position_in_grid]]
 ) {
    uint global_row_start = gid.y * TILE_M;
    uint global_col_start = gid.x * TILE_N;
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Wt(proj_w, dextents<int32_t, 2>(p.N, p.N));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> dAttnt(dAttn, dextents<int32_t, 2>(p.N, p.M));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> dZt(dZ, dextents<int32_t, 2>(p.N, p.M));
    constexpr auto desc_dx = matmul2d_descriptor(
        TILE_M,
        TILE_N,
        static_cast<int>(dynamic_extent),
        false, true, false); //dZ @ W^T
        
    matmul2d<desc_dx, execution_simdgroups<4>> dX_op;
    auto mdAttn = dAttnt.slice(global_col_start, global_row_start);
    auto mW = Wt.slice(0, global_col_start);
    auto mdZ = dZt.slice(0, global_row_start);
    dX_op.run(mdZ, mW, mdAttn);
}


