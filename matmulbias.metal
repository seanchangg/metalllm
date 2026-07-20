#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

struct MatMulParams {
    metal::uint M;
    metal::uint N;
    metal::uint K;
};

kernel void MPPMatMulBias(
    device float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant MatMulParams& p [[buffer(4)]],
    metal::uint tid [[thread_index_in_threadgroup]],
    metal::uint2 tgid [[threadgroup_position_in_grid]]
) {

    metal::tensor<device float, metal::dextents<int32_t, 2>, metal::tensor_inline> At(A, metal::dextents<int32_t, 2>(p.K, p.M)); // x=K, y=M
    metal::tensor<device float, metal::dextents<int32_t, 2>, metal::tensor_inline> Bt(B, metal::dextents<int32_t, 2>(p.N, p.K)); // x=N, y=K
    metal::tensor<device float, metal::dextents<int32_t, 2>, metal::tensor_inline> Ct(out, metal::dextents<int32_t, 2>(p.N, p.M)); // x=N, y=M

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        64,
        64,
        static_cast<int>(metal::dynamic_extent),
        false, false, false
    );

    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<4>> op;

    auto mA = At.slice(0,           tgid.x * 64); // A rows [tgid.x*64 ..], all K
    auto mB = Bt.slice(tgid.y * 64, 0);           // B cols [tgid.y*64 ..], all K
    auto mC = Ct.slice(tgid.y * 64, tgid.x * 64); // the 64x64 output tile
    op.run(mA, mB, mC);

    threadgroup_barrier(metal::mem_flags::mem_device);

    metal::uint row_start = tgid.x * 64;
    metal::uint col_start = tgid.y * 64;
    for (metal::uint idx = tid; idx < 64u * 64u; idx += 128u) {
        metal::uint lr = idx / 64, lc = idx % 64;
        out[(row_start + lr) * p.N + (col_start + lc)] += bias[col_start + lc];
    }
}

kernel void MPPMatMulBiasBfloat(
    device bfloat* A [[buffer(0)]],
    device bfloat* B [[buffer(1)]],
    device float* bias [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant MatMulParams& p [[buffer(4)]],
    metal::uint tid [[thread_index_in_threadgroup]],
    metal::uint2 tgid [[threadgroup_position_in_grid]]
) {
    using namespace metal;
    using namespace mpp::tensor_ops;

    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> At(A, dextents<int32_t, 2>(p.K, p.M)); // x=K, y=M
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Bt(B, dextents<int32_t, 2>(p.N, p.K)); // x=N, y=K
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Ct(out, dextents<int32_t, 2>(p.N, p.M)); // x=N, y=M

    constexpr auto desc = matmul2d_descriptor(
        64,
        64,
        static_cast<int>(dynamic_extent),
        false, false, false
    );

    matmul2d<desc, execution_simdgroups<4>> op;

    auto mA = At.slice(0,           tgid.x * 64);
    auto mB = Bt.slice(tgid.y * 64, 0);
    auto mC = Ct.slice(tgid.y * 64, tgid.x * 64);
    op.run(mA, mB, mC);

    threadgroup_barrier(mem_flags::mem_device);

    uint row_start = tgid.x * 64;
    uint col_start = tgid.y * 64;
    for (uint idx = tid; idx < 64u * 64u; idx += 128u) {
        uint lr = idx / 64, lc = idx % 64;
        out[(row_start + lr) * p.N + (col_start + lc)] += bfloat(bias[col_start + lc]);
    }
}
