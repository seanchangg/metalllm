#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

struct MatmulParams {
    metal::uint M;
    metal::uint N;
    metal::uint K;
};

// ───────────────── Tensor-op matmul (Metal 4 MetalPerformancePrimitives) ────
//
// C(MxN) = A(MxK) * B(KxN), all row-major float. Uses mpp::tensor_ops::matmul2d,
// which runs on the GPU matrix/tensor units. The raw device buffers are wrapped
// as `tensor_inline` tensors in-kernel, so the host can keep using plain
// MTL::Buffer (no MTLTensor binding required).
//
// Each threadgroup computes a 64x64 output tile cooperatively with 4 SIMD groups.
// Dispatch: threadgroupsPerGrid=(M/64, N/64, 1), threadsPerThreadgroup=(128,1,1)
// (4 SIMD groups x 32-wide). Assumes M and N multiples of 64.
kernel void MPPMatMul(
    device float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant MatmulParams& p [[buffer(3)]],
    metal::uint2 tgid [[threadgroup_position_in_grid]])
{
    using namespace metal;
    using namespace mpp::tensor_ops;

    // Wrap row-major buffers as 2D tensors. Coordinate is [x, y] with x the
    // contiguous (inner) dimension: element[x,y] == ptr[y*extent0 + x].
    tensor<device float, dextents<int32_t, 2>, tensor_inline> At(A, dextents<int32_t, 2>(p.K, p.M)); // x=K, y=M
    tensor<device float, dextents<int32_t, 2>, tensor_inline> Bt(B, dextents<int32_t, 2>(p.N, p.K)); // x=N, y=K
    tensor<device float, dextents<int32_t, 2>, tensor_inline> Ct(out, dextents<int32_t, 2>(p.N, p.M)); // x=N, y=M

    constexpr auto desc = matmul2d_descriptor(
        64,                               // m: output tile rows
        64,                               // n: output tile cols
        static_cast<int>(dynamic_extent), // k: read full K from the tensors
        false, false, false);            // no transpose, full precision

    matmul2d<desc, execution_simdgroups<4>> op;

    auto mA = At.slice(0,           tgid.x * 64); // A rows [tgid.x*64 ..], all K
    auto mB = Bt.slice(tgid.y * 64, 0);           // B cols [tgid.y*64 ..], all K
    auto mC = Ct.slice(tgid.y * 64, tgid.x * 64); // the 64x64 output tile

    op.run(mA, mB, mC);
}

kernel void MPPMatMulBfloat(
    device bfloat* A [[buffer(0)]],
    device bfloat* B [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant MatmulParams& p [[buffer(3)]],
    metal::uint2 tgid [[threadgroup_position_in_grid]])
{
    using namespace metal;
    using namespace mpp::tensor_ops;

    // Wrap row-major buffers as 2D tensors. Coordinate is [x, y] with x the
    // contiguous (inner) dimension: element[x,y] == ptr[y*extent0 + x].
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> At(A, dextents<int32_t, 2>(p.K, p.M)); // x=K, y=M
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Bt(B, dextents<int32_t, 2>(p.N, p.K)); // x=N, y=K
    tensor<device float, dextents<int32_t, 2>, tensor_inline> Ct(out, dextents<int32_t, 2>(p.N, p.M)); // x=N, y=M

    constexpr auto desc = matmul2d_descriptor(
        64,                               // m: output tile rows
        64,                               // n: output tile cols
        static_cast<int>(dynamic_extent), // k: read full K from the tensors
        false, false, false);            // no transpose, full precision

    matmul2d<desc, execution_simdgroups<4>> op;

    auto mA = At.slice(0,           tgid.x * 64); // A rows [tgid.x*64 ..], all K
    auto mB = Bt.slice(tgid.y * 64, 0);           // B cols [tgid.y*64 ..], all K
    auto mC = Ct.slice(tgid.y * 64, tgid.x * 64); // the 64x64 output tile

    op.run(mA, mB, mC);
}

kernel void MPPMatMulBfloatBfloat(
    device bfloat* A [[buffer(0)]],
    device bfloat* B [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant MatmulParams& p [[buffer(3)]],
    metal::uint2 tgid [[threadgroup_position_in_grid]])
{
    using namespace metal;
    using namespace mpp::tensor_ops;

    // Wrap row-major buffers as 2D tensors. Coordinate is [x, y] with x the
    // contiguous (inner) dimension: element[x,y] == ptr[y*extent0 + x].
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> At(A, dextents<int32_t, 2>(p.K, p.M)); // x=K, y=M
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Bt(B, dextents<int32_t, 2>(p.N, p.K)); // x=N, y=K
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Ct(out, dextents<int32_t, 2>(p.N, p.M)); // x=N, y=M

    constexpr auto desc = matmul2d_descriptor(
        64,                               // m: output tile rows
        64,                               // n: output tile cols
        static_cast<int>(dynamic_extent), // k: read full K from the tensors
        false, false, false);            // no transpose, full precision

    matmul2d<desc, execution_simdgroups<4>> op;

    auto mA = At.slice(0,           tgid.x * 64); // A rows [tgid.x*64 ..], all K
    auto mB = Bt.slice(tgid.y * 64, 0);           // B cols [tgid.y*64 ..], all K
    auto mC = Ct.slice(tgid.y * 64, tgid.x * 64); // the 64x64 output tile

    op.run(mA, mB, mC);
}