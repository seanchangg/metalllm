#include <metal_stdlib>

//glue kernels so the whole forward/backward pass fits in one command buffer:
//residual adds and the float->bfloat handoff used to happen on the CPU
//between command buffers.

kernel void residualAdd(
    device const bfloat* a [[buffer(0)]],
    device const bfloat* b [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant metal::uint& count [[buffer(3)]],
    metal::uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        out[gid] = bfloat(float(a[gid]) + float(b[gid]));
    }
}

kernel void castFloatToBfloat(
    device const float* in [[buffer(0)]],
    device bfloat* out [[buffer(1)]],
    constant metal::uint& count [[buffer(2)]],
    metal::uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        out[gid] = bfloat(in[gid]);
    }
}
