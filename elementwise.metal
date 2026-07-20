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

//GPU-resident optimizer plumbing: gather copies each per-parameter gradient
//buffer into its slice of the flat Adam gradient buffer (converting the float
//atomic accumulators to bfloat), scatter writes the updated float master
//params back out to the per-parameter bfloat weight buffers the forward pass
//reads. offset is in elements, into the flat optimizer buffers.
struct FlatCopyParams {
    metal::uint count;
    metal::uint offset;
};

kernel void gatherGradFloat(
    device const float* src [[buffer(0)]],
    device bfloat* dst [[buffer(1)]],
    constant FlatCopyParams& p [[buffer(2)]],
    metal::uint gid [[thread_position_in_grid]]
) {
    if (gid < p.count) {
        dst[p.offset + gid] = bfloat(src[gid]);
    }
}

kernel void gatherGradBfloat(
    device const bfloat* src [[buffer(0)]],
    device bfloat* dst [[buffer(1)]],
    constant FlatCopyParams& p [[buffer(2)]],
    metal::uint gid [[thread_position_in_grid]]
) {
    if (gid < p.count) {
        dst[p.offset + gid] = src[gid];
    }
}

kernel void scatterWeight(
    device const float* master [[buffer(0)]],
    device bfloat* dst [[buffer(1)]],
    constant FlatCopyParams& p [[buffer(2)]],
    metal::uint gid [[thread_position_in_grid]]
) {
    if (gid < p.count) {
        dst[gid] = bfloat(master[p.offset + gid]);
    }
}
