#include <metal_stdlib>

struct StepParams {
    const metal::uint paramCount;
    const float beta1;
    const float beta2;
    const float lr_t; //= lr * sqrt(1-beta2^t) / (1-beta1^t)
    const float one_minus_beta1; //1-beta1
    const float one_minus_beta2; //1-beta2
    const float eps;
};

kernel void MetalStep (
    device float4* params4 [[buffer(0)]],
    device bfloat4* grads4 [[buffer(1)]],
    device bfloat4* m [[buffer(2)]],
    device bfloat4* v [[buffer(3)]],
    constant StepParams& p [[buffer(4)]],
    metal::uint tid [[thread_position_in_threadgroup]],
    metal::uint gid [[threadgroup_position_in_grid]]
) {
    //design choices (idk) - 256 threads per threadgroup. Each thread will compute exactly 16 outputs.
    //So each threadgroup is responsible for a batch of 16384 output elements
    constexpr metal::uint ELEMS_PER_GROUP = 16384/ 4; //n batches of 4
    constexpr metal::uint THREADS = 256;
    metal::uint start = gid*ELEMS_PER_GROUP; //the beginning "batch"
    metal::uint n4 = p.paramCount / 4; //total batches of 4


    for (int i = tid; i< ELEMS_PER_GROUP; i+= THREADS) {
        metal::uint idx = start + i;
        float4 gi = float4(grads4[idx]);
        float4 mi = float4(m[idx]);
        float4 vi = float4(v[idx]);
        float4 pi = params4[idx];
        vi = metal::fma(float4(p.beta2), vi, p.one_minus_beta2 * (gi * gi));
        mi = metal::fma(float4(p.beta1), mi, p.one_minus_beta1 * gi);
        float4 update = p.lr_t * mi * metal::rsqrt(vi + p.eps);
        v[idx] = bfloat4(vi);
        m[idx] = bfloat4(mi);
        params4[idx] = pi - update;
     }
}