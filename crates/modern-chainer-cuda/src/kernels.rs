use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use modern_chainer_core::{Error, Result};

const SOURCE: &str = r#"
__device__ __forceinline__ float kv_h2f(unsigned short h) {
    float f;
    asm("{ cvt.f32.f16 %0, %1; }" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ unsigned short kv_f2h(float f) {
    unsigned short h;
    asm("{ cvt.rn.f16.f32 %0, %1; }" : "=h"(h) : "f"(f));
    return h;
}
__device__ __forceinline__ float kv_b2f(unsigned short b) {
    return __uint_as_float(((unsigned int)b) << 16);
}
__device__ __forceinline__ unsigned short kv_f2b(float f) {
    return (unsigned short)(__float_as_uint(f) >> 16);
}

__device__ __forceinline__ float warp_sum_f32(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float warp_max_f32(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

__device__ float block_reduce_sum(float value, float* shared) {
    int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    float result = shared[0];
    __syncthreads();
    return result;
}

__device__ float block_reduce_max(float value, float* shared) {
    int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        __syncthreads();
    }
    float result = shared[0];
    __syncthreads();
    return result;
}


extern "C" __global__ void causal_softmax_warp_f32(float* scores, int rows, int len) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row = blockIdx.x * 4 + warp;
    if (row >= rows) return;
    int qpos = row % len;
    float* base = scores + (long long)row * len;
    float maxv = -1e30f;
    for (int j = lane; j <= qpos; j += 32) {
        if (base[j] > maxv) maxv = base[j];
    }
    maxv = warp_max_f32(maxv);
    maxv = __shfl_sync(0xffffffffu, maxv, 0);
    float sum = 0.0f;
    for (int j = lane; j <= qpos; j += 32) {
        sum += expf(base[j] - maxv);
    }
    sum = warp_sum_f32(sum);
    sum = __shfl_sync(0xffffffffu, sum, 0);
    float inv = 1.0f / sum;
    for (int j = lane; j <= qpos; j += 32) {
        base[j] = expf(base[j] - maxv) * inv;
    }
    for (int j = qpos + 1 + lane; j < len; j += 32) {
        base[j] = 0.0f;
    }
}



extern "C" __global__ void copy_i32(const int* src, int* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

extern "C" __global__ void gather_last_rows_var_f32(
    const float* x, const int* lengths, float* out, int batch, int rows, int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * dim;
    if (i < total) {
        int b = i / dim;
        int d = i % dim;
        int len = lengths[b];
        out[i] = x[(b * rows + len - 1) * dim + d];
    }
}

extern "C" __global__ void gather_last_rows_f32(
    const float* x, float* out, int batch, int rows, int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * dim;
    if (i < total) {
        int b = i / dim;
        int d = i % dim;
        out[i] = x[((b + 1) * rows - 1) * dim + d];
    }
}


__device__ __forceinline__ float flash_load_kv(const void* p, long long idx, int dtype) {
    if (dtype == 0) return ((const float*)p)[idx];
    if (dtype == 1) return kv_h2f(((const unsigned short*)p)[idx]);
    return kv_b2f(((const unsigned short*)p)[idx]);
}

extern "C" __global__ void flash_attention_prefill_f32(
    const float* q, const void* k_cache, const void* v_cache, float* out,
    int batch_heads, int prefill_len, int max_seq, int head_dim, float scale, int dtype
) {
    const int TQ = 8;
    const int TK = 32;
    extern __shared__ float smem[];
    int hd = head_dim;
    float* s_q = smem;
    float* s_k = s_q + TQ * hd;
    float* s_v = s_k + TK * hd;
    float* s_scores = s_v + TK * hd;
    float* s_out = s_scores + TQ * TK;
    float* s_m = s_out + TQ * hd;
    float* s_l = s_m + TQ;
    int tid = threadIdx.x;
    int q0 = blockIdx.x * TQ;
    int bh = blockIdx.y;
    for (int i = tid; i < TQ * hd; i += blockDim.x) {
        int qi = i / hd;
        int d = i % hd;
        int qpos = q0 + qi;
        s_q[i] = qpos < prefill_len
            ? q[((long long)bh * prefill_len + qpos) * hd + d]
            : 0.0f;
        s_out[i] = 0.0f;
    }
    for (int qi = tid; qi < TQ; qi += blockDim.x) {
        s_m[qi] = -1e30f;
        s_l[qi] = 0.0f;
    }
    __syncthreads();

    for (int kt = 0; kt < prefill_len; kt += TK) {
        for (int i = tid; i < TK * hd; i += blockDim.x) {
            int ki = i / hd;
            int d = i % hd;
            int j = kt + ki;
            if (j < prefill_len) {
                long long idx = ((long long)bh * max_seq + j) * hd + d;
                s_k[i] = flash_load_kv(k_cache, idx, dtype);
                s_v[i] = flash_load_kv(v_cache, idx, dtype);
            } else {
                s_k[i] = 0.0f;
                s_v[i] = 0.0f;
            }
        }
        __syncthreads();
        for (int i = tid; i < TQ * TK; i += blockDim.x) {
            int qi = i / TK;
            int ki = i % TK;
            int qpos = q0 + qi;
            int j = kt + ki;
            float dot = 0.0f;
            if (qpos < prefill_len && j < prefill_len && j <= qpos) {
                for (int d = 0; d < hd; d++) dot += s_q[qi * hd + d] * s_k[ki * hd + d];
                s_scores[i] = dot * scale;
            } else {
                s_scores[i] = -1e30f;
            }
        }
        __syncthreads();

        int warp = tid >> 5;
        int lane = tid & 31;
        for (int qi = warp; qi < TQ; qi += 4) {
            if (q0 + qi >= prefill_len) continue;
            float m = s_m[qi];
            float tmax = -1e30f;
            for (int ki = lane; ki < TK; ki += 32) {
                tmax = fmaxf(tmax, s_scores[qi * TK + ki]);
            }
            tmax = warp_max_f32(tmax);
            tmax = __shfl_sync(0xffffffffu, tmax, 0);
            float mnew = fmaxf(m, tmax);
            float alpha = __expf(m - mnew);
            float sum = 0.0f;
            for (int ki = lane; ki < TK; ki += 32) {
                float w = __expf(s_scores[qi * TK + ki] - mnew);
                s_scores[qi * TK + ki] = w;
                sum += w;
            }
            sum = warp_sum_f32(sum);
            sum = __shfl_sync(0xffffffffu, sum, 0);
            __syncwarp();
            float l = s_l[qi] * alpha + sum;
            for (int d = lane; d < hd; d += 32) {
                float acc = 0.0f;
                for (int ki = 0; ki < TK; ki++) {
                    acc += s_scores[qi * TK + ki] * s_v[ki * hd + d];
                }
                s_out[qi * hd + d] = s_out[qi * hd + d] * alpha + acc;
            }
            if (lane == 0) {
                s_m[qi] = mnew;
                s_l[qi] = l;
            }
        }
        __syncthreads();
    }

    for (int i = tid; i < TQ * hd; i += blockDim.x) {
        int qi = i / hd;
        int qpos = q0 + qi;
        if (qpos < prefill_len) {
            int d = i % hd;
            out[((long long)bh * prefill_len + qpos) * hd + d] = s_out[i] / s_l[qi];
        }
    }
}

extern "C" __global__ void flash_attention_decode_lowp_f32(
    const float* q, const void* k_cache, const void* v_cache,
    const int* positions, float* out,
    int batch, int heads, int max_seq, int head_dim, float scale, int dtype
) {
    extern __shared__ float smem[];
    int hd = head_dim;
    const int W = 8;
    float* s_acc = smem;
    float* s_m = s_acc + W * hd;
    float* s_l = s_m + W;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int idx = blockIdx.x;
    int head = idx % heads;
    int b = idx / heads;
    int n = positions[b] + 1;
    if (n > max_seq) n = max_seq;
    int q_base = (b * heads + head) * hd;
    long long cache_base = (long long)(b * heads + head) * max_seq * hd;
    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.0f;
    float m = -1e30f;
    float l = 0.0f;
    for (int p = warp; p < n; p += W) {
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int d = lane + i * 32;
            if (d < hd) dot += q[q_base + d] * flash_load_kv(k_cache, cache_base + (long long)p * hd + d, dtype);
        }
        dot = warp_sum_f32(dot);
        dot = __shfl_sync(0xffffffffu, dot, 0) * scale;
        float mnew = fmaxf(m, dot);
        float alpha = __expf(m - mnew);
        float weight = __expf(dot - mnew);
        l = l * alpha + weight;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int d = lane + i * 32;
            if (d < hd) {
                acc[i] = acc[i] * alpha + weight * flash_load_kv(v_cache, cache_base + (long long)p * hd + d, dtype);
            }
        }
        m = mnew;
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int d = lane + i * 32;
        if (d < hd) s_acc[warp * hd + d] = acc[i];
    }
    if (lane == 0) {
        s_m[warp] = m;
        s_l[warp] = l;
    }
    __syncthreads();
    float gm = -1e30f;
    for (int w = 0; w < W; w++) gm = fmaxf(gm, s_m[w]);
    for (int d = tid; d < hd; d += blockDim.x) {
        float sum = 0.0f;
        float denom = 0.0f;
        for (int w = 0; w < W; w++) {
            float s = __expf(s_m[w] - gm);
            sum += s_acc[w * hd + d] * s;
            denom += s_l[w] * s;
        }
        out[q_base + d] = sum / denom;
    }
}

extern "C" __global__ void split_heads_decode_f32(
    const float* q, float* qh, int batch, int heads, int head_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * head_dim;
    if (i < total) {
        int d = i % head_dim;
        int t = i / head_dim;
        int head = t % heads;
        int b = t / heads;
        qh[i] = q[(b * heads + head) * head_dim + d];
    }
}

extern "C" __global__ void merge_heads_decode_f32(
    const float* attn, float* out, int batch, int heads, int head_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * head_dim;
    if (i < total) {
        int d = i % head_dim;
        int t = i / head_dim;
        int head = t % heads;
        int b = t / heads;
        out[(b * heads + head) * head_dim + d] = attn[i];
    }
}

extern "C" __global__ void decode_softmax_f32(
    float* scores, const int* positions, int batch, int heads, int max_seq
) {
    __shared__ float shared[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int b = row / heads;
    int n = positions[b] + 1;
    if (n > max_seq) n = max_seq;
    float* base = scores + (long long)row * max_seq;
    float maxv = -1e30f;
    for (int j = tid; j < n; j += blockDim.x) {
        if (base[j] > maxv) maxv = base[j];
    }
    maxv = block_reduce_max(maxv, shared);
    float sum = 0.0f;
    for (int j = tid; j < n; j += blockDim.x) {
        sum += expf(base[j] - maxv);
    }
    sum = block_reduce_sum(sum, shared);
    float inv = 1.0f / sum;
    for (int j = tid; j < n; j += blockDim.x) {
        base[j] = expf(base[j] - maxv) * inv;
    }
    for (int j = n + tid; j < max_seq; j += blockDim.x) {
        base[j] = 0.0f;
    }
}




extern "C" __global__ void split_heads_prefill_f32(
    const float* q, float* qh, int batch, int prefill_len, int heads, int head_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * prefill_len * heads * head_dim;
    if (i < total) {
        int d = i % head_dim;
        int t = i / head_dim;
        int head = t % heads;
        int t2 = t / heads;
        int pos = t2 % prefill_len;
        int b = t2 / prefill_len;
        int src = ((b * prefill_len + pos) * heads + head) * head_dim + d;
        int dst = ((b * heads + head) * prefill_len + pos) * head_dim + d;
        qh[dst] = q[src];
    }
}

extern "C" __global__ void merge_heads_prefill_f32(
    const float* attn, float* out, int batch, int prefill_len, int heads, int head_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * prefill_len * heads * head_dim;
    if (i < total) {
        int d = i % head_dim;
        int t = i / head_dim;
        int head = t % heads;
        int t2 = t / heads;
        int pos = t2 % prefill_len;
        int b = t2 / prefill_len;
        int src = ((b * heads + head) * prefill_len + pos) * head_dim + d;
        int dst = ((b * prefill_len + pos) * heads + head) * head_dim + d;
        out[dst] = attn[src];
    }
}






extern "C" __global__ void attention_batch_decode_wide_f32(
    const float* q, const float* k_cache, const float* v_cache,
    const int* positions, float* out,
    int batch, int heads, int max_seq, int head_dim, float scale
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int idx = blockIdx.x * 4 + warp;
    int total = batch * heads;
    if (idx >= total) return;
    int head = idx % heads;
    int b = idx / heads;
    int n = positions[b] + 1;
    if (n > max_seq) n = max_seq;
    int q_base = (b * heads + head) * head_dim;
    int cache_base = (b * heads + head) * max_seq * head_dim;
    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.0f;
    float m = -1e30f;
    float l = 0.0f;
    for (int p = 0; p < n; p++) {
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int d = lane + i * 32;
            if (d < head_dim) dot += q[q_base + d] * k_cache[cache_base + p * head_dim + d];
        }
        dot = warp_sum_f32(dot);
        dot = __shfl_sync(0xffffffffu, dot, 0) * scale;
        float mnew = fmaxf(m, dot);
        float alpha = __expf(m - mnew);
        float weight = __expf(dot - mnew);
        l = l * alpha + weight;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int d = lane + i * 32;
            if (d < head_dim) {
                acc[i] = acc[i] * alpha + weight * v_cache[cache_base + p * head_dim + d];
            }
        }
        m = mnew;
    }
    float inv = 1.0f / l;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int d = lane + i * 32;
        if (d < head_dim) out[q_base + d] = acc[i] * inv;
    }
}

extern "C" __global__ void attention_batch_decode_wide_f16_f32(
    const float* q, const unsigned short* k_cache, const unsigned short* v_cache,
    const int* positions, float* out,
    int batch, int heads, int max_seq, int head_dim, float scale
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int idx = blockIdx.x * 4 + warp;
    int total = batch * heads;
    if (idx >= total) return;
    int head = idx % heads;
    int b = idx / heads;
    int n = positions[b] + 1;
    if (n > max_seq) n = max_seq;
    int q_base = (b * heads + head) * head_dim;
    int cache_base = (b * heads + head) * max_seq * head_dim;
    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.0f;
    float m = -1e30f;
    float l = 0.0f;
    for (int p = 0; p < n; p++) {
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int d = lane + i * 32;
            if (d < head_dim) dot += q[q_base + d] * kv_h2f(k_cache[cache_base + p * head_dim + d]);
        }
        dot = warp_sum_f32(dot);
        dot = __shfl_sync(0xffffffffu, dot, 0) * scale;
        float mnew = fmaxf(m, dot);
        float alpha = __expf(m - mnew);
        float weight = __expf(dot - mnew);
        l = l * alpha + weight;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int d = lane + i * 32;
            if (d < head_dim) {
                acc[i] = acc[i] * alpha + weight * kv_h2f(v_cache[cache_base + p * head_dim + d]);
            }
        }
        m = mnew;
    }
    float inv = 1.0f / l;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int d = lane + i * 32;
        if (d < head_dim) out[q_base + d] = acc[i] * inv;
    }
}

extern "C" __global__ void attention_batch_decode_wide_bf16_f32(
    const float* q, const unsigned short* k_cache, const unsigned short* v_cache,
    const int* positions, float* out,
    int batch, int heads, int max_seq, int head_dim, float scale
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int idx = blockIdx.x * 4 + warp;
    int total = batch * heads;
    if (idx >= total) return;
    int head = idx % heads;
    int b = idx / heads;
    int n = positions[b] + 1;
    if (n > max_seq) n = max_seq;
    int q_base = (b * heads + head) * head_dim;
    int cache_base = (b * heads + head) * max_seq * head_dim;
    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.0f;
    float m = -1e30f;
    float l = 0.0f;
    for (int p = 0; p < n; p++) {
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int d = lane + i * 32;
            if (d < head_dim) dot += q[q_base + d] * kv_b2f(k_cache[cache_base + p * head_dim + d]);
        }
        dot = warp_sum_f32(dot);
        dot = __shfl_sync(0xffffffffu, dot, 0) * scale;
        float mnew = fmaxf(m, dot);
        float alpha = __expf(m - mnew);
        float weight = __expf(dot - mnew);
        l = l * alpha + weight;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int d = lane + i * 32;
            if (d < head_dim) {
                acc[i] = acc[i] * alpha + weight * kv_b2f(v_cache[cache_base + p * head_dim + d]);
            }
        }
        m = mnew;
    }
    float inv = 1.0f / l;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int d = lane + i * 32;
        if (d < head_dim) out[q_base + d] = acc[i] * inv;
    }
}


extern "C" __global__ void add_f32(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
extern "C" __global__ void sub_f32(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] - b[i];
}
extern "C" __global__ void mul_f32(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}
extern "C" __global__ void div_f32(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] / b[i];
}
extern "C" __global__ void sqrt_f32(const float* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = sqrtf(a[i]);
}
extern "C" __global__ void relu_f32(const float* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fmaxf(a[i], 0.0f);
}
extern "C" __global__ void relu_backward_f32(const float* x, const float* g, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] > 0.0f ? g[i] : 0.0f;
}
extern "C" __global__ void silu_f32(const float* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] / (1.0f + expf(-a[i]));
}
extern "C" __global__ void scale_f32(const float* a, float* out, float factor, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * factor;
}
extern "C" __global__ void fill_f32(float* out, float value, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = value;
}
extern "C" __global__ void bias_add_f32(float* out, const float* bias, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (i < total) out[i] += bias[i % cols];
}
extern "C" __global__ void transpose2d_f32(const float* a, float* out, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (i < total) {
        int row = i / cols;
        int col = i % cols;
        out[col * rows + row] = a[i];
    }
}
extern "C" __global__ void adam_update_f32(
    const float* p, const float* g, const float* m, const float* v,
    float* np, float* nm, float* nv,
    int n, float lr, float b1, float b2, float eps, float bc1, float bc2
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float gi = g[i];
        float nmi = b1 * m[i] + (1.0f - b1) * gi;
        float nvi = b2 * v[i] + (1.0f - b2) * gi * gi;
        float mhat = nmi / bc1;
        float vhat = nvi / bc2;
        nm[i] = nmi;
        nv[i] = nvi;
        np[i] = p[i] - lr * mhat / (sqrtf(vhat) + eps);
    }
}




extern "C" __global__ void causal_softmax_f32(float* scores, int rows, int len) {
    __shared__ float shared[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int qpos = row % len;
    float* base = scores + (long long)row * len;
    float maxv = -1e30f;
    for (int j = tid; j <= qpos; j += blockDim.x) {
        if (base[j] > maxv) maxv = base[j];
    }
    maxv = block_reduce_max(maxv, shared);
    float sum = 0.0f;
    for (int j = tid; j <= qpos; j += blockDim.x) {
        sum += expf(base[j] - maxv);
    }
    sum = block_reduce_sum(sum, shared);
    float inv = 1.0f / sum;
    for (int j = tid; j <= qpos; j += blockDim.x) {
        base[j] = expf(base[j] - maxv) * inv;
    }
    for (int j = qpos + 1 + tid; j < len; j += blockDim.x) {
        base[j] = 0.0f;
    }
}


extern "C" __global__ void sum_all_f32(const float* a, float* out, int n) {
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        sum += a[i];
    }
    atomicAdd(out, sum);
}

extern "C" __global__ void embedding_f32(const long long* ids, const float* weight, float* out, int rows, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * dim;
    if (i < total) {
        int row = i / dim;
        int d = i % dim;
        long long id = ids[row];
        out[i] = weight[id * dim + d];
    }
}

extern "C" __global__ void layer_norm_f32(
    const float* x, const float* gamma, const float* beta, float* out,
    int rows, int d, float eps
) {
    __shared__ float shared[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + row * d;
    float sum = 0.0f;
    float sumsq = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        float v = xr[i];
        sum += v;
        sumsq += v * v;
    }
    float total = block_reduce_sum(sum, shared);
    float mean = total / d;
    float total_sq = block_reduce_sum(sumsq, shared);
    float var = total_sq / d - mean * mean;
    float inv = rsqrtf(var + eps);
    for (int i = tid; i < d; i += blockDim.x) {
        out[row * d + i] = (xr[i] - mean) * inv * gamma[i] + beta[i];
    }
}

extern "C" __global__ void softmax_last_dim_f32(const float* x, float* out, int rows, int cols) {
    __shared__ float shared[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + row * cols;
    float maxv = -1e30f;
    for (int i = tid; i < cols; i += blockDim.x) maxv = fmaxf(maxv, xr[i]);
    maxv = block_reduce_max(maxv, shared);
    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) sum += expf(xr[i] - maxv);
    sum = block_reduce_sum(sum, shared);
    float inv = 1.0f / sum;
    for (int i = tid; i < cols; i += blockDim.x) {
        out[row * cols + i] = expf(xr[i] - maxv) * inv;
    }
}


extern "C" __global__ void embedding_backward_f32(
    const long long* ids, const float* grad, float* out, int n, int dim, int vocab
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vocab * dim) return;
    int v = i / dim;
    int d = i % dim;
    float acc = 0.0f;
    for (int r = 0; r < n; r++) {
        if (ids[r] == v) acc += grad[r * dim + d];
    }
    out[i] = acc;
}

extern "C" __global__ void layer_norm_backward_dx_f32(
    const float* x, const float* gamma, const float* grad, float* out,
    int rows, int d, float eps
) {
    __shared__ float shared[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + row * d;
    const float* gr = grad + row * d;
    float sum = 0.0f;
    float sumsq = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        float v = xr[i];
        sum += v;
        sumsq += v * v;
    }
    float mean = block_reduce_sum(sum, shared) / d;
    float var = block_reduce_sum(sumsq, shared) / d - mean * mean;
    float inv = rsqrtf(var + eps);
    float sum_dy = 0.0f;
    float sum_dyn = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        float norm = (xr[i] - mean) * inv;
        float dy = gr[i] * gamma[i];
        sum_dy += dy;
        sum_dyn += dy * norm;
    }
    float total_dy = block_reduce_sum(sum_dy, shared);
    float total_dyn = block_reduce_sum(sum_dyn, shared);
    for (int i = tid; i < d; i += blockDim.x) {
        float norm = (xr[i] - mean) * inv;
        float dy = gr[i] * gamma[i];
        out[row * d + i] = (d * dy - total_dy - norm * total_dyn) * inv / d;
    }
}

extern "C" __global__ void layer_norm_backward_params_f32(
    const float* x, const float* grad, float* dgamma, float* dbeta,
    int rows, int d, float eps
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= d) return;
    float dg = 0.0f;
    float db = 0.0f;
    for (int r = 0; r < rows; r++) {
        const float* xr = x + r * d;
        float sum = 0.0f;
        float sq = 0.0f;
        for (int i = 0; i < d; i++) {
            float v = xr[i];
            sum += v;
            sq += v * v;
        }
        float mean = sum / d;
        float var = sq / d - mean * mean;
        float norm = (xr[c] - mean) * rsqrtf(var + eps);
        float g = grad[r * d + c];
        dg += g * norm;
        db += g;
    }
    dgamma[c] = dg;
    dbeta[c] = db;
}

extern "C" __global__ void softmax_backward_f32(
    const float* x, const float* grad, float* out, int rows, int cols
) {
    __shared__ float shared[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + row * cols;
    const float* gr = grad + row * cols;
    float maxv = -1e30f;
    for (int i = tid; i < cols; i += blockDim.x) maxv = fmaxf(maxv, xr[i]);
    maxv = block_reduce_max(maxv, shared);
    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) sum += expf(xr[i] - maxv);
    sum = block_reduce_sum(sum, shared);
    float inv = 1.0f / sum;
    float dot = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        dot += expf(xr[i] - maxv) * inv * gr[i];
    }
    dot = block_reduce_sum(dot, shared);
    for (int i = tid; i < cols; i += blockDim.x) {
        float p = expf(xr[i] - maxv) * inv;
        out[row * cols + i] = p * (gr[i] - dot);
    }
}

extern "C" __global__ void split_heads_f32(const float* x, float* out, int seq, int heads, int head_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * heads * head_dim;
    if (i < total) {
        int s = i / (heads * head_dim);
        int rem = i % (heads * head_dim);
        int h = rem / head_dim;
        int d = rem % head_dim;
        out[(h * seq + s) * head_dim + d] = x[i];
    }
}

extern "C" __global__ void merge_heads_f32(const float* x, float* out, int heads, int seq, int head_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = heads * seq * head_dim;
    if (i < total) {
        int h = i / (seq * head_dim);
        int rem = i % (seq * head_dim);
        int s = rem / head_dim;
        int d = rem % head_dim;
        out[(s * heads + h) * head_dim + d] = x[i];
    }
}

extern "C" __global__ void concat_seq_f32(
    const float* a, const float* b, float* out,
    int heads, int s1, int s2, int head_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = heads * (s1 + s2) * head_dim;
    if (i < total) {
        int h = i / ((s1 + s2) * head_dim);
        int rem = i % ((s1 + s2) * head_dim);
        int s = rem / head_dim;
        int d = rem % head_dim;
        out[i] = s < s1
            ? a[(h * s1 + s) * head_dim + d]
            : b[(h * s2 + (s - s1)) * head_dim + d];
    }
}

extern "C" __global__ void last_row_f32(const float* x, float* out, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cols) out[i] = x[(rows - 1) * cols + i];
}

extern "C" __global__ void attention_f32(
    const float* q, const float* k, const float* v, float* out,
    int heads, int q_seq, int kv_seq, int head_dim, float scale, int causal
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = heads * q_seq;
    if (idx >= total) return;
    int h = idx / q_seq;
    int i = idx % q_seq;
    int offset = kv_seq - q_seq;
    int q_base = (h * q_seq + i) * head_dim;
    int kv_base = h * kv_seq * head_dim;

    float maxv = -1e30f;
    for (int j = 0; j < kv_seq; j++) {
        if (causal && j > offset + i) continue;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * k[kv_base + j * head_dim + d];
        dot *= scale;
        if (dot > maxv) maxv = dot;
    }
    float sum = 0.0f;
    for (int j = 0; j < kv_seq; j++) {
        if (causal && j > offset + i) continue;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * k[kv_base + j * head_dim + d];
        sum += expf(dot * scale - maxv);
    }
    float inv = 1.0f / sum;
    for (int d = 0; d < head_dim; d++) out[q_base + d] = 0.0f;
    for (int j = 0; j < kv_seq; j++) {
        if (causal && j > offset + i) continue;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * k[kv_base + j * head_dim + d];
        float p = expf(dot * scale - maxv) * inv;
        for (int d = 0; d < head_dim; d++) out[q_base + d] += p * v[kv_base + j * head_dim + d];
    }
}

extern "C" __global__ void paged_attention_f32(
    const float* q, const float* k_blocks, const float* v_blocks,
    const long long* table, float* out,
    int heads, int q_seq, int seq_len, int head_dim, int block_size, int num_blocks,
    float scale, int causal
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = heads * q_seq;
    if (idx >= total) return;
    int h = idx / q_seq;
    int i = idx % q_seq;
    int offset = seq_len - q_seq;
    int q_base = (h * q_seq + i) * head_dim;

    float maxv = -1e30f;
    for (int p = 0; p < seq_len; p++) {
        if (causal && p > offset + i) continue;
        int logical = p / block_size;
        int physical = (int)table[logical];
        if (physical < 0 || physical >= num_blocks) return;
        int within = p % block_size;
        int base = ((physical * block_size + within) * heads + h) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * k_blocks[base + d];
        dot *= scale;
        if (dot > maxv) maxv = dot;
    }
    float sum = 0.0f;
    for (int p = 0; p < seq_len; p++) {
        if (causal && p > offset + i) continue;
        int logical = p / block_size;
        int physical = (int)table[logical];
        if (physical < 0 || physical >= num_blocks) return;
        int within = p % block_size;
        int base = ((physical * block_size + within) * heads + h) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * k_blocks[base + d];
        sum += expf(dot * scale - maxv);
    }
    float inv = 1.0f / sum;
    for (int d = 0; d < head_dim; d++) out[q_base + d] = 0.0f;
    for (int p = 0; p < seq_len; p++) {
        if (causal && p > offset + i) continue;
        int logical = p / block_size;
        int physical = (int)table[logical];
        if (physical < 0 || physical >= num_blocks) return;
        int within = p % block_size;
        int base = ((physical * block_size + within) * heads + h) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * k_blocks[base + d];
        float prob = expf(dot * scale - maxv) * inv;
        for (int d = 0; d < head_dim; d++) out[q_base + d] += prob * v_blocks[base + d];
    }
}

extern "C" __global__ void paged_attention_batch_f32(
    const float* q, const float* k_blocks, const float* v_blocks,
    const long long* block_tables, const long long* seq_lens, float* out,
    int batch, int heads, int q_seq, int head_dim, int block_size,
    int num_blocks, int max_blocks, float scale, int causal
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * q_seq;
    if (idx >= total) return;
    int b = idx / (heads * q_seq);
    int rem = idx % (heads * q_seq);
    int h = rem / q_seq;
    int i = rem % q_seq;
    int seq_len = (int)seq_lens[b];
    if (seq_len <= 0 || q_seq > seq_len) return;
    int offset = seq_len - q_seq;
    int q_base = ((b * heads + h) * q_seq + i) * head_dim;
    const long long* table = block_tables + b * max_blocks;

    float maxv = -1e30f;
    for (int p = 0; p < seq_len; p++) {
        if (causal && p > offset + i) continue;
        int logical = p / block_size;
        int physical = (int)table[logical];
        if (physical < 0 || physical >= num_blocks) return;
        int within = p % block_size;
        int base = ((physical * block_size + within) * heads + h) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * k_blocks[base + d];
        dot *= scale;
        if (dot > maxv) maxv = dot;
    }
    float sum = 0.0f;
    for (int p = 0; p < seq_len; p++) {
        if (causal && p > offset + i) continue;
        int logical = p / block_size;
        int physical = (int)table[logical];
        if (physical < 0 || physical >= num_blocks) return;
        int within = p % block_size;
        int base = ((physical * block_size + within) * heads + h) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * k_blocks[base + d];
        sum += expf(dot * scale - maxv);
    }
    float inv = 1.0f / sum;
    for (int d = 0; d < head_dim; d++) out[q_base + d] = 0.0f;
    for (int p = 0; p < seq_len; p++) {
        if (causal && p > offset + i) continue;
        int logical = p / block_size;
        int physical = (int)table[logical];
        if (physical < 0 || physical >= num_blocks) return;
        int within = p % block_size;
        int base = ((physical * block_size + within) * heads + h) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * k_blocks[base + d];
        float prob = expf(dot * scale - maxv) * inv;
        for (int d = 0; d < head_dim; d++) out[q_base + d] += prob * v_blocks[base + d];
    }
}

extern "C" __global__ void paged_append_f32(
    const float* k_new, const float* v_new, const long long* table,
    float* k_out, float* v_out,
    int len, int n, int heads, int head_dim, int block_size, int num_blocks
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * heads * head_dim;
    if (idx >= total) return;
    int i = idx / (heads * head_dim);
    int rem = idx % (heads * head_dim);
    int h = rem / head_dim;
    int d = rem % head_dim;
    int position = len + i;
    int logical = position / block_size;
    int physical = (int)table[logical];
    if (physical < 0 || physical >= num_blocks) return;
    int within = position % block_size;
    int dst = ((physical * block_size + within) * heads + h) * head_dim + d;
    int src = (h * n + i) * head_dim + d;
    k_out[dst] = k_new[src];
    v_out[dst] = v_new[src];
}

extern "C" __global__ void softmax_ce_f32(
    const float* logits, const long long* target, float* out, int rows, int cols
) {
    __shared__ float shared[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = logits + row * cols;
    long long label = target[row];
    if (label < 0 || label >= cols) return;
    float maxv = -1e30f;
    for (int i = tid; i < cols; i += blockDim.x) maxv = fmaxf(maxv, xr[i]);
    maxv = block_reduce_max(maxv, shared);
    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) sum += expf(xr[i] - maxv);
    sum = block_reduce_sum(sum, shared);
    if (tid == 0) {
        float loss = (maxv + logf(sum)) - xr[label];
        atomicAdd(out, loss / rows);
    }
}

extern "C" __global__ void softmax_ce_backward_f32(
    const float* logits, const long long* target, float* out, int rows, int cols
) {
    __shared__ float shared[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = logits + row * cols;
    long long label = target[row];
    if (label < 0 || label >= cols) return;
    float maxv = -1e30f;
    for (int i = tid; i < cols; i += blockDim.x) maxv = fmaxf(maxv, xr[i]);
    maxv = block_reduce_max(maxv, shared);
    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) sum += expf(xr[i] - maxv);
    sum = block_reduce_sum(sum, shared);
    float inv = 1.0f / sum;
    float scale = 1.0f / rows;
    for (int i = tid; i < cols; i += blockDim.x) {
        float p = expf(xr[i] - maxv) * inv;
        float onehot = (i == label) ? 1.0f : 0.0f;
        out[row * cols + i] = (p - onehot) * scale;
    }
}

extern "C" __global__ void sum_to_bias_f32(const float* a, float* out, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int row = 0; row < rows; row++) sum += a[row * cols + col];
        out[col] = sum;
    }
}

extern "C" __global__ void gelu_f32(const float* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i];
        out[i] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}
extern "C" __global__ void exp_f32(const float* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = expf(a[i]);
}
extern "C" __global__ void log_f32(const float* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = logf(a[i]);
}
extern "C" __global__ void sigmoid_f32(const float* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1.0f / (1.0f + expf(-a[i]));
}
extern "C" __global__ void clamp_f32(const float* a, float* out, float minv, float maxv, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fminf(fmaxf(a[i], minv), maxv);
}
extern "C" __global__ void abs_f32(const float* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fabsf(a[i]);
}

extern "C" __global__ void paged_attention_wide_f32(
    const float* q, const float* k_blocks, const float* v_blocks,
    const long long* table, float* out,
    int heads, int q_seq, int seq_len, int head_dim, int block_size, int num_blocks,
    float scale, int causal
) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    float* s_max = shared;
    float* s_sum = s_max + blockDim.x;
    float* s_out = s_sum + blockDim.x;

    int query = blockIdx.x;
    int h = query / q_seq;
    int i = query % q_seq;
    int offset = seq_len - q_seq;
    int q_base = (h * q_seq + i) * head_dim;

    for (int d = tid; d < head_dim; d += blockDim.x) s_out[d] = 0.0f;
    __syncthreads();

    float local_max = -1e30f;
    for (int p = tid; p < seq_len; p += blockDim.x) {
        if (causal && p > offset + i) continue;
        int logical = p / block_size;
        int physical = (int)table[logical];
        if (physical < 0 || physical >= num_blocks) continue;
        int within = p % block_size;
        int base = ((physical * block_size + within) * heads + h) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * k_blocks[base + d];
        local_max = fmaxf(local_max, dot * scale);
    }
    s_max[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        __syncthreads();
    }
    float maxv = s_max[0];

    float local_sum = 0.0f;
    for (int p = tid; p < seq_len; p += blockDim.x) {
        if (causal && p > offset + i) continue;
        int logical = p / block_size;
        int physical = (int)table[logical];
        if (physical < 0 || physical >= num_blocks) continue;
        int within = p % block_size;
        int base = ((physical * block_size + within) * heads + h) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * k_blocks[base + d];
        local_sum += expf(dot * scale - maxv);
    }
    s_sum[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_sum[tid] += s_sum[tid + stride];
        __syncthreads();
    }
    float inv = 1.0f / s_sum[0];

    for (int p = tid; p < seq_len; p += blockDim.x) {
        if (causal && p > offset + i) continue;
        int logical = p / block_size;
        int physical = (int)table[logical];
        if (physical < 0 || physical >= num_blocks) continue;
        int within = p % block_size;
        int base = ((physical * block_size + within) * heads + h) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * k_blocks[base + d];
        float prob = expf(dot * scale - maxv) * inv;
        for (int d = 0; d < head_dim; d++) atomicAdd(&s_out[d], prob * v_blocks[base + d]);
    }
    __syncthreads();
    for (int d = tid; d < head_dim; d += blockDim.x) out[q_base + d] = s_out[d];
}

extern "C" __global__ void paged_attention_batch_wide_f32(
    const float* q, const float* k_blocks, const float* v_blocks,
    const long long* block_tables, const long long* seq_lens, float* out,
    int batch, int heads, int q_seq, int head_dim, int block_size,
    int num_blocks, int max_blocks, float scale, int causal
) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    float* s_max = shared;
    float* s_sum = s_max + blockDim.x;
    float* s_out = s_sum + blockDim.x;

    int item = blockIdx.x;
    int b = item / (heads * q_seq);
    int rem = item % (heads * q_seq);
    int h = rem / q_seq;
    int i = rem % q_seq;
    int seq_len = (int)seq_lens[b];
    if (seq_len <= 0 || q_seq > seq_len) return;
    int offset = seq_len - q_seq;
    int q_base = ((b * heads + h) * q_seq + i) * head_dim;
    const long long* table = block_tables + b * max_blocks;

    for (int d = tid; d < head_dim; d += blockDim.x) s_out[d] = 0.0f;
    __syncthreads();

    float local_max = -1e30f;
    for (int p = tid; p < seq_len; p += blockDim.x) {
        if (causal && p > offset + i) continue;
        int logical = p / block_size;
        int physical = (int)table[logical];
        if (physical < 0 || physical >= num_blocks) continue;
        int within = p % block_size;
        int base = ((physical * block_size + within) * heads + h) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * k_blocks[base + d];
        local_max = fmaxf(local_max, dot * scale);
    }
    s_max[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        __syncthreads();
    }
    float maxv = s_max[0];

    float local_sum = 0.0f;
    for (int p = tid; p < seq_len; p += blockDim.x) {
        if (causal && p > offset + i) continue;
        int logical = p / block_size;
        int physical = (int)table[logical];
        if (physical < 0 || physical >= num_blocks) continue;
        int within = p % block_size;
        int base = ((physical * block_size + within) * heads + h) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * k_blocks[base + d];
        local_sum += expf(dot * scale - maxv);
    }
    s_sum[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_sum[tid] += s_sum[tid + stride];
        __syncthreads();
    }
    float inv = 1.0f / s_sum[0];

    for (int p = tid; p < seq_len; p += blockDim.x) {
        if (causal && p > offset + i) continue;
        int logical = p / block_size;
        int physical = (int)table[logical];
        if (physical < 0 || physical >= num_blocks) continue;
        int within = p % block_size;
        int base = ((physical * block_size + within) * heads + h) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * k_blocks[base + d];
        float prob = expf(dot * scale - maxv) * inv;
        for (int d = 0; d < head_dim; d++) atomicAdd(&s_out[d], prob * v_blocks[base + d]);
    }
    __syncthreads();
    for (int d = tid; d < head_dim; d += blockDim.x) out[q_base + d] = s_out[d];
}


extern "C" __global__ void kv_prefill_f16_f32(
    const float* k, const float* v, unsigned short* k_cache, unsigned short* v_cache,
    int batch, int prefill_len, int heads, int head_dim, int max_seq
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * prefill_len * heads * head_dim;
    if (i < total) {
        int d = i % head_dim;
        int t = i / head_dim;
        int head = t % heads;
        int t2 = t / heads;
        int pos = t2 % prefill_len;
        int b = t2 / prefill_len;
        int src = ((b * prefill_len + pos) * heads + head) * head_dim + d;
        int dst = ((b * heads + head) * max_seq + pos) * head_dim + d;
        k_cache[dst] = kv_f2h(k[src]);
        v_cache[dst] = kv_f2h(v[src]);
    }
}

extern "C" __global__ void kv_prefill_bf16_f32(
    const float* k, const float* v, unsigned short* k_cache, unsigned short* v_cache,
    int batch, int prefill_len, int heads, int head_dim, int max_seq
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * prefill_len * heads * head_dim;
    if (i < total) {
        int d = i % head_dim;
        int t = i / head_dim;
        int head = t % heads;
        int t2 = t / heads;
        int pos = t2 % prefill_len;
        int b = t2 / prefill_len;
        int src = ((b * prefill_len + pos) * heads + head) * head_dim + d;
        int dst = ((b * heads + head) * max_seq + pos) * head_dim + d;
        k_cache[dst] = kv_f2b(k[src]);
        v_cache[dst] = kv_f2b(v[src]);
    }
}

extern "C" __global__ void kv_append_batch_f16_f32(
    const float* k, const float* v, unsigned short* k_cache, unsigned short* v_cache,
    const int* positions, int batch, int heads, int head_dim, int max_seq
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * head_dim;
    if (i < total) {
        int d = i % head_dim;
        int t = i / head_dim;
        int head = t % heads;
        int b = t / heads;
        int pos = positions[b];
        if (pos < max_seq) {
            int src = (b * heads + head) * head_dim + d;
            int dst = ((b * heads + head) * max_seq + pos) * head_dim + d;
            k_cache[dst] = kv_f2h(k[src]);
            v_cache[dst] = kv_f2h(v[src]);
        }
    }
}

extern "C" __global__ void kv_append_batch_bf16_f32(
    const float* k, const float* v, unsigned short* k_cache, unsigned short* v_cache,
    const int* positions, int batch, int heads, int head_dim, int max_seq
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * head_dim;
    if (i < total) {
        int d = i % head_dim;
        int t = i / head_dim;
        int head = t % heads;
        int b = t / heads;
        int pos = positions[b];
        if (pos < max_seq) {
            int src = (b * heads + head) * head_dim + d;
            int dst = ((b * heads + head) * max_seq + pos) * head_dim + d;
            k_cache[dst] = kv_f2b(k[src]);
            v_cache[dst] = kv_f2b(v[src]);
        }
    }
}

extern "C" __global__ void attention_batch_decode_f16_f32(
    const float* q, const unsigned short* k_cache, const unsigned short* v_cache,
    const int* positions, float* out,
    int batch, int heads, int max_seq, int head_dim, float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads;
    if (idx >= total) return;
    int head = idx % heads;
    int b = idx / heads;
    int n = positions[b] + 1;
    if (n > max_seq) n = max_seq;
    int q_base = (b * heads + head) * head_dim;
    int cache_base = (b * heads + head) * max_seq * head_dim;
    float maxv = -1e30f;
    for (int p = 0; p < n; p++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * kv_h2f(k_cache[cache_base + p * head_dim + d]);
        dot *= scale;
        if (dot > maxv) maxv = dot;
    }
    float sum = 0.0f;
    for (int p = 0; p < n; p++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * kv_h2f(k_cache[cache_base + p * head_dim + d]);
        sum += expf(dot * scale - maxv);
    }
    float inv = 1.0f / sum;
    for (int d = 0; d < head_dim; d++) out[q_base + d] = 0.0f;
    for (int p = 0; p < n; p++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * kv_h2f(k_cache[cache_base + p * head_dim + d]);
        float prob = expf(dot * scale - maxv) * inv;
        for (int d = 0; d < head_dim; d++) out[q_base + d] += prob * kv_h2f(v_cache[cache_base + p * head_dim + d]);
    }
}

extern "C" __global__ void attention_batch_decode_bf16_f32(
    const float* q, const unsigned short* k_cache, const unsigned short* v_cache,
    const int* positions, float* out,
    int batch, int heads, int max_seq, int head_dim, float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads;
    if (idx >= total) return;
    int head = idx % heads;
    int b = idx / heads;
    int n = positions[b] + 1;
    if (n > max_seq) n = max_seq;
    int q_base = (b * heads + head) * head_dim;
    int cache_base = (b * heads + head) * max_seq * head_dim;
    float maxv = -1e30f;
    for (int p = 0; p < n; p++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * kv_b2f(k_cache[cache_base + p * head_dim + d]);
        dot *= scale;
        if (dot > maxv) maxv = dot;
    }
    float sum = 0.0f;
    for (int p = 0; p < n; p++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * kv_b2f(k_cache[cache_base + p * head_dim + d]);
        sum += expf(dot * scale - maxv);
    }
    float inv = 1.0f / sum;
    for (int d = 0; d < head_dim; d++) out[q_base + d] = 0.0f;
    for (int p = 0; p < n; p++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * kv_b2f(k_cache[cache_base + p * head_dim + d]);
        float prob = expf(dot * scale - maxv) * inv;
        for (int d = 0; d < head_dim; d++) out[q_base + d] += prob * kv_b2f(v_cache[cache_base + p * head_dim + d]);
    }
}

extern "C" __global__ void kv_cache_to_f32_f16(
    const unsigned short* k_cache, const unsigned short* v_cache, float* k, float* v,
    int batch, int heads, int prefill_len, int max_seq, int head_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * prefill_len * head_dim;
    if (i < total) {
        int d = i % head_dim;
        int t = i / head_dim;
        int pos = t % prefill_len;
        int t2 = t / prefill_len;
        int head = t2 % heads;
        int b = t2 / heads;
        int src = ((b * heads + head) * max_seq + pos) * head_dim + d;
        int dst = ((b * heads + head) * prefill_len + pos) * head_dim + d;
        k[dst] = kv_h2f(k_cache[src]);
        v[dst] = kv_h2f(v_cache[src]);
    }
}

extern "C" __global__ void kv_cache_to_f32_bf16(
    const unsigned short* k_cache, const unsigned short* v_cache, float* k, float* v,
    int batch, int heads, int prefill_len, int max_seq, int head_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * prefill_len * head_dim;
    if (i < total) {
        int d = i % head_dim;
        int t = i / head_dim;
        int pos = t % prefill_len;
        int t2 = t / prefill_len;
        int head = t2 % heads;
        int b = t2 / heads;
        int src = ((b * heads + head) * max_seq + pos) * head_dim + d;
        int dst = ((b * heads + head) * prefill_len + pos) * head_dim + d;
        k[dst] = kv_b2f(k_cache[src]);
        v[dst] = kv_b2f(v_cache[src]);
    }
}

extern "C" __global__ void f32_to_f16_f32(const float* a, unsigned short* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned short h;
        asm("{ cvt.rn.f16.f32 %0, %1; }" : "=h"(h) : "f"(a[i]));
        out[i] = h;
    }
}

extern "C" __global__ void f16_to_f32_f32(const unsigned short* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float f;
        asm("{ cvt.f32.f16 %0, %1; }" : "=f"(f) : "h"(a[i]));
        out[i] = f;
    }
}

extern "C" __global__ void f32_to_bf16_f32(const float* a, unsigned short* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned int bits = __float_as_uint(a[i]);
        out[i] = (unsigned short)(bits >> 16);
    }
}

extern "C" __global__ void bf16_to_f32_f32(const unsigned short* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __uint_as_float(((unsigned int)a[i]) << 16);
    }
}

extern "C" __global__ void mse_forward_f32(
    const float* y, const float* t, float* diff, float* loss, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float d = y[i] - t[i];
        diff[i] = d;
        atomicAdd(loss, d * d / n);
    }
}

extern "C" __global__ void grad_y_f32(const float* diff, float* grad_y, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad_y[i] = 2.0f * diff[i] * scale;
}

extern "C" __global__ void grad_b_f32(
    const float* grad_y, float* grad_b, int batch, int out
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < out) {
        float sum = 0.0f;
        for (int i = 0; i < batch; i++) sum += grad_y[i * out + j];
        grad_b[j] = sum;
    }
}

extern "C" __global__ void bump_step_f32(int* step) {
    if (blockIdx.x == 0 && threadIdx.x == 0) (*step) += 1;
}

extern "C" __global__ void adam_step_f32(
    float* p, const float* g, float* m, float* v,
    int n, float lr, float b1, float b2, float eps, const int* step
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int t = *step;
        float bc1 = 1.0f - powf(b1, (float)t);
        float bc2 = 1.0f - powf(b2, (float)t);
        float gi = g[i];
        float nm = b1 * m[i] + (1.0f - b1) * gi;
        float nv = b2 * v[i] + (1.0f - b2) * gi * gi;
        float mhat = nm / bc1;
        float vhat = nv / bc2;
        m[i] = nm;
        v[i] = nv;
        p[i] = p[i] - lr * mhat / (sqrtf(vhat) + eps);
    }
}

extern "C" __global__ void embedding_one_f32(
    const int* token, const float* weight, float* out, int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) out[i] = weight[(*token) * dim + i];
}

extern "C" __global__ void kv_append_one_f32(
    const float* k_new, const float* v_new, float* k_cache, float* v_cache,
    const int* position, int heads, int head_dim, int max_seq
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = heads * head_dim;
    int pos = *position;
    if (i < total && pos < max_seq) {
        int head = i / head_dim;
        int d = i % head_dim;
        int dst = (head * max_seq + pos) * head_dim + d;
        k_cache[dst] = k_new[i];
        v_cache[dst] = v_new[i];
    }
}

extern "C" __global__ void attention_single_f32(
    const float* q, const float* k_cache, const float* v_cache,
    const int* position, float* out,
    int heads, int max_seq, int head_dim, float scale
) {
    int head = blockIdx.x;
    int n = *position + 1;
    if (n > max_seq) n = max_seq;
    int q_base = head * head_dim;
    int cache_base = head * max_seq * head_dim;
    float maxv = -1e30f;
    for (int p = 0; p < n; p++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * k_cache[cache_base + p * head_dim + d];
        dot *= scale;
        if (dot > maxv) maxv = dot;
    }
    float sum = 0.0f;
    for (int p = 0; p < n; p++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * k_cache[cache_base + p * head_dim + d];
        sum += expf(dot * scale - maxv);
    }
    float inv = 1.0f / sum;
    for (int d = 0; d < head_dim; d++) out[q_base + d] = 0.0f;
    for (int p = 0; p < n; p++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * k_cache[cache_base + p * head_dim + d];
        float prob = expf(dot * scale - maxv) * inv;
        for (int d = 0; d < head_dim; d++) out[q_base + d] += prob * v_cache[cache_base + p * head_dim + d];
    }
}

extern "C" __global__ void residual_inplace_f32(float* x, const float* delta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += delta[i];
}

extern "C" __global__ void gelu_inplace_f32(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

extern "C" __global__ void embedding_batch_i32(
    const int* tokens, const float* weight, float* out, int rows, int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * dim;
    if (i < total) {
        int row = i / dim;
        int d = i % dim;
        out[i] = weight[tokens[row] * dim + d];
    }
}

extern "C" __global__ void kv_prefill_f32(
    const float* k, const float* v, float* k_cache, float* v_cache,
    int batch, int prefill_len, int heads, int head_dim, int max_seq
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * prefill_len * heads * head_dim;
    if (i < total) {
        int d = i % head_dim;
        int t = i / head_dim;
        int head = t % heads;
        int t2 = t / heads;
        int pos = t2 % prefill_len;
        int b = t2 / prefill_len;
        int src = ((b * prefill_len + pos) * heads + head) * head_dim + d;
        int dst = ((b * heads + head) * max_seq + pos) * head_dim + d;
        k_cache[dst] = k[src];
        v_cache[dst] = v[src];
    }
}

extern "C" __global__ void attention_prefill_f32(
    const float* q, const float* k_cache, const float* v_cache, float* out,
    int batch, int prefill_len, int heads, int max_seq, int head_dim, float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * prefill_len;
    if (idx >= total) return;
    int qpos = idx % prefill_len;
    int t = idx / prefill_len;
    int head = t % heads;
    int b = t / heads;
    int q_base = ((b * prefill_len + qpos) * heads + head) * head_dim;
    int cache_base = (b * heads + head) * max_seq * head_dim;

    float maxv = -1e30f;
    for (int p = 0; p <= qpos; p++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * k_cache[cache_base + p * head_dim + d];
        dot *= scale;
        if (dot > maxv) maxv = dot;
    }
    float sum = 0.0f;
    for (int p = 0; p <= qpos; p++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * k_cache[cache_base + p * head_dim + d];
        sum += expf(dot * scale - maxv);
    }
    float inv = 1.0f / sum;
    for (int d = 0; d < head_dim; d++) out[q_base + d] = 0.0f;
    for (int p = 0; p <= qpos; p++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * k_cache[cache_base + p * head_dim + d];
        float prob = expf(dot * scale - maxv) * inv;
        for (int d = 0; d < head_dim; d++) out[q_base + d] += prob * v_cache[cache_base + p * head_dim + d];
    }
}

extern "C" __global__ void kv_append_batch_f32(
    const float* k, const float* v, float* k_cache, float* v_cache,
    const int* positions, int batch, int heads, int head_dim, int max_seq
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * head_dim;
    if (i < total) {
        int d = i % head_dim;
        int t = i / head_dim;
        int head = t % heads;
        int b = t / heads;
        int pos = positions[b];
        if (pos < max_seq) {
            int src = (b * heads + head) * head_dim + d;
            int dst = ((b * heads + head) * max_seq + pos) * head_dim + d;
            k_cache[dst] = k[src];
            v_cache[dst] = v[src];
        }
    }
}

extern "C" __global__ void attention_batch_decode_f32(
    const float* q, const float* k_cache, const float* v_cache,
    const int* positions, float* out,
    int batch, int heads, int max_seq, int head_dim, float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads;
    if (idx >= total) return;
    int head = idx % heads;
    int b = idx / heads;
    int n = positions[b] + 1;
    if (n > max_seq) n = max_seq;
    int q_base = (b * heads + head) * head_dim;
    int cache_base = (b * heads + head) * max_seq * head_dim;
    float maxv = -1e30f;
    for (int p = 0; p < n; p++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * k_cache[cache_base + p * head_dim + d];
        dot *= scale;
        if (dot > maxv) maxv = dot;
    }
    float sum = 0.0f;
    for (int p = 0; p < n; p++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * k_cache[cache_base + p * head_dim + d];
        sum += expf(dot * scale - maxv);
    }
    float inv = 1.0f / sum;
    for (int d = 0; d < head_dim; d++) out[q_base + d] = 0.0f;
    for (int p = 0; p < n; p++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[q_base + d] * k_cache[cache_base + p * head_dim + d];
        float prob = expf(dot * scale - maxv) * inv;
        for (int d = 0; d < head_dim; d++) out[q_base + d] += prob * v_cache[cache_base + p * head_dim + d];
    }
}

extern "C" __global__ void set_i32_value(int* dst, int value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *dst = value;
}

extern "C" __global__ void fill_i32_value(int* dst, int value, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = value;
}

extern "C" __global__ void bump_positions_i32(int* positions, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) positions[i] += 1;
}

"#;

pub(crate) struct Kernels {
    pub add: CudaFunction,
    pub sub: CudaFunction,
    pub mul: CudaFunction,
    pub div: CudaFunction,
    pub sqrt: CudaFunction,
    pub relu: CudaFunction,
    pub relu_backward: CudaFunction,
    pub silu: CudaFunction,
    pub gelu: CudaFunction,
    pub exp: CudaFunction,
    pub log: CudaFunction,
    pub sigmoid: CudaFunction,
    pub clamp: CudaFunction,
    pub abs: CudaFunction,
    pub f32_to_f16: CudaFunction,
    pub f16_to_f32: CudaFunction,
    pub f32_to_bf16: CudaFunction,
    pub bf16_to_f32: CudaFunction,
    pub bias_add: CudaFunction,
    pub mse_forward: CudaFunction,
    pub grad_y: CudaFunction,
    pub grad_b: CudaFunction,
    pub bump_step: CudaFunction,
    pub adam_step: CudaFunction,
    pub embedding_one: CudaFunction,
    pub kv_append_one: CudaFunction,
    pub attention_single: CudaFunction,
    pub residual_inplace: CudaFunction,
    pub gelu_inplace: CudaFunction,
    pub embedding_batch: CudaFunction,
    pub kv_prefill: CudaFunction,
    pub attention_prefill: CudaFunction,
    pub kv_append_batch: CudaFunction,
    pub attention_batch_decode: CudaFunction,
    pub set_i32_value: CudaFunction,
    pub kv_prefill_f16: CudaFunction,
    pub kv_prefill_bf16: CudaFunction,
    pub kv_append_batch_f16: CudaFunction,
    pub kv_append_batch_bf16: CudaFunction,
    pub attention_batch_decode_f16: CudaFunction,
    pub attention_batch_decode_bf16: CudaFunction,
    pub kv_cache_to_f32_f16: CudaFunction,
    pub kv_cache_to_f32_bf16: CudaFunction,
    pub split_heads_prefill: CudaFunction,
    pub merge_heads_prefill: CudaFunction,
    pub causal_softmax_warp: CudaFunction,
    pub gather_last_rows: CudaFunction,
    pub copy_i32: CudaFunction,
    pub gather_last_rows_var: CudaFunction,
    pub flash_attention_prefill: CudaFunction,
    pub flash_attention_decode_lowp: CudaFunction,
    pub attention_batch_decode_wide: CudaFunction,
    pub attention_batch_decode_wide_f16: CudaFunction,
    pub attention_batch_decode_wide_bf16: CudaFunction,
    pub split_heads_decode: CudaFunction,
    pub merge_heads_decode: CudaFunction,
    pub decode_softmax: CudaFunction,
    pub fill_i32_value: CudaFunction,
    pub bump_positions: CudaFunction,
    pub scale: CudaFunction,
    pub fill: CudaFunction,
    pub transpose2d: CudaFunction,
    pub adam_update: CudaFunction,
    pub sum_all: CudaFunction,
    pub sum_to_bias: CudaFunction,
    pub embedding_backward: CudaFunction,
    pub layer_norm_backward_dx: CudaFunction,
    pub layer_norm_backward_params: CudaFunction,
    pub softmax_backward: CudaFunction,
    pub embedding: CudaFunction,
    pub layer_norm: CudaFunction,
    pub softmax_last_dim: CudaFunction,
    pub split_heads: CudaFunction,
    pub merge_heads: CudaFunction,
    pub concat_seq: CudaFunction,
    pub last_row: CudaFunction,
    pub attention: CudaFunction,
    pub paged_attention_wide: CudaFunction,
    pub paged_attention_batch_wide: CudaFunction,
    pub paged_append: CudaFunction,
    pub softmax_ce: CudaFunction,
    pub softmax_ce_backward: CudaFunction,
}

impl Kernels {
    pub fn compile(ctx: &Arc<CudaContext>, arch: &'static str) -> Result<Self> {
        let options = CompileOptions {
            arch: Some(arch),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(SOURCE, options)
            .map_err(|error| Error::Backend(format!("nvrtc: {error}")))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|error| Error::Backend(format!("load module: {error}")))?;
        let load = |name: &str| {
            module
                .load_function(name)
                .map_err(|error| Error::Backend(format!("load {name}: {error}")))
        };
        Ok(Self {
            add: load("add_f32")?,
            sub: load("sub_f32")?,
            mul: load("mul_f32")?,
            div: load("div_f32")?,
            sqrt: load("sqrt_f32")?,
            relu: load("relu_f32")?,
            relu_backward: load("relu_backward_f32")?,
            silu: load("silu_f32")?,
            gelu: load("gelu_f32")?,
            exp: load("exp_f32")?,
            log: load("log_f32")?,
            sigmoid: load("sigmoid_f32")?,
            clamp: load("clamp_f32")?,
            abs: load("abs_f32")?,
            f32_to_f16: load("f32_to_f16_f32")?,
            f16_to_f32: load("f16_to_f32_f32")?,
            f32_to_bf16: load("f32_to_bf16_f32")?,
            bf16_to_f32: load("bf16_to_f32_f32")?,
            bias_add: load("bias_add_f32")?,
            mse_forward: load("mse_forward_f32")?,
            grad_y: load("grad_y_f32")?,
            grad_b: load("grad_b_f32")?,
            bump_step: load("bump_step_f32")?,
            adam_step: load("adam_step_f32")?,
            embedding_one: load("embedding_one_f32")?,
            kv_append_one: load("kv_append_one_f32")?,
            attention_single: load("attention_single_f32")?,
            residual_inplace: load("residual_inplace_f32")?,
            gelu_inplace: load("gelu_inplace_f32")?,
            embedding_batch: load("embedding_batch_i32")?,
            kv_prefill: load("kv_prefill_f32")?,
            attention_prefill: load("attention_prefill_f32")?,
            kv_append_batch: load("kv_append_batch_f32")?,
            attention_batch_decode: load("attention_batch_decode_f32")?,
            set_i32_value: load("set_i32_value")?,
            kv_prefill_f16: load("kv_prefill_f16_f32")?,
            kv_prefill_bf16: load("kv_prefill_bf16_f32")?,
            kv_append_batch_f16: load("kv_append_batch_f16_f32")?,
            kv_append_batch_bf16: load("kv_append_batch_bf16_f32")?,
            attention_batch_decode_f16: load("attention_batch_decode_f16_f32")?,
            attention_batch_decode_bf16: load("attention_batch_decode_bf16_f32")?,
            kv_cache_to_f32_f16: load("kv_cache_to_f32_f16")?,
            kv_cache_to_f32_bf16: load("kv_cache_to_f32_bf16")?,
            split_heads_prefill: load("split_heads_prefill_f32")?,
            merge_heads_prefill: load("merge_heads_prefill_f32")?,
            causal_softmax_warp: load("causal_softmax_warp_f32")?,
            gather_last_rows: load("gather_last_rows_f32")?,
            copy_i32: load("copy_i32")?,
            gather_last_rows_var: load("gather_last_rows_var_f32")?,
            flash_attention_prefill: load("flash_attention_prefill_f32")?,
            flash_attention_decode_lowp: load("flash_attention_decode_lowp_f32")?,
            attention_batch_decode_wide: load("attention_batch_decode_wide_f32")?,
            attention_batch_decode_wide_f16: load("attention_batch_decode_wide_f16_f32")?,
            attention_batch_decode_wide_bf16: load("attention_batch_decode_wide_bf16_f32")?,
            split_heads_decode: load("split_heads_decode_f32")?,
            merge_heads_decode: load("merge_heads_decode_f32")?,
            decode_softmax: load("decode_softmax_f32")?,
            fill_i32_value: load("fill_i32_value")?,
            bump_positions: load("bump_positions_i32")?,
            scale: load("scale_f32")?,
            fill: load("fill_f32")?,
            transpose2d: load("transpose2d_f32")?,
            adam_update: load("adam_update_f32")?,
            sum_all: load("sum_all_f32")?,
            sum_to_bias: load("sum_to_bias_f32")?,
            embedding_backward: load("embedding_backward_f32")?,
            layer_norm_backward_dx: load("layer_norm_backward_dx_f32")?,
            layer_norm_backward_params: load("layer_norm_backward_params_f32")?,
            softmax_backward: load("softmax_backward_f32")?,
            embedding: load("embedding_f32")?,
            layer_norm: load("layer_norm_f32")?,
            softmax_last_dim: load("softmax_last_dim_f32")?,
            split_heads: load("split_heads_f32")?,
            merge_heads: load("merge_heads_f32")?,
            concat_seq: load("concat_seq_f32")?,
            last_row: load("last_row_f32")?,
            attention: load("attention_f32")?,
            paged_attention_wide: load("paged_attention_wide_f32")?,
            paged_attention_batch_wide: load("paged_attention_batch_wide_f32")?,
            paged_append: load("paged_append_f32")?,
            softmax_ce: load("softmax_ce_f32")?,
            softmax_ce_backward: load("softmax_ce_backward_f32")?,
        })
    }
}
