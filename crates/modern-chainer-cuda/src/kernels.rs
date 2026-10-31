use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use modern_chainer_core::{Error, Result};

const SOURCE: &str = r#"
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
    pub scale: CudaFunction,
    pub fill: CudaFunction,
    pub transpose2d: CudaFunction,
    pub adam_update: CudaFunction,
    pub sum_all: CudaFunction,
    pub sum_to_bias: CudaFunction,
    pub embedding: CudaFunction,
    pub layer_norm: CudaFunction,
    pub softmax_last_dim: CudaFunction,
    pub split_heads: CudaFunction,
    pub merge_heads: CudaFunction,
    pub concat_seq: CudaFunction,
    pub last_row: CudaFunction,
    pub attention: CudaFunction,
    pub paged_attention: CudaFunction,
    pub paged_attention_batch: CudaFunction,
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
            scale: load("scale_f32")?,
            fill: load("fill_f32")?,
            transpose2d: load("transpose2d_f32")?,
            adam_update: load("adam_update_f32")?,
            sum_all: load("sum_all_f32")?,
            sum_to_bias: load("sum_to_bias_f32")?,
            embedding: load("embedding_f32")?,
            layer_norm: load("layer_norm_f32")?,
            softmax_last_dim: load("softmax_last_dim_f32")?,
            split_heads: load("split_heads_f32")?,
            merge_heads: load("merge_heads_f32")?,
            concat_seq: load("concat_seq_f32")?,
            last_row: load("last_row_f32")?,
            attention: load("attention_f32")?,
            paged_attention: load("paged_attention_f32")?,
            paged_attention_batch: load("paged_attention_batch_f32")?,
            paged_append: load("paged_append_f32")?,
            softmax_ce: load("softmax_ce_f32")?,
            softmax_ce_backward: load("softmax_ce_backward_f32")?,
        })
    }
}
