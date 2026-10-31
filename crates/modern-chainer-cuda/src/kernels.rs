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
    pub bias_add: CudaFunction,
    pub transpose2d: CudaFunction,
    pub adam_update: CudaFunction,
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
            bias_add: load("bias_add_f32")?,
            transpose2d: load("transpose2d_f32")?,
            adam_update: load("adam_update_f32")?,
        })
    }
}
