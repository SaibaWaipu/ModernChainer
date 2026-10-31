use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use half::f16;

use std::collections::HashMap;

use modern_chainer_core::{AdamConfig, Backend, DType, Error, Result, Shape, Tensor};
use modern_chainer_llm::{Decoder, DecoderConfig};

use crate::backend::{CudaBackend, CudaStorage};
use crate::graph::CudaGraphRunner;

fn storage_mut(tensor: &mut Tensor<CudaBackend>) -> Result<&mut CudaStorage> {
    tensor
        .storage_mut()
        .ok_or_else(|| Error::invalid("static tensor storage is shared"))
}

pub struct StaticBuilder {
    backend: CudaBackend,
}

impl StaticBuilder {
    fn new(backend: CudaBackend) -> Self {
        Self { backend }
    }

    pub fn backend(&self) -> &CudaBackend {
        &self.backend
    }

    pub fn zeros(&self, shape: impl Into<Shape>, dtype: DType) -> Result<Tensor<CudaBackend>> {
        Tensor::zeros(self.backend.clone(), shape, dtype)
    }

    pub fn from_f32(&self, shape: impl Into<Shape>, values: &[f32]) -> Result<Tensor<CudaBackend>> {
        Tensor::from_f32(self.backend.clone(), values, shape)
    }

    pub fn from_tensor(&self, tensor: &Tensor<CudaBackend>) -> Result<Tensor<CudaBackend>> {
        let bytes = tensor.to_bytes()?;
        let mut out = self.zeros(tensor.shape().clone(), tensor.dtype())?;
        self.backend
            .copy_from_host(storage_mut(&mut out)?, &bytes)?;
        Ok(out)
    }

    pub fn i32_scalar(&self) -> Result<CudaSlice<i32>> {
        self.i32_zeros(1)
    }

    pub fn i32_zeros(&self, len: usize) -> Result<CudaSlice<i32>> {
        self.backend
            .runtime
            .stream
            .alloc_zeros::<i32>(len)
            .map_err(|error| Error::Backend(format!("cuda alloc: {error}")))
    }
}

/// In-place primitives available to a captured static step.  Every method
/// enqueues work on the backend stream and never allocates.
pub struct CudaStaticContext {
    backend: CudaBackend,
}

impl CudaStaticContext {
    fn new(backend: CudaBackend) -> Self {
        Self { backend }
    }

    pub fn backend(&self) -> &CudaBackend {
        &self.backend
    }

    pub fn write_f32(&self, tensor: &mut Tensor<CudaBackend>, values: &[f32]) -> Result<()> {
        if values.len() != tensor.numel() {
            return Err(Error::invalid("static input size mismatch"));
        }
        let bytes = bytemuck::cast_slice::<f32, u8>(values);
        self.backend.copy_from_host(storage_mut(tensor)?, bytes)
    }

    pub fn write_i32(&self, slice: &mut CudaSlice<i32>, value: i32) -> Result<()> {
        self.backend
            .runtime
            .stream
            .memcpy_htod(&[value], slice)
            .map_err(|error| Error::Backend(format!("htod: {error}")))
    }

    pub fn read_f32(&self, tensor: &Tensor<CudaBackend>) -> Result<f32> {
        let bytes = tensor.to_bytes()?;
        let (chunks, rest) = bytes.as_chunks::<4>();
        if !rest.is_empty() {
            return Err(Error::invalid("f32 tensor size is not a multiple of 4"));
        }
        chunks
            .first()
            .map(|chunk| f32::from_ne_bytes(*chunk))
            .ok_or_else(|| Error::invalid("empty f32 tensor"))
    }

    pub fn write_i32_slice(&self, dst: &mut CudaSlice<i32>, values: &[i32]) -> Result<()> {
        if values.len() != dst.len() {
            return Err(Error::invalid("static i32 slice size mismatch"));
        }
        self.backend
            .runtime
            .stream
            .memcpy_htod(values, dst)
            .map_err(|error| Error::Backend(format!("htod: {error}")))
    }

    pub fn read_i32_vec(&self, slice: &CudaSlice<i32>) -> Result<Vec<i32>> {
        self.backend
            .runtime
            .stream
            .clone_dtoh(slice)
            .map_err(|error| Error::Backend(format!("dtoh: {error}")))
    }

    pub fn read_i32(&self, slice: &CudaSlice<i32>) -> Result<i32> {
        let host = self
            .backend
            .runtime
            .stream
            .clone_dtoh(slice)
            .map_err(|error| Error::Backend(format!("dtoh: {error}")))?;
        host.first()
            .copied()
            .ok_or_else(|| Error::invalid("empty i32 slice"))
    }

    pub fn read_f32_vec(&self, tensor: &Tensor<CudaBackend>) -> Result<Vec<f32>> {
        let bytes = tensor.to_bytes()?;
        let (chunks, rest) = bytes.as_chunks::<4>();
        if !rest.is_empty() {
            return Err(Error::invalid("f32 tensor size is not a multiple of 4"));
        }
        Ok(chunks
            .iter()
            .map(|chunk| f32::from_ne_bytes(*chunk))
            .collect())
    }

    pub fn fill_zero(&self, out: &mut Tensor<CudaBackend>) -> Result<()> {
        let n = out.numel();
        let n_i = i32::try_from(n).map_err(|_| Error::invalid("tensor too large"))?;
        let zero = 0.0f32;
        let runtime = self.backend.runtime.clone();
        let out = storage_mut(out)?.f32_mut()?;
        let mut builder = runtime.stream.launch_builder(&runtime.kernels.fill);
        builder.arg(out).arg(&zero).arg(&n_i);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(n as u32))
                .map_err(|error| Error::Backend(format!("static fill: {error}")))?;
        }
        Ok(())
    }

    pub fn linear_into(
        &self,
        x: &Tensor<CudaBackend>,
        weight: &Tensor<CudaBackend>,
        bias: Option<&Tensor<CudaBackend>>,
        out: &mut Tensor<CudaBackend>,
    ) -> Result<()> {
        if x.rank() != 2 || weight.rank() != 2 {
            return Err(Error::invalid("static linear expects 2-D tensors"));
        }
        let (batch, in_features) = (x.shape().dims()[0], x.shape().dims()[1]);
        let (out_features, k2) = (weight.shape().dims()[0], weight.shape().dims()[1]);
        if in_features != k2 || out.shape().dims() != [batch, out_features] {
            return Err(Error::invalid("static linear shape mismatch"));
        }
        let x_slice = x.storage().f32()?;
        let weight_slice = weight.storage().f32()?;
        let out_slice = storage_mut(out)?.f32_mut()?;
        unsafe {
            self.backend.runtime.blas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_T,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: out_features as i32,
                    n: batch as i32,
                    k: in_features as i32,
                    alpha: 1.0,
                    lda: in_features as i32,
                    ldb: in_features as i32,
                    beta: 0.0,
                    ldc: out_features as i32,
                },
                weight_slice,
                x_slice,
                out_slice,
            )
        }
        .map_err(|error| Error::Backend(format!("static cublas linear: {error}")))?;
        if let Some(bias) = bias {
            let bias_slice = bias.storage().f32()?;
            let rows = i32::try_from(batch).map_err(|_| Error::invalid("tensor too large"))?;
            let cols =
                i32::try_from(out_features).map_err(|_| Error::invalid("tensor too large"))?;
            let total = u32::try_from(batch * out_features)
                .map_err(|_| Error::invalid("tensor too large"))?;
            let runtime = self.backend.runtime.clone();
            let out_slice = storage_mut(out)?.f32_mut()?;
            let mut builder = runtime.stream.launch_builder(&runtime.kernels.bias_add);
            builder.arg(out_slice).arg(bias_slice).arg(&rows).arg(&cols);
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(total))
                    .map_err(|error| Error::Backend(format!("static bias: {error}")))?;
            }
        }
        Ok(())
    }

    pub fn matmul_grad_w_into(
        &self,
        x: &Tensor<CudaBackend>,
        grad_y: &Tensor<CudaBackend>,
        grad_w: &mut Tensor<CudaBackend>,
    ) -> Result<()> {
        let (batch, in_features) = (x.shape().dims()[0], x.shape().dims()[1]);
        let out_features = grad_y.shape().dims()[1];
        if grad_w.shape().dims() != [out_features, in_features] {
            return Err(Error::invalid("static grad_w shape mismatch"));
        }
        let x_slice = x.storage().f32()?;
        let grad_y_slice = grad_y.storage().f32()?;
        let grad_w_slice = storage_mut(grad_w)?.f32_mut()?;
        unsafe {
            self.backend.runtime.blas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_T,
                    m: in_features as i32,
                    n: out_features as i32,
                    k: batch as i32,
                    alpha: 1.0,
                    lda: in_features as i32,
                    ldb: out_features as i32,
                    beta: 0.0,
                    ldc: in_features as i32,
                },
                x_slice,
                grad_y_slice,
                grad_w_slice,
            )
        }
        .map_err(|error| Error::Backend(format!("static cublas grad_w: {error}")))
    }

    pub fn mse_into(
        &self,
        y: &Tensor<CudaBackend>,
        target: &Tensor<CudaBackend>,
        diff: &mut Tensor<CudaBackend>,
        loss: &mut Tensor<CudaBackend>,
    ) -> Result<()> {
        let n = y.numel();
        if target.numel() != n || diff.numel() != n {
            return Err(Error::invalid("static mse shape mismatch"));
        }
        self.fill_zero(loss)?;
        let n_i = i32::try_from(n).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let y_slice = y.storage().f32()?;
        let target_slice = target.storage().f32()?;
        let diff_slice = storage_mut(diff)?.f32_mut()?;
        let loss_slice = storage_mut(loss)?.f32_mut()?;
        let mut builder = runtime.stream.launch_builder(&runtime.kernels.mse_forward);
        builder
            .arg(y_slice)
            .arg(target_slice)
            .arg(diff_slice)
            .arg(loss_slice)
            .arg(&n_i);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(n as u32))
                .map_err(|error| Error::Backend(format!("static mse: {error}")))?;
        }
        Ok(())
    }

    pub fn grad_y_into(
        &self,
        diff: &Tensor<CudaBackend>,
        grad_y: &mut Tensor<CudaBackend>,
        scale: f32,
    ) -> Result<()> {
        let n = diff.numel();
        let n_i = i32::try_from(n).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let diff_slice = diff.storage().f32()?;
        let grad_slice = storage_mut(grad_y)?.f32_mut()?;
        let mut builder = runtime.stream.launch_builder(&runtime.kernels.grad_y);
        builder
            .arg(diff_slice)
            .arg(grad_slice)
            .arg(&n_i)
            .arg(&scale);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(n as u32))
                .map_err(|error| Error::Backend(format!("static grad_y: {error}")))?;
        }
        Ok(())
    }

    pub fn grad_b_into(
        &self,
        grad_y: &Tensor<CudaBackend>,
        grad_b: &mut Tensor<CudaBackend>,
        batch: usize,
        out_features: usize,
    ) -> Result<()> {
        let rows = i32::try_from(batch).map_err(|_| Error::invalid("tensor too large"))?;
        let cols = i32::try_from(out_features).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let grad_y_slice = grad_y.storage().f32()?;
        let grad_b_slice = storage_mut(grad_b)?.f32_mut()?;
        let mut builder = runtime.stream.launch_builder(&runtime.kernels.grad_b);
        builder
            .arg(grad_y_slice)
            .arg(grad_b_slice)
            .arg(&rows)
            .arg(&cols);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(cols as u32))
                .map_err(|error| Error::Backend(format!("static grad_b: {error}")))?;
        }
        Ok(())
    }

    pub fn bump_step(&self, step: &mut CudaSlice<i32>) -> Result<()> {
        let runtime = self.backend.runtime.clone();
        let mut builder = runtime.stream.launch_builder(&runtime.kernels.bump_step);
        builder.arg(step);
        unsafe {
            builder
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|error| Error::Backend(format!("static step: {error}")))?;
        }
        Ok(())
    }

    pub fn adam_into(
        &self,
        param: &mut Tensor<CudaBackend>,
        grad: &Tensor<CudaBackend>,
        m: &mut Tensor<CudaBackend>,
        v: &mut Tensor<CudaBackend>,
        step: &CudaSlice<i32>,
        cfg: &AdamConfig,
    ) -> Result<()> {
        let n = param.numel();
        if grad.numel() != n || m.numel() != n || v.numel() != n {
            return Err(Error::invalid("static adam shape mismatch"));
        }
        let n_i = i32::try_from(n).map_err(|_| Error::invalid("tensor too large"))?;
        let lr = cfg.lr as f32;
        let b1 = cfg.beta1 as f32;
        let b2 = cfg.beta2 as f32;
        let eps = cfg.eps as f32;
        let runtime = self.backend.runtime.clone();
        let grad_slice = grad.storage().f32()?;
        let p_slice = storage_mut(param)?.f32_mut()?;
        let m_slice = storage_mut(m)?.f32_mut()?;
        let v_slice = storage_mut(v)?.f32_mut()?;
        let mut builder = runtime.stream.launch_builder(&runtime.kernels.adam_step);
        builder
            .arg(p_slice)
            .arg(grad_slice)
            .arg(m_slice)
            .arg(v_slice)
            .arg(&n_i)
            .arg(&lr)
            .arg(&b1)
            .arg(&b2)
            .arg(&eps)
            .arg(step);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(n as u32))
                .map_err(|error| Error::Backend(format!("static adam: {error}")))?;
        }
        Ok(())
    }

    pub fn embedding_one(
        &self,
        token: &CudaSlice<i32>,
        weight: &Tensor<CudaBackend>,
        out: &mut Tensor<CudaBackend>,
        dim: usize,
    ) -> Result<()> {
        let dim_i = i32::try_from(dim).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let weight_slice = weight.storage().f32()?;
        let out_slice = storage_mut(out)?.f32_mut()?;
        let mut builder = runtime
            .stream
            .launch_builder(&runtime.kernels.embedding_one);
        builder
            .arg(token)
            .arg(weight_slice)
            .arg(out_slice)
            .arg(&dim_i);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(dim as u32))
                .map_err(|error| Error::Backend(format!("static embedding: {error}")))?;
        }
        Ok(())
    }

    pub fn layer_norm_into(
        &self,
        x: &Tensor<CudaBackend>,
        gamma: &Tensor<CudaBackend>,
        beta: &Tensor<CudaBackend>,
        out: &mut Tensor<CudaBackend>,
        dim: usize,
        eps: f32,
    ) -> Result<()> {
        let rows = 1i32;
        let dim_i = i32::try_from(dim).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let x_slice = x.storage().f32()?;
        let gamma_slice = gamma.storage().f32()?;
        let beta_slice = beta.storage().f32()?;
        let out_slice = storage_mut(out)?.f32_mut()?;
        let mut builder = runtime.stream.launch_builder(&runtime.kernels.layer_norm);
        builder
            .arg(x_slice)
            .arg(gamma_slice)
            .arg(beta_slice)
            .arg(out_slice)
            .arg(&rows)
            .arg(&dim_i)
            .arg(&eps);
        unsafe {
            builder
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|error| Error::Backend(format!("static layer_norm: {error}")))?;
        }
        Ok(())
    }

    pub fn gelu_inplace(&self, x: &mut Tensor<CudaBackend>, n: usize) -> Result<()> {
        let n_i = i32::try_from(n).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let x_slice = storage_mut(x)?.f32_mut()?;
        let mut builder = runtime.stream.launch_builder(&runtime.kernels.gelu_inplace);
        builder.arg(x_slice).arg(&n_i);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(n as u32))
                .map_err(|error| Error::Backend(format!("static gelu: {error}")))?;
        }
        Ok(())
    }

    pub fn residual_inplace(
        &self,
        x: &mut Tensor<CudaBackend>,
        delta: &Tensor<CudaBackend>,
        n: usize,
    ) -> Result<()> {
        let n_i = i32::try_from(n).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let delta_slice = delta.storage().f32()?;
        let x_slice = storage_mut(x)?.f32_mut()?;
        let mut builder = runtime
            .stream
            .launch_builder(&runtime.kernels.residual_inplace);
        builder.arg(x_slice).arg(delta_slice).arg(&n_i);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(n as u32))
                .map_err(|error| Error::Backend(format!("static residual: {error}")))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn kv_append_one(
        &self,
        k_new: &Tensor<CudaBackend>,
        v_new: &Tensor<CudaBackend>,
        k_cache: &mut Tensor<CudaBackend>,
        v_cache: &mut Tensor<CudaBackend>,
        position: &CudaSlice<i32>,
        heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> Result<()> {
        let total = heads * head_dim;
        let heads_i = i32::try_from(heads).map_err(|_| Error::invalid("tensor too large"))?;
        let dim_i = i32::try_from(head_dim).map_err(|_| Error::invalid("tensor too large"))?;
        let seq_i = i32::try_from(max_seq).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let k_slice = k_new.storage().f32()?;
        let v_slice = v_new.storage().f32()?;
        let k_cache_slice = storage_mut(k_cache)?.f32_mut()?;
        let v_cache_slice = storage_mut(v_cache)?.f32_mut()?;
        let mut builder = runtime
            .stream
            .launch_builder(&runtime.kernels.kv_append_one);
        builder
            .arg(k_slice)
            .arg(v_slice)
            .arg(k_cache_slice)
            .arg(v_cache_slice)
            .arg(position)
            .arg(&heads_i)
            .arg(&dim_i)
            .arg(&seq_i);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(total as u32))
                .map_err(|error| Error::Backend(format!("static kv append: {error}")))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_single(
        &self,
        q: &Tensor<CudaBackend>,
        k_cache: &Tensor<CudaBackend>,
        v_cache: &Tensor<CudaBackend>,
        position: &CudaSlice<i32>,
        out: &mut Tensor<CudaBackend>,
        heads: usize,
        max_seq: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<()> {
        let heads_i = i32::try_from(heads).map_err(|_| Error::invalid("tensor too large"))?;
        let seq_i = i32::try_from(max_seq).map_err(|_| Error::invalid("tensor too large"))?;
        let dim_i = i32::try_from(head_dim).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let q_slice = q.storage().f32()?;
        let k_slice = k_cache.storage().f32()?;
        let v_slice = v_cache.storage().f32()?;
        let out_slice = storage_mut(out)?.f32_mut()?;
        let mut builder = runtime
            .stream
            .launch_builder(&runtime.kernels.attention_single);
        builder
            .arg(q_slice)
            .arg(k_slice)
            .arg(v_slice)
            .arg(position)
            .arg(out_slice)
            .arg(&heads_i)
            .arg(&seq_i)
            .arg(&dim_i)
            .arg(&scale);
        unsafe {
            builder
                .launch(LaunchConfig {
                    grid_dim: (heads as u32, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|error| Error::Backend(format!("static attention: {error}")))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn embedding_batch(
        &self,
        tokens: &CudaSlice<i32>,
        weight: &Tensor<CudaBackend>,
        out: &mut Tensor<CudaBackend>,
        rows: usize,
        dim: usize,
    ) -> Result<()> {
        let total = rows * dim;
        let rows_i = i32::try_from(rows).map_err(|_| Error::invalid("tensor too large"))?;
        let dim_i = i32::try_from(dim).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let weight_slice = weight.storage().f32()?;
        let out_slice = storage_mut(out)?.f32_mut()?;
        let mut builder = runtime
            .stream
            .launch_builder(&runtime.kernels.embedding_batch);
        builder
            .arg(tokens)
            .arg(weight_slice)
            .arg(out_slice)
            .arg(&rows_i)
            .arg(&dim_i);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(total as u32))
                .map_err(|error| Error::Backend(format!("static embedding batch: {error}")))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn layer_norm_batch(
        &self,
        x: &Tensor<CudaBackend>,
        gamma: &Tensor<CudaBackend>,
        beta: &Tensor<CudaBackend>,
        out: &mut Tensor<CudaBackend>,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<()> {
        let rows_i = i32::try_from(rows).map_err(|_| Error::invalid("tensor too large"))?;
        let dim_i = i32::try_from(dim).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let x_slice = x.storage().f32()?;
        let gamma_slice = gamma.storage().f32()?;
        let beta_slice = beta.storage().f32()?;
        let out_slice = storage_mut(out)?.f32_mut()?;
        let mut builder = runtime.stream.launch_builder(&runtime.kernels.layer_norm);
        builder
            .arg(x_slice)
            .arg(gamma_slice)
            .arg(beta_slice)
            .arg(out_slice)
            .arg(&rows_i)
            .arg(&dim_i)
            .arg(&eps);
        unsafe {
            builder
                .launch(LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|error| Error::Backend(format!("static layer_norm batch: {error}")))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn kv_prefill(
        &self,
        k: &Tensor<CudaBackend>,
        v: &Tensor<CudaBackend>,
        k_cache: &mut Tensor<CudaBackend>,
        v_cache: &mut Tensor<CudaBackend>,
        batch: usize,
        prefill_len: usize,
        heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> Result<()> {
        let total = batch * prefill_len * heads * head_dim;
        let batch_i = i32::try_from(batch).map_err(|_| Error::invalid("tensor too large"))?;
        let len_i = i32::try_from(prefill_len).map_err(|_| Error::invalid("tensor too large"))?;
        let heads_i = i32::try_from(heads).map_err(|_| Error::invalid("tensor too large"))?;
        let dim_i = i32::try_from(head_dim).map_err(|_| Error::invalid("tensor too large"))?;
        let seq_i = i32::try_from(max_seq).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let k_slice = k.storage().f32()?;
        let v_slice = v.storage().f32()?;
        let k_cache_slice = storage_mut(k_cache)?.f32_mut()?;
        let v_cache_slice = storage_mut(v_cache)?.f32_mut()?;
        let mut builder = runtime.stream.launch_builder(&runtime.kernels.kv_prefill);
        builder
            .arg(k_slice)
            .arg(v_slice)
            .arg(k_cache_slice)
            .arg(v_cache_slice)
            .arg(&batch_i)
            .arg(&len_i)
            .arg(&heads_i)
            .arg(&dim_i)
            .arg(&seq_i);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(total as u32))
                .map_err(|error| Error::Backend(format!("static kv prefill: {error}")))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn kv_prefill_typed(
        &self,
        k: &Tensor<CudaBackend>,
        v: &Tensor<CudaBackend>,
        k_cache: &mut Tensor<CudaBackend>,
        v_cache: &mut Tensor<CudaBackend>,
        batch: usize,
        prefill_len: usize,
        heads: usize,
        head_dim: usize,
        max_seq: usize,
        dtype: DType,
    ) -> Result<()> {
        if dtype == DType::F32 {
            return self.kv_prefill(
                k,
                v,
                k_cache,
                v_cache,
                batch,
                prefill_len,
                heads,
                head_dim,
                max_seq,
            );
        }
        let total = batch * prefill_len * heads * head_dim;
        let batch_i = i32::try_from(batch).map_err(|_| Error::invalid("tensor too large"))?;
        let len_i = i32::try_from(prefill_len).map_err(|_| Error::invalid("tensor too large"))?;
        let heads_i = i32::try_from(heads).map_err(|_| Error::invalid("tensor too large"))?;
        let dim_i = i32::try_from(head_dim).map_err(|_| Error::invalid("tensor too large"))?;
        let seq_i = i32::try_from(max_seq).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let k_slice = k.storage().f32()?;
        let v_slice = v.storage().f32()?;
        let launch = |k_cache: &CudaSlice<f16>,
                      v_cache: &CudaSlice<f16>,
                      kernel: &CudaFunction,
                      name: &str|
         -> Result<()> {
            let mut builder = runtime.stream.launch_builder(kernel);
            builder
                .arg(k_slice)
                .arg(v_slice)
                .arg(k_cache)
                .arg(v_cache)
                .arg(&batch_i)
                .arg(&len_i)
                .arg(&heads_i)
                .arg(&dim_i)
                .arg(&seq_i);
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(total as u32))
                    .map_err(|error| Error::Backend(format!("static {name}: {error}")))?;
            }
            Ok(())
        };
        match dtype {
            DType::F16 => {
                let k_cache = storage_mut(k_cache)?.f16()?;
                let v_cache = storage_mut(v_cache)?.f16()?;
                launch(
                    k_cache,
                    v_cache,
                    &runtime.kernels.kv_prefill_f16,
                    "kv prefill f16",
                )
            }
            DType::BF16 => {
                let k_cache = storage_mut(k_cache)?.bf16()?;
                let v_cache = storage_mut(v_cache)?.bf16()?;
                let mut builder = runtime
                    .stream
                    .launch_builder(&runtime.kernels.kv_prefill_bf16);
                builder
                    .arg(k_slice)
                    .arg(v_slice)
                    .arg(k_cache)
                    .arg(v_cache)
                    .arg(&batch_i)
                    .arg(&len_i)
                    .arg(&heads_i)
                    .arg(&dim_i)
                    .arg(&seq_i);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(total as u32))
                        .map_err(|error| {
                            Error::Backend(format!("static kv prefill bf16: {error}"))
                        })?;
                }
                Ok(())
            }
            _ => Err(Error::invalid("unsupported KV dtype")),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn kv_append_batch_typed(
        &self,
        k: &Tensor<CudaBackend>,
        v: &Tensor<CudaBackend>,
        k_cache: &mut Tensor<CudaBackend>,
        v_cache: &mut Tensor<CudaBackend>,
        positions: &CudaSlice<i32>,
        batch: usize,
        heads: usize,
        head_dim: usize,
        max_seq: usize,
        dtype: DType,
    ) -> Result<()> {
        if dtype == DType::F32 {
            return self.kv_append_batch(
                k, v, k_cache, v_cache, positions, batch, heads, head_dim, max_seq,
            );
        }
        let total = batch * heads * head_dim;
        let batch_i = i32::try_from(batch).map_err(|_| Error::invalid("tensor too large"))?;
        let heads_i = i32::try_from(heads).map_err(|_| Error::invalid("tensor too large"))?;
        let dim_i = i32::try_from(head_dim).map_err(|_| Error::invalid("tensor too large"))?;
        let seq_i = i32::try_from(max_seq).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let k_slice = k.storage().f32()?;
        let v_slice = v.storage().f32()?;
        let kernel = match dtype {
            DType::F16 => &runtime.kernels.kv_append_batch_f16,
            DType::BF16 => &runtime.kernels.kv_append_batch_bf16,
            _ => return Err(Error::invalid("unsupported KV dtype")),
        };
        match dtype {
            DType::F16 => {
                let k_cache = storage_mut(k_cache)?.f16()?;
                let v_cache = storage_mut(v_cache)?.f16()?;
                let mut builder = runtime.stream.launch_builder(kernel);
                builder
                    .arg(k_slice)
                    .arg(v_slice)
                    .arg(k_cache)
                    .arg(v_cache)
                    .arg(positions)
                    .arg(&batch_i)
                    .arg(&heads_i)
                    .arg(&dim_i)
                    .arg(&seq_i);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(total as u32))
                        .map_err(|error| {
                            Error::Backend(format!("static kv append f16: {error}"))
                        })?;
                }
            }
            DType::BF16 => {
                let k_cache = storage_mut(k_cache)?.bf16()?;
                let v_cache = storage_mut(v_cache)?.bf16()?;
                let mut builder = runtime.stream.launch_builder(kernel);
                builder
                    .arg(k_slice)
                    .arg(v_slice)
                    .arg(k_cache)
                    .arg(v_cache)
                    .arg(positions)
                    .arg(&batch_i)
                    .arg(&heads_i)
                    .arg(&dim_i)
                    .arg(&seq_i);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(total as u32))
                        .map_err(|error| {
                            Error::Backend(format!("static kv append bf16: {error}"))
                        })?;
                }
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_batch_decode_typed(
        &self,
        q: &Tensor<CudaBackend>,
        k_cache: &Tensor<CudaBackend>,
        v_cache: &Tensor<CudaBackend>,
        positions: &CudaSlice<i32>,
        out: &mut Tensor<CudaBackend>,
        batch: usize,
        heads: usize,
        max_seq: usize,
        head_dim: usize,
        scale: f32,
        dtype: DType,
    ) -> Result<()> {
        if dtype == DType::F32 {
            return self.attention_batch_decode(
                q, k_cache, v_cache, positions, out, batch, heads, max_seq, head_dim, scale,
            );
        }
        let total = batch * heads;
        let batch_i = i32::try_from(batch).map_err(|_| Error::invalid("tensor too large"))?;
        let heads_i = i32::try_from(heads).map_err(|_| Error::invalid("tensor too large"))?;
        let seq_i = i32::try_from(max_seq).map_err(|_| Error::invalid("tensor too large"))?;
        let dim_i = i32::try_from(head_dim).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let q_slice = q.storage().f32()?;
        let out_slice = storage_mut(out)?.f32_mut()?;
        let wide = head_dim <= 256;
        let kernel = match (dtype, wide) {
            (DType::F16, true) => &runtime.kernels.attention_batch_decode_wide_f16,
            (DType::BF16, true) => &runtime.kernels.attention_batch_decode_wide_bf16,
            (DType::F16, false) => &runtime.kernels.attention_batch_decode_f16,
            (DType::BF16, false) => &runtime.kernels.attention_batch_decode_bf16,
            _ => return Err(Error::invalid("unsupported KV dtype")),
        };
        match dtype {
            DType::F16 => {
                let k_slice = k_cache.storage().f16()?;
                let v_slice = v_cache.storage().f16()?;
                let mut builder = runtime.stream.launch_builder(kernel);
                builder
                    .arg(q_slice)
                    .arg(k_slice)
                    .arg(v_slice)
                    .arg(positions)
                    .arg(out_slice)
                    .arg(&batch_i)
                    .arg(&heads_i)
                    .arg(&seq_i)
                    .arg(&dim_i)
                    .arg(&scale);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(total as u32))
                        .map_err(|error| {
                            Error::Backend(format!("static attention decode f16: {error}"))
                        })?;
                }
            }
            DType::BF16 => {
                let k_slice = k_cache.storage().bf16()?;
                let v_slice = v_cache.storage().bf16()?;
                let mut builder = runtime.stream.launch_builder(kernel);
                builder
                    .arg(q_slice)
                    .arg(k_slice)
                    .arg(v_slice)
                    .arg(positions)
                    .arg(out_slice)
                    .arg(&batch_i)
                    .arg(&heads_i)
                    .arg(&seq_i)
                    .arg(&dim_i)
                    .arg(&scale);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(total as u32))
                        .map_err(|error| {
                            Error::Backend(format!("static attention decode bf16: {error}"))
                        })?;
                }
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn kv_cache_to_f32_typed(
        &self,
        k_cache: &Tensor<CudaBackend>,
        v_cache: &Tensor<CudaBackend>,
        k: &mut Tensor<CudaBackend>,
        v: &mut Tensor<CudaBackend>,
        batch: usize,
        heads: usize,
        prefill_len: usize,
        max_seq: usize,
        head_dim: usize,
        dtype: DType,
    ) -> Result<()> {
        let total = batch * heads * prefill_len * head_dim;
        let batch_i = i32::try_from(batch).map_err(|_| Error::invalid("tensor too large"))?;
        let heads_i = i32::try_from(heads).map_err(|_| Error::invalid("tensor too large"))?;
        let len_i = i32::try_from(prefill_len).map_err(|_| Error::invalid("tensor too large"))?;
        let seq_i = i32::try_from(max_seq).map_err(|_| Error::invalid("tensor too large"))?;
        let dim_i = i32::try_from(head_dim).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let k_out = storage_mut(k)?.f32_mut()?;
        let v_out = storage_mut(v)?.f32_mut()?;
        let kernel = match dtype {
            DType::F16 => &runtime.kernels.kv_cache_to_f32_f16,
            DType::BF16 => &runtime.kernels.kv_cache_to_f32_bf16,
            _ => return Err(Error::invalid("KV cache is already f32")),
        };
        match dtype {
            DType::F16 => {
                let k_cache = k_cache.storage().f16()?;
                let v_cache = v_cache.storage().f16()?;
                let mut builder = runtime.stream.launch_builder(kernel);
                builder
                    .arg(k_cache)
                    .arg(v_cache)
                    .arg(k_out)
                    .arg(v_out)
                    .arg(&batch_i)
                    .arg(&heads_i)
                    .arg(&len_i)
                    .arg(&seq_i)
                    .arg(&dim_i);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(total as u32))
                        .map_err(|error| {
                            Error::Backend(format!("static kv upcast f16: {error}"))
                        })?;
                }
            }
            DType::BF16 => {
                let k_cache = k_cache.storage().bf16()?;
                let v_cache = v_cache.storage().bf16()?;
                let mut builder = runtime.stream.launch_builder(kernel);
                builder
                    .arg(k_cache)
                    .arg(v_cache)
                    .arg(k_out)
                    .arg(v_out)
                    .arg(&batch_i)
                    .arg(&heads_i)
                    .arg(&len_i)
                    .arg(&seq_i)
                    .arg(&dim_i);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(total as u32))
                        .map_err(|error| {
                            Error::Backend(format!("static kv upcast bf16: {error}"))
                        })?;
                }
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    pub fn split_heads_prefill(
        &self,
        q: &Tensor<CudaBackend>,
        qh: &mut Tensor<CudaBackend>,
        batch: usize,
        prefill_len: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        let total = batch * prefill_len * heads * head_dim;
        let batch_i = i32::try_from(batch).map_err(|_| Error::invalid("tensor too large"))?;
        let len_i = i32::try_from(prefill_len).map_err(|_| Error::invalid("tensor too large"))?;
        let heads_i = i32::try_from(heads).map_err(|_| Error::invalid("tensor too large"))?;
        let dim_i = i32::try_from(head_dim).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let q_slice = q.storage().f32()?;
        let qh_slice = storage_mut(qh)?.f32_mut()?;
        let mut builder = runtime
            .stream
            .launch_builder(&runtime.kernels.split_heads_prefill);
        builder
            .arg(q_slice)
            .arg(qh_slice)
            .arg(&batch_i)
            .arg(&len_i)
            .arg(&heads_i)
            .arg(&dim_i);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(total as u32))
                .map_err(|error| Error::Backend(format!("static split heads prefill: {error}")))?;
        }
        Ok(())
    }

    pub fn merge_heads_prefill(
        &self,
        attn: &Tensor<CudaBackend>,
        out: &mut Tensor<CudaBackend>,
        batch: usize,
        prefill_len: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        let total = batch * prefill_len * heads * head_dim;
        let batch_i = i32::try_from(batch).map_err(|_| Error::invalid("tensor too large"))?;
        let len_i = i32::try_from(prefill_len).map_err(|_| Error::invalid("tensor too large"))?;
        let heads_i = i32::try_from(heads).map_err(|_| Error::invalid("tensor too large"))?;
        let dim_i = i32::try_from(head_dim).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let attn_slice = attn.storage().f32()?;
        let out_slice = storage_mut(out)?.f32_mut()?;
        let mut builder = runtime
            .stream
            .launch_builder(&runtime.kernels.merge_heads_prefill);
        builder
            .arg(attn_slice)
            .arg(out_slice)
            .arg(&batch_i)
            .arg(&len_i)
            .arg(&heads_i)
            .arg(&dim_i);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(total as u32))
                .map_err(|error| Error::Backend(format!("static merge heads prefill: {error}")))?;
        }
        Ok(())
    }

    pub fn causal_softmax(
        &self,
        scores: &mut Tensor<CudaBackend>,
        rows: usize,
        len: usize,
    ) -> Result<()> {
        let rows_i = i32::try_from(rows).map_err(|_| Error::invalid("tensor too large"))?;
        let len_i = i32::try_from(len).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let scores = storage_mut(scores)?.f32_mut()?;
        let mut builder = runtime
            .stream
            .launch_builder(&runtime.kernels.causal_softmax_warp);
        builder.arg(scores).arg(&rows_i).arg(&len_i);
        unsafe {
            builder
                .launch(LaunchConfig {
                    grid_dim: ((rows as u32).div_ceil(4), 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|error| Error::Backend(format!("static causal softmax: {error}")))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prefill_scores(
        &self,
        qh: &Tensor<CudaBackend>,
        k_cache: &Tensor<CudaBackend>,
        scores: &mut Tensor<CudaBackend>,
        batch_heads: usize,
        prefill_len: usize,
        k_stride: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<()> {
        let qh = qh.storage().f32()?;
        let k_cache = k_cache.storage().f32()?;
        let scores = storage_mut(scores)?.f32_mut()?;
        let len_i = i32::try_from(prefill_len).map_err(|_| Error::invalid("tensor too large"))?;
        let dim_i = i32::try_from(head_dim).map_err(|_| Error::invalid("tensor too large"))?;
        let bh_i = i32::try_from(batch_heads).map_err(|_| Error::invalid("tensor too large"))?;
        let q_stride = (prefill_len * head_dim) as i64;
        let k_stride = i64::try_from(k_stride).map_err(|_| Error::invalid("tensor too large"))?;
        let c_stride = (prefill_len * prefill_len) as i64;
        unsafe {
            self.backend.runtime.blas.gemm_strided_batched(
                StridedBatchedConfig {
                    gemm: GemmConfig {
                        transa: cublasOperation_t::CUBLAS_OP_T,
                        transb: cublasOperation_t::CUBLAS_OP_N,
                        m: len_i,
                        n: len_i,
                        k: dim_i,
                        alpha: scale,
                        lda: dim_i,
                        ldb: dim_i,
                        ldc: len_i,
                        beta: 0.0,
                    },
                    batch_size: bh_i,
                    stride_a: k_stride,
                    stride_b: q_stride,
                    stride_c: c_stride,
                },
                k_cache,
                qh,
                scores,
            )
        }
        .map_err(|error| Error::Backend(format!("static prefill scores: {error}")))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prefill_out(
        &self,
        scores: &Tensor<CudaBackend>,
        v_cache: &Tensor<CudaBackend>,
        attn: &mut Tensor<CudaBackend>,
        batch_heads: usize,
        prefill_len: usize,
        v_stride: usize,
        head_dim: usize,
    ) -> Result<()> {
        let scores = scores.storage().f32()?;
        let v_cache = v_cache.storage().f32()?;
        let attn = storage_mut(attn)?.f32_mut()?;
        let len_i = i32::try_from(prefill_len).map_err(|_| Error::invalid("tensor too large"))?;
        let dim_i = i32::try_from(head_dim).map_err(|_| Error::invalid("tensor too large"))?;
        let bh_i = i32::try_from(batch_heads).map_err(|_| Error::invalid("tensor too large"))?;
        let b_stride = (prefill_len * prefill_len) as i64;
        let v_stride = i64::try_from(v_stride).map_err(|_| Error::invalid("tensor too large"))?;
        let c_stride = (prefill_len * head_dim) as i64;
        unsafe {
            self.backend.runtime.blas.gemm_strided_batched(
                StridedBatchedConfig {
                    gemm: GemmConfig {
                        transa: cublasOperation_t::CUBLAS_OP_N,
                        transb: cublasOperation_t::CUBLAS_OP_N,
                        m: dim_i,
                        n: len_i,
                        k: len_i,
                        alpha: 1.0,
                        lda: dim_i,
                        ldb: len_i,
                        ldc: dim_i,
                        beta: 0.0,
                    },
                    batch_size: bh_i,
                    stride_a: v_stride,
                    stride_b: b_stride,
                    stride_c: c_stride,
                },
                v_cache,
                scores,
                attn,
            )
        }
        .map_err(|error| Error::Backend(format!("static prefill out: {error}")))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_prefill(
        &self,
        q: &Tensor<CudaBackend>,
        k_cache: &Tensor<CudaBackend>,
        v_cache: &Tensor<CudaBackend>,
        out: &mut Tensor<CudaBackend>,
        batch: usize,
        prefill_len: usize,
        heads: usize,
        max_seq: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<()> {
        let total = batch * heads * prefill_len;
        let batch_i = i32::try_from(batch).map_err(|_| Error::invalid("tensor too large"))?;
        let len_i = i32::try_from(prefill_len).map_err(|_| Error::invalid("tensor too large"))?;
        let heads_i = i32::try_from(heads).map_err(|_| Error::invalid("tensor too large"))?;
        let seq_i = i32::try_from(max_seq).map_err(|_| Error::invalid("tensor too large"))?;
        let dim_i = i32::try_from(head_dim).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let q_slice = q.storage().f32()?;
        let k_slice = k_cache.storage().f32()?;
        let v_slice = v_cache.storage().f32()?;
        let out_slice = storage_mut(out)?.f32_mut()?;
        let mut builder = runtime
            .stream
            .launch_builder(&runtime.kernels.attention_prefill);
        builder
            .arg(q_slice)
            .arg(k_slice)
            .arg(v_slice)
            .arg(out_slice)
            .arg(&batch_i)
            .arg(&len_i)
            .arg(&heads_i)
            .arg(&seq_i)
            .arg(&dim_i)
            .arg(&scale);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(total as u32))
                .map_err(|error| Error::Backend(format!("static attention prefill: {error}")))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn kv_append_batch(
        &self,
        k: &Tensor<CudaBackend>,
        v: &Tensor<CudaBackend>,
        k_cache: &mut Tensor<CudaBackend>,
        v_cache: &mut Tensor<CudaBackend>,
        position: &CudaSlice<i32>,
        batch: usize,
        heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> Result<()> {
        let total = batch * heads * head_dim;
        let batch_i = i32::try_from(batch).map_err(|_| Error::invalid("tensor too large"))?;
        let heads_i = i32::try_from(heads).map_err(|_| Error::invalid("tensor too large"))?;
        let dim_i = i32::try_from(head_dim).map_err(|_| Error::invalid("tensor too large"))?;
        let seq_i = i32::try_from(max_seq).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let k_slice = k.storage().f32()?;
        let v_slice = v.storage().f32()?;
        let k_cache_slice = storage_mut(k_cache)?.f32_mut()?;
        let v_cache_slice = storage_mut(v_cache)?.f32_mut()?;
        let mut builder = runtime
            .stream
            .launch_builder(&runtime.kernels.kv_append_batch);
        builder
            .arg(k_slice)
            .arg(v_slice)
            .arg(k_cache_slice)
            .arg(v_cache_slice)
            .arg(position)
            .arg(&batch_i)
            .arg(&heads_i)
            .arg(&dim_i)
            .arg(&seq_i);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(total as u32))
                .map_err(|error| Error::Backend(format!("static kv append batch: {error}")))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_batch_decode(
        &self,
        q: &Tensor<CudaBackend>,
        k_cache: &Tensor<CudaBackend>,
        v_cache: &Tensor<CudaBackend>,
        position: &CudaSlice<i32>,
        out: &mut Tensor<CudaBackend>,
        batch: usize,
        heads: usize,
        max_seq: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<()> {
        let total = batch * heads;
        let batch_i = i32::try_from(batch).map_err(|_| Error::invalid("tensor too large"))?;
        let heads_i = i32::try_from(heads).map_err(|_| Error::invalid("tensor too large"))?;
        let seq_i = i32::try_from(max_seq).map_err(|_| Error::invalid("tensor too large"))?;
        let dim_i = i32::try_from(head_dim).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let q_slice = q.storage().f32()?;
        let k_slice = k_cache.storage().f32()?;
        let v_slice = v_cache.storage().f32()?;
        let out_slice = storage_mut(out)?.f32_mut()?;
        let wide = head_dim <= 256;
        let kernel = if wide {
            &runtime.kernels.attention_batch_decode_wide
        } else {
            &runtime.kernels.attention_batch_decode
        };
        let mut builder = runtime.stream.launch_builder(kernel);
        builder
            .arg(q_slice)
            .arg(k_slice)
            .arg(v_slice)
            .arg(position)
            .arg(out_slice)
            .arg(&batch_i)
            .arg(&heads_i)
            .arg(&seq_i)
            .arg(&dim_i)
            .arg(&scale);
        unsafe {
            let config = if wide {
                LaunchConfig {
                    grid_dim: ((total as u32).div_ceil(4), 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                }
            } else {
                LaunchConfig::for_num_elems(total as u32)
            };
            builder
                .launch(config)
                .map_err(|error| Error::Backend(format!("static attention batch: {error}")))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn flash_attention_prefill(
        &self,
        q: &Tensor<CudaBackend>,
        k_cache: &Tensor<CudaBackend>,
        v_cache: &Tensor<CudaBackend>,
        out: &mut Tensor<CudaBackend>,
        batch_heads: usize,
        prefill_len: usize,
        max_seq: usize,
        head_dim: usize,
        scale: f32,
        dtype: DType,
    ) -> Result<()> {
        if head_dim > 128 {
            return Err(Error::invalid("flash prefill requires head_dim <= 128"));
        }
        let tq = 8usize;
        let tk = 32usize;
        let shared = (((2 * tq + 2 * tk) * head_dim + tq * tk + 2 * tq) * 4) as u32;
        let bh_i = i32::try_from(batch_heads).map_err(|_| Error::invalid("tensor too large"))?;
        let len_i = i32::try_from(prefill_len).map_err(|_| Error::invalid("tensor too large"))?;
        let seq_i = i32::try_from(max_seq).map_err(|_| Error::invalid("tensor too large"))?;
        let dim_i = i32::try_from(head_dim).map_err(|_| Error::invalid("tensor too large"))?;
        let dtype_code = match dtype {
            DType::F32 => 0i32,
            DType::F16 => 1,
            DType::BF16 => 2,
            _ => return Err(Error::invalid("unsupported KV dtype")),
        };
        let runtime = self.backend.runtime.clone();
        let q_slice = q.storage().f32()?;
        let out_slice = storage_mut(out)?.f32_mut()?;
        macro_rules! launch {
            ($k:expr, $v:expr) => {{
                let mut builder = runtime
                    .stream
                    .launch_builder(&runtime.kernels.flash_attention_prefill);
                builder
                    .arg(q_slice)
                    .arg($k)
                    .arg($v)
                    .arg(out_slice)
                    .arg(&bh_i)
                    .arg(&len_i)
                    .arg(&seq_i)
                    .arg(&dim_i)
                    .arg(&scale)
                    .arg(&dtype_code);
                unsafe {
                    builder
                        .launch(LaunchConfig {
                            grid_dim: (
                                (prefill_len as u32).div_ceil(tq as u32),
                                batch_heads as u32,
                                1,
                            ),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: shared,
                        })
                        .map_err(|error| {
                            Error::Backend(format!("static flash prefill: {error}"))
                        })?;
                }
            }};
        }
        match dtype {
            DType::F32 => {
                launch!(k_cache.storage().f32()?, v_cache.storage().f32()?);
            }
            DType::F16 => {
                launch!(k_cache.storage().f16()?, v_cache.storage().f16()?);
            }
            DType::BF16 => {
                launch!(k_cache.storage().bf16()?, v_cache.storage().bf16()?);
            }
            _ => return Err(Error::invalid("unsupported KV dtype")),
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn flash_attention_decode_lowp(
        &self,
        q: &Tensor<CudaBackend>,
        k_cache: &Tensor<CudaBackend>,
        v_cache: &Tensor<CudaBackend>,
        positions: &CudaSlice<i32>,
        out: &mut Tensor<CudaBackend>,
        batch: usize,
        heads: usize,
        max_seq: usize,
        head_dim: usize,
        scale: f32,
        dtype: DType,
    ) -> Result<()> {
        if head_dim > 256 {
            return Err(Error::invalid("flash decode requires head_dim <= 256"));
        }
        let shared = ((8 * head_dim + 16) * 4) as u32;
        let batch_i = i32::try_from(batch).map_err(|_| Error::invalid("tensor too large"))?;
        let heads_i = i32::try_from(heads).map_err(|_| Error::invalid("tensor too large"))?;
        let seq_i = i32::try_from(max_seq).map_err(|_| Error::invalid("tensor too large"))?;
        let dim_i = i32::try_from(head_dim).map_err(|_| Error::invalid("tensor too large"))?;
        let dtype_code = match dtype {
            DType::F16 => 1i32,
            DType::BF16 => 2,
            _ => return Err(Error::invalid("flash decode requires low-precision KV")),
        };
        let runtime = self.backend.runtime.clone();
        let q_slice = q.storage().f32()?;
        let out_slice = storage_mut(out)?.f32_mut()?;
        macro_rules! launch {
            ($k:expr, $v:expr) => {{
                let mut builder = runtime
                    .stream
                    .launch_builder(&runtime.kernels.flash_attention_decode_lowp);
                builder
                    .arg(q_slice)
                    .arg($k)
                    .arg($v)
                    .arg(positions)
                    .arg(out_slice)
                    .arg(&batch_i)
                    .arg(&heads_i)
                    .arg(&seq_i)
                    .arg(&dim_i)
                    .arg(&scale)
                    .arg(&dtype_code);
                unsafe {
                    builder
                        .launch(LaunchConfig {
                            grid_dim: ((batch * heads) as u32, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: shared,
                        })
                        .map_err(|error| Error::Backend(format!("static flash decode: {error}")))?;
                }
            }};
        }
        match dtype {
            DType::F16 => launch!(k_cache.storage().f16()?, v_cache.storage().f16()?),
            DType::BF16 => launch!(k_cache.storage().bf16()?, v_cache.storage().bf16()?),
            _ => return Err(Error::invalid("flash decode requires low-precision KV")),
        }
        Ok(())
    }

    pub fn copy_i32(
        &self,
        src: &CudaSlice<i32>,
        dst: &mut CudaSlice<i32>,
        len: usize,
    ) -> Result<()> {
        let len_i = i32::try_from(len).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let mut builder = runtime.stream.launch_builder(&runtime.kernels.copy_i32);
        builder.arg(src).arg(dst).arg(&len_i);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(len as u32))
                .map_err(|error| Error::Backend(format!("static copy i32: {error}")))?;
        }
        Ok(())
    }

    pub fn gather_last_rows_var(
        &self,
        x: &Tensor<CudaBackend>,
        lengths: &CudaSlice<i32>,
        out: &mut Tensor<CudaBackend>,
        batch: usize,
        rows: usize,
        dim: usize,
    ) -> Result<()> {
        let total = batch * dim;
        let batch_i = i32::try_from(batch).map_err(|_| Error::invalid("tensor too large"))?;
        let rows_i = i32::try_from(rows).map_err(|_| Error::invalid("tensor too large"))?;
        let dim_i = i32::try_from(dim).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let x_slice = x.storage().f32()?;
        let out_slice = storage_mut(out)?.f32_mut()?;
        let mut builder = runtime
            .stream
            .launch_builder(&runtime.kernels.gather_last_rows_var);
        builder
            .arg(x_slice)
            .arg(lengths)
            .arg(out_slice)
            .arg(&batch_i)
            .arg(&rows_i)
            .arg(&dim_i);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(total as u32))
                .map_err(|error| Error::Backend(format!("static gather last rows var: {error}")))?;
        }
        Ok(())
    }

    pub fn gather_last_rows(
        &self,
        x: &Tensor<CudaBackend>,
        out: &mut Tensor<CudaBackend>,
        batch: usize,
        rows: usize,
        dim: usize,
    ) -> Result<()> {
        let total = batch * dim;
        let batch_i = i32::try_from(batch).map_err(|_| Error::invalid("tensor too large"))?;
        let rows_i = i32::try_from(rows).map_err(|_| Error::invalid("tensor too large"))?;
        let dim_i = i32::try_from(dim).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let x_slice = x.storage().f32()?;
        let out_slice = storage_mut(out)?.f32_mut()?;
        let mut builder = runtime
            .stream
            .launch_builder(&runtime.kernels.gather_last_rows);
        builder
            .arg(x_slice)
            .arg(out_slice)
            .arg(&batch_i)
            .arg(&rows_i)
            .arg(&dim_i);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(total as u32))
                .map_err(|error| Error::Backend(format!("static gather last rows: {error}")))?;
        }
        Ok(())
    }

    pub fn split_heads_decode(
        &self,
        q: &Tensor<CudaBackend>,
        qh: &mut Tensor<CudaBackend>,
        batch: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        let total = batch * heads * head_dim;
        let batch_i = i32::try_from(batch).map_err(|_| Error::invalid("tensor too large"))?;
        let heads_i = i32::try_from(heads).map_err(|_| Error::invalid("tensor too large"))?;
        let dim_i = i32::try_from(head_dim).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let q_slice = q.storage().f32()?;
        let qh_slice = storage_mut(qh)?.f32_mut()?;
        let mut builder = runtime
            .stream
            .launch_builder(&runtime.kernels.split_heads_decode);
        builder
            .arg(q_slice)
            .arg(qh_slice)
            .arg(&batch_i)
            .arg(&heads_i)
            .arg(&dim_i);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(total as u32))
                .map_err(|error| Error::Backend(format!("static split heads decode: {error}")))?;
        }
        Ok(())
    }

    pub fn merge_heads_decode(
        &self,
        attn: &Tensor<CudaBackend>,
        out: &mut Tensor<CudaBackend>,
        batch: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        let total = batch * heads * head_dim;
        let batch_i = i32::try_from(batch).map_err(|_| Error::invalid("tensor too large"))?;
        let heads_i = i32::try_from(heads).map_err(|_| Error::invalid("tensor too large"))?;
        let dim_i = i32::try_from(head_dim).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let attn_slice = attn.storage().f32()?;
        let out_slice = storage_mut(out)?.f32_mut()?;
        let mut builder = runtime
            .stream
            .launch_builder(&runtime.kernels.merge_heads_decode);
        builder
            .arg(attn_slice)
            .arg(out_slice)
            .arg(&batch_i)
            .arg(&heads_i)
            .arg(&dim_i);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(total as u32))
                .map_err(|error| Error::Backend(format!("static merge heads decode: {error}")))?;
        }
        Ok(())
    }

    pub fn decode_softmax(
        &self,
        scores: &mut Tensor<CudaBackend>,
        positions: &CudaSlice<i32>,
        batch: usize,
        heads: usize,
        max_seq: usize,
    ) -> Result<()> {
        let batch_i = i32::try_from(batch).map_err(|_| Error::invalid("tensor too large"))?;
        let heads_i = i32::try_from(heads).map_err(|_| Error::invalid("tensor too large"))?;
        let seq_i = i32::try_from(max_seq).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let scores = storage_mut(scores)?.f32_mut()?;
        let mut builder = runtime
            .stream
            .launch_builder(&runtime.kernels.decode_softmax);
        builder
            .arg(scores)
            .arg(positions)
            .arg(&batch_i)
            .arg(&heads_i)
            .arg(&seq_i);
        unsafe {
            builder
                .launch(LaunchConfig {
                    grid_dim: ((batch * heads) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|error| Error::Backend(format!("static decode softmax: {error}")))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn decode_scores(
        &self,
        qh: &Tensor<CudaBackend>,
        k_cache: &Tensor<CudaBackend>,
        scores: &mut Tensor<CudaBackend>,
        batch_heads: usize,
        max_seq: usize,
        k_stride: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<()> {
        let qh = qh.storage().f32()?;
        let k_cache = k_cache.storage().f32()?;
        let scores = storage_mut(scores)?.f32_mut()?;
        let seq_i = i32::try_from(max_seq).map_err(|_| Error::invalid("tensor too large"))?;
        let dim_i = i32::try_from(head_dim).map_err(|_| Error::invalid("tensor too large"))?;
        let bh_i = i32::try_from(batch_heads).map_err(|_| Error::invalid("tensor too large"))?;
        let k_stride = i64::try_from(k_stride).map_err(|_| Error::invalid("tensor too large"))?;
        let q_stride = head_dim as i64;
        let c_stride = max_seq as i64;
        unsafe {
            self.backend.runtime.blas.gemm_strided_batched(
                StridedBatchedConfig {
                    gemm: GemmConfig {
                        transa: cublasOperation_t::CUBLAS_OP_T,
                        transb: cublasOperation_t::CUBLAS_OP_N,
                        m: seq_i,
                        n: 1,
                        k: dim_i,
                        alpha: scale,
                        lda: dim_i,
                        ldb: dim_i,
                        ldc: seq_i,
                        beta: 0.0,
                    },
                    batch_size: bh_i,
                    stride_a: k_stride,
                    stride_b: q_stride,
                    stride_c: c_stride,
                },
                k_cache,
                qh,
                scores,
            )
        }
        .map_err(|error| Error::Backend(format!("static decode scores: {error}")))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn decode_out(
        &self,
        scores: &Tensor<CudaBackend>,
        v_cache: &Tensor<CudaBackend>,
        attn: &mut Tensor<CudaBackend>,
        batch_heads: usize,
        max_seq: usize,
        v_stride: usize,
        head_dim: usize,
    ) -> Result<()> {
        let scores = scores.storage().f32()?;
        let v_cache = v_cache.storage().f32()?;
        let attn = storage_mut(attn)?.f32_mut()?;
        let seq_i = i32::try_from(max_seq).map_err(|_| Error::invalid("tensor too large"))?;
        let dim_i = i32::try_from(head_dim).map_err(|_| Error::invalid("tensor too large"))?;
        let bh_i = i32::try_from(batch_heads).map_err(|_| Error::invalid("tensor too large"))?;
        let v_stride = i64::try_from(v_stride).map_err(|_| Error::invalid("tensor too large"))?;
        let s_stride = max_seq as i64;
        let c_stride = head_dim as i64;
        unsafe {
            self.backend.runtime.blas.gemm_strided_batched(
                StridedBatchedConfig {
                    gemm: GemmConfig {
                        transa: cublasOperation_t::CUBLAS_OP_N,
                        transb: cublasOperation_t::CUBLAS_OP_N,
                        m: dim_i,
                        n: 1,
                        k: seq_i,
                        alpha: 1.0,
                        lda: dim_i,
                        ldb: seq_i,
                        ldc: dim_i,
                        beta: 0.0,
                    },
                    batch_size: bh_i,
                    stride_a: v_stride,
                    stride_b: s_stride,
                    stride_c: c_stride,
                },
                v_cache,
                scores,
                attn,
            )
        }
        .map_err(|error| Error::Backend(format!("static decode out: {error}")))
    }

    pub fn set_position(&self, position: &mut CudaSlice<i32>, value: i32) -> Result<()> {
        let runtime = self.backend.runtime.clone();
        let mut builder = runtime
            .stream
            .launch_builder(&runtime.kernels.set_i32_value);
        builder.arg(position).arg(&value);
        unsafe {
            builder
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|error| Error::Backend(format!("static set position: {error}")))?;
        }
        Ok(())
    }

    pub fn fill_positions(
        &self,
        positions: &mut CudaSlice<i32>,
        value: i32,
        len: usize,
    ) -> Result<()> {
        let len_i = i32::try_from(len).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let mut builder = runtime
            .stream
            .launch_builder(&runtime.kernels.fill_i32_value);
        builder.arg(positions).arg(&value).arg(&len_i);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(len as u32))
                .map_err(|error| Error::Backend(format!("static fill positions: {error}")))?;
        }
        Ok(())
    }

    pub fn bump_positions(&self, positions: &mut CudaSlice<i32>, len: usize) -> Result<()> {
        let len_i = i32::try_from(len).map_err(|_| Error::invalid("tensor too large"))?;
        let runtime = self.backend.runtime.clone();
        let mut builder = runtime
            .stream
            .launch_builder(&runtime.kernels.bump_positions);
        builder.arg(positions).arg(&len_i);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(len as u32))
                .map_err(|error| Error::Backend(format!("static bump positions: {error}")))?;
        }
        Ok(())
    }
}

/// A model that declares fixed buffers and enqueues an in-place step.
pub trait CudaStaticStep {
    type Buffers;
    type Input;
    type Output;

    fn allocate(&self, builder: &mut StaticBuilder) -> Result<Self::Buffers>;
    fn write_input(
        &self,
        ctx: &CudaStaticContext,
        buffers: &mut Self::Buffers,
        input: &Self::Input,
    ) -> Result<()>;
    fn enqueue(&self, ctx: &CudaStaticContext, buffers: &mut Self::Buffers) -> Result<()>;
    fn read_output(&self, ctx: &CudaStaticContext, buffers: &Self::Buffers)
    -> Result<Self::Output>;
}

pub struct StaticRun<M: CudaStaticStep> {
    model: M,
    backend: CudaBackend,
    ctx: CudaStaticContext,
    buffers: M::Buffers,
    graph: CudaGraphRunner,
}

impl<M: CudaStaticStep> StaticRun<M> {
    pub fn new(model: M, backend: CudaBackend) -> Result<Self> {
        let mut builder = StaticBuilder::new(backend.clone());
        let mut buffers = model.allocate(&mut builder)?;
        let ctx = CudaStaticContext::new(backend.clone());
        let graph = backend.capture(|| model.enqueue(&ctx, &mut buffers))?;
        Ok(Self {
            model,
            backend,
            ctx,
            buffers,
            graph,
        })
    }

    pub fn replay(&mut self, input: &M::Input) -> Result<()> {
        self.model
            .write_input(&self.ctx, &mut self.buffers, input)?;
        self.graph.replay()
    }

    pub fn output(&self) -> Result<M::Output> {
        self.model.read_output(&self.ctx, &self.buffers)
    }

    pub fn synchronize(&self) -> Result<()> {
        self.backend.synchronize()
    }

    pub fn model(&self) -> &M {
        &self.model
    }
}

#[derive(Clone, Debug)]
pub struct StaticLinearConfig {
    pub batch: usize,
    pub in_features: usize,
    pub out_features: usize,
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
}

#[derive(Clone, Debug)]
pub struct StaticLinearInput {
    pub x: Vec<f32>,
    pub target: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct StaticLinearOutput {
    pub loss: f32,
    pub step: i32,
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

pub struct LinearMseBuffers {
    w: Tensor<CudaBackend>,
    b: Tensor<CudaBackend>,
    x: Tensor<CudaBackend>,
    target: Tensor<CudaBackend>,
    y: Tensor<CudaBackend>,
    diff: Tensor<CudaBackend>,
    loss: Tensor<CudaBackend>,
    grad_y: Tensor<CudaBackend>,
    grad_w: Tensor<CudaBackend>,
    grad_b: Tensor<CudaBackend>,
    m_w: Tensor<CudaBackend>,
    v_w: Tensor<CudaBackend>,
    m_b: Tensor<CudaBackend>,
    v_b: Tensor<CudaBackend>,
    step: CudaSlice<i32>,
}

pub struct LinearMseAdamStep {
    config: StaticLinearConfig,
    seed: u64,
}

impl LinearMseAdamStep {
    pub fn new(config: StaticLinearConfig, seed: u64) -> Result<Self> {
        if config.batch == 0
            || config.in_features == 0
            || config.out_features == 0
            || config.lr <= 0.0
        {
            return Err(Error::invalid("invalid static linear config"));
        }
        Ok(Self { config, seed })
    }

    pub fn config(&self) -> &StaticLinearConfig {
        &self.config
    }
}

impl CudaStaticStep for LinearMseAdamStep {
    type Buffers = LinearMseBuffers;
    type Input = StaticLinearInput;
    type Output = StaticLinearOutput;

    fn allocate(&self, builder: &mut StaticBuilder) -> Result<Self::Buffers> {
        let cfg = &self.config;
        let weight = modern_chainer_core::Rng::seed_from_u64(self.seed).normal_vec(
            cfg.out_features * cfg.in_features,
            0.0,
            (2.0 / cfg.in_features as f64).sqrt(),
        );
        Ok(LinearMseBuffers {
            w: builder.from_f32([cfg.out_features, cfg.in_features], &weight)?,
            b: builder.zeros([cfg.out_features], DType::F32)?,
            x: builder.zeros([cfg.batch, cfg.in_features], DType::F32)?,
            target: builder.zeros([cfg.batch, cfg.out_features], DType::F32)?,
            y: builder.zeros([cfg.batch, cfg.out_features], DType::F32)?,
            diff: builder.zeros([cfg.batch, cfg.out_features], DType::F32)?,
            loss: builder.zeros(Shape::scalar(), DType::F32)?,
            grad_y: builder.zeros([cfg.batch, cfg.out_features], DType::F32)?,
            grad_w: builder.zeros([cfg.out_features, cfg.in_features], DType::F32)?,
            grad_b: builder.zeros([cfg.out_features], DType::F32)?,
            m_w: builder.zeros([cfg.out_features, cfg.in_features], DType::F32)?,
            v_w: builder.zeros([cfg.out_features, cfg.in_features], DType::F32)?,
            m_b: builder.zeros([cfg.out_features], DType::F32)?,
            v_b: builder.zeros([cfg.out_features], DType::F32)?,
            step: builder.i32_scalar()?,
        })
    }

    fn write_input(
        &self,
        ctx: &CudaStaticContext,
        buffers: &mut Self::Buffers,
        input: &Self::Input,
    ) -> Result<()> {
        ctx.write_f32(&mut buffers.x, &input.x)?;
        ctx.write_f32(&mut buffers.target, &input.target)
    }

    fn enqueue(&self, ctx: &CudaStaticContext, buffers: &mut Self::Buffers) -> Result<()> {
        let cfg = &self.config;
        ctx.linear_into(&buffers.x, &buffers.w, Some(&buffers.b), &mut buffers.y)?;
        ctx.mse_into(
            &buffers.y,
            &buffers.target,
            &mut buffers.diff,
            &mut buffers.loss,
        )?;
        let scale = 2.0 / (cfg.batch * cfg.out_features) as f32;
        ctx.grad_y_into(&buffers.diff, &mut buffers.grad_y, scale)?;
        ctx.matmul_grad_w_into(&buffers.x, &buffers.grad_y, &mut buffers.grad_w)?;
        ctx.grad_b_into(
            &buffers.grad_y,
            &mut buffers.grad_b,
            cfg.batch,
            cfg.out_features,
        )?;
        ctx.bump_step(&mut buffers.step)?;
        let adam = AdamConfig {
            lr: cfg.lr,
            beta1: cfg.beta1,
            beta2: cfg.beta2,
            eps: cfg.eps,
            step: 0,
        };
        ctx.adam_into(
            &mut buffers.w,
            &buffers.grad_w,
            &mut buffers.m_w,
            &mut buffers.v_w,
            &buffers.step,
            &adam,
        )?;
        ctx.adam_into(
            &mut buffers.b,
            &buffers.grad_b,
            &mut buffers.m_b,
            &mut buffers.v_b,
            &buffers.step,
            &adam,
        )
    }

    fn read_output(
        &self,
        ctx: &CudaStaticContext,
        buffers: &Self::Buffers,
    ) -> Result<Self::Output> {
        Ok(StaticLinearOutput {
            loss: ctx.read_f32(&buffers.loss)?,
            step: ctx.read_i32(&buffers.step)?,
            weight: ctx.read_f32_vec(&buffers.w)?,
            bias: ctx.read_f32_vec(&buffers.b)?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct StaticForwardConfig {
    pub batch: usize,
    pub in_features: usize,
    pub out_features: usize,
}

pub struct StaticForwardInput {
    pub x: Vec<f32>,
}

pub struct LinearForwardStep {
    config: StaticForwardConfig,
    seed: u64,
}

pub struct LinearForwardBuffers {
    w: Tensor<CudaBackend>,
    b: Tensor<CudaBackend>,
    x: Tensor<CudaBackend>,
    y: Tensor<CudaBackend>,
}

impl LinearForwardStep {
    pub fn new(config: StaticForwardConfig, seed: u64) -> Result<Self> {
        if config.batch == 0 || config.in_features == 0 || config.out_features == 0 {
            return Err(Error::invalid("invalid static forward config"));
        }
        Ok(Self { config, seed })
    }
}

impl CudaStaticStep for LinearForwardStep {
    type Buffers = LinearForwardBuffers;
    type Input = StaticForwardInput;
    type Output = Vec<f32>;

    fn allocate(&self, builder: &mut StaticBuilder) -> Result<Self::Buffers> {
        let weight = modern_chainer_core::Rng::seed_from_u64(self.seed).normal_vec(
            self.config.out_features * self.config.in_features,
            0.0,
            (2.0 / self.config.in_features as f64).sqrt(),
        );
        Ok(LinearForwardBuffers {
            w: builder.from_f32([self.config.out_features, self.config.in_features], &weight)?,
            b: builder.zeros([self.config.out_features], DType::F32)?,
            x: builder.zeros([self.config.batch, self.config.in_features], DType::F32)?,
            y: builder.zeros([self.config.batch, self.config.out_features], DType::F32)?,
        })
    }

    fn write_input(
        &self,
        ctx: &CudaStaticContext,
        buffers: &mut Self::Buffers,
        input: &Self::Input,
    ) -> Result<()> {
        ctx.write_f32(&mut buffers.x, &input.x)
    }

    fn enqueue(&self, ctx: &CudaStaticContext, buffers: &mut Self::Buffers) -> Result<()> {
        ctx.linear_into(&buffers.x, &buffers.w, Some(&buffers.b), &mut buffers.y)
    }

    fn read_output(
        &self,
        ctx: &CudaStaticContext,
        buffers: &Self::Buffers,
    ) -> Result<Self::Output> {
        ctx.read_f32_vec(&buffers.y)
    }
}

#[derive(Clone, Debug)]
pub struct StaticDecoderInput {
    pub token: u32,
}

#[derive(Clone, Debug)]
pub struct StaticDecoderOutput {
    pub logits: Vec<f32>,
    pub position: i32,
}

pub struct StaticDecoderBuffers {
    weights: HashMap<String, Tensor<CudaBackend>>,
    kv: Vec<(Tensor<CudaBackend>, Tensor<CudaBackend>)>,
    x: Tensor<CudaBackend>,
    norm: Tensor<CudaBackend>,
    q: Tensor<CudaBackend>,
    k: Tensor<CudaBackend>,
    v: Tensor<CudaBackend>,
    attn: Tensor<CudaBackend>,
    hidden: Tensor<CudaBackend>,
    update: Tensor<CudaBackend>,
    logits: Tensor<CudaBackend>,
    token: CudaSlice<i32>,
    position: CudaSlice<i32>,
}

pub struct StaticDecoderStep {
    decoder: Decoder<CudaBackend>,
    config: DecoderConfig,
}

impl StaticDecoderStep {
    pub fn from_decoder(decoder: Decoder<CudaBackend>) -> Result<Self> {
        let config = decoder.config().clone();
        Ok(Self { decoder, config })
    }
}

impl CudaStaticStep for StaticDecoderStep {
    type Buffers = StaticDecoderBuffers;
    type Input = StaticDecoderInput;
    type Output = StaticDecoderOutput;

    fn allocate(&self, builder: &mut StaticBuilder) -> Result<Self::Buffers> {
        let cfg = &self.config;
        let mut weights = HashMap::new();
        for (name, tensor) in self.decoder.parameters() {
            weights.insert(name, builder.from_tensor(&tensor)?);
        }
        let head_dim = cfg.head_dim();
        let mut kv = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            kv.push((
                builder.zeros([cfg.n_heads, cfg.max_seq, head_dim], DType::F32)?,
                builder.zeros([cfg.n_heads, cfg.max_seq, head_dim], DType::F32)?,
            ));
        }
        Ok(StaticDecoderBuffers {
            weights,
            kv,
            x: builder.zeros([1, cfg.d_model], DType::F32)?,
            norm: builder.zeros([1, cfg.d_model], DType::F32)?,
            q: builder.zeros([1, cfg.d_model], DType::F32)?,
            k: builder.zeros([1, cfg.d_model], DType::F32)?,
            v: builder.zeros([1, cfg.d_model], DType::F32)?,
            attn: builder.zeros([1, cfg.d_model], DType::F32)?,
            hidden: builder.zeros([1, cfg.d_ff], DType::F32)?,
            update: builder.zeros([1, cfg.d_model], DType::F32)?,
            logits: builder.zeros([1, cfg.vocab], DType::F32)?,
            token: builder.i32_scalar()?,
            position: builder.i32_scalar()?,
        })
    }

    fn write_input(
        &self,
        ctx: &CudaStaticContext,
        buffers: &mut Self::Buffers,
        input: &Self::Input,
    ) -> Result<()> {
        ctx.write_i32(&mut buffers.token, input.token as i32)
    }

    fn enqueue(&self, ctx: &CudaStaticContext, buffers: &mut Self::Buffers) -> Result<()> {
        let cfg = &self.config;
        let d_model = cfg.d_model;
        let head_dim = cfg.head_dim();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let StaticDecoderBuffers {
            weights,
            kv,
            x,
            norm,
            q,
            k,
            v,
            attn,
            hidden,
            update,
            logits,
            token,
            position,
        } = buffers;
        let weight = |name: &str| -> Result<&Tensor<CudaBackend>> {
            weights
                .get(name)
                .ok_or_else(|| Error::invalid(format!("missing decoder weight {name}")))
        };

        ctx.embedding_one(token, weight("tok_emb")?, x, d_model)?;
        for (layer, (k_cache, v_cache)) in kv.iter_mut().enumerate() {
            let prefix = format!("layers.{layer}");
            let name = |suffix: &str| format!("{prefix}.{suffix}");
            ctx.layer_norm_into(
                x,
                weight(&name("ln1_g"))?,
                weight(&name("ln1_b"))?,
                norm,
                d_model,
                cfg.eps as f32,
            )?;
            ctx.linear_into(norm, weight(&name("wq"))?, Some(weight(&name("bq"))?), q)?;
            ctx.linear_into(norm, weight(&name("wk"))?, Some(weight(&name("bk"))?), k)?;
            ctx.linear_into(norm, weight(&name("wv"))?, Some(weight(&name("bv"))?), v)?;
            ctx.kv_append_one(
                k,
                v,
                k_cache,
                v_cache,
                position,
                cfg.n_heads,
                head_dim,
                cfg.max_seq,
            )?;
            ctx.attention_single(
                q,
                k_cache,
                v_cache,
                position,
                attn,
                cfg.n_heads,
                cfg.max_seq,
                head_dim,
                scale,
            )?;
            ctx.linear_into(
                attn,
                weight(&name("wo"))?,
                Some(weight(&name("bo"))?),
                update,
            )?;
            ctx.residual_inplace(x, update, d_model)?;

            ctx.layer_norm_into(
                x,
                weight(&name("ln2_g"))?,
                weight(&name("ln2_b"))?,
                norm,
                d_model,
                cfg.eps as f32,
            )?;
            ctx.linear_into(
                norm,
                weight(&name("w1"))?,
                Some(weight(&name("b1"))?),
                hidden,
            )?;
            ctx.gelu_inplace(hidden, cfg.d_ff)?;
            ctx.linear_into(
                hidden,
                weight(&name("w2"))?,
                Some(weight(&name("b2"))?),
                update,
            )?;
            ctx.residual_inplace(x, update, d_model)?;
        }
        ctx.layer_norm_into(
            x,
            weight("ln_f_g")?,
            weight("ln_f_b")?,
            norm,
            d_model,
            cfg.eps as f32,
        )?;
        ctx.linear_into(norm, weight("lm_head")?, None, logits)?;
        ctx.bump_step(position)?;
        Ok(())
    }

    fn read_output(
        &self,
        ctx: &CudaStaticContext,
        buffers: &Self::Buffers,
    ) -> Result<Self::Output> {
        Ok(StaticDecoderOutput {
            logits: ctx.read_f32_vec(&buffers.logits)?,
            position: ctx.read_i32(&buffers.position)?,
        })
    }
}

struct BatchTensors {
    x: Tensor<CudaBackend>,
    norm: Tensor<CudaBackend>,
    q: Tensor<CudaBackend>,
    k: Tensor<CudaBackend>,
    v: Tensor<CudaBackend>,
    attn: Tensor<CudaBackend>,
    hidden: Tensor<CudaBackend>,
    update: Tensor<CudaBackend>,
    logits: Tensor<CudaBackend>,
    qh: Option<Tensor<CudaBackend>>,
    attn_h: Option<Tensor<CudaBackend>>,
    scores: Option<Tensor<CudaBackend>>,
    k_work: Option<Tensor<CudaBackend>>,
    v_work: Option<Tensor<CudaBackend>>,
    last_x: Option<Tensor<CudaBackend>>,
    last_norm: Option<Tensor<CudaBackend>>,
}

impl BatchTensors {
    fn allocate(
        builder: &StaticBuilder,
        rows: usize,
        config: &DecoderConfig,
        prefill: Option<(usize, usize, bool)>,
        flash_prefill: bool,
    ) -> Result<Self> {
        let (qh, attn_h, scores, k_work, v_work) = if let Some((batch, prefill_len, lowp)) = prefill
        {
            let batch_heads = batch * config.n_heads;
            let width = config.head_dim();
            let flash = flash_prefill && width <= 128;
            let qh = builder.zeros([batch_heads * prefill_len * width], DType::F32)?;
            let attn_h = builder.zeros([batch_heads * prefill_len * width], DType::F32)?;
            let scores = if flash {
                None
            } else {
                Some(builder.zeros([batch_heads * prefill_len * prefill_len], DType::F32)?)
            };
            let k_work = if lowp && !flash {
                Some(builder.zeros([batch_heads * prefill_len * width], DType::F32)?)
            } else {
                None
            };
            let v_work = if lowp && !flash {
                Some(builder.zeros([batch_heads * prefill_len * width], DType::F32)?)
            } else {
                None
            };
            (Some(qh), Some(attn_h), scores, k_work, v_work)
        } else {
            (None, None, None, None, None)
        };
        let (last_x, last_norm, logits_rows) = if let Some((batch, _, _)) = prefill {
            (
                Some(builder.zeros([batch, config.d_model], DType::F32)?),
                Some(builder.zeros([batch, config.d_model], DType::F32)?),
                batch,
            )
        } else {
            (None, None, rows)
        };
        Ok(Self {
            x: builder.zeros([rows, config.d_model], DType::F32)?,
            norm: builder.zeros([rows, config.d_model], DType::F32)?,
            q: builder.zeros([rows, config.d_model], DType::F32)?,
            k: builder.zeros([rows, config.d_model], DType::F32)?,
            v: builder.zeros([rows, config.d_model], DType::F32)?,
            attn: builder.zeros([rows, config.d_model], DType::F32)?,
            hidden: builder.zeros([rows, config.d_ff], DType::F32)?,
            update: builder.zeros([rows, config.d_model], DType::F32)?,
            logits: builder.zeros([logits_rows, config.vocab], DType::F32)?,
            qh,
            attn_h,
            scores,
            k_work,
            v_work,
            last_x,
            last_norm,
        })
    }
}

struct DecodeWorkspace {
    qh: Tensor<CudaBackend>,
    attn_h: Tensor<CudaBackend>,
    scores: Tensor<CudaBackend>,
    k_work: Option<Tensor<CudaBackend>>,
    v_work: Option<Tensor<CudaBackend>>,
}

impl DecodeWorkspace {
    fn allocate(
        builder: &StaticBuilder,
        batch: usize,
        max_seq: usize,
        config: &DecoderConfig,
        lowp: bool,
    ) -> Result<Self> {
        let heads = config.n_heads;
        let head_dim = config.head_dim();
        let batch_heads = batch * heads;
        let k_work = if lowp {
            Some(builder.zeros([batch_heads * max_seq * head_dim], DType::F32)?)
        } else {
            None
        };
        let v_work = if lowp {
            Some(builder.zeros([batch_heads * max_seq * head_dim], DType::F32)?)
        } else {
            None
        };
        Ok(Self {
            qh: builder.zeros([batch_heads * head_dim], DType::F32)?,
            attn_h: builder.zeros([batch_heads * head_dim], DType::F32)?,
            scores: builder.zeros([batch_heads * max_seq], DType::F32)?,
            k_work,
            v_work,
        })
    }
}

/// Persistent KV cache element type. Weights and activations stay f32.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvDType {
    #[default]
    F32,
    F16,
    Bf16,
}

impl KvDType {
    pub fn dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::Bf16 => DType::BF16,
        }
    }
}

struct BatchedStaticDecoderBuffers {
    weights: HashMap<String, Tensor<CudaBackend>>,
    kv: Vec<(Tensor<CudaBackend>, Tensor<CudaBackend>)>,
    prefill_tokens: CudaSlice<i32>,
    prefill_lengths: CudaSlice<i32>,
    decode_tokens: CudaSlice<i32>,
    positions: CudaSlice<i32>,
    prefill: BatchTensors,
    decode: BatchTensors,
    decode_ws: Option<DecodeWorkspace>,
}

/// Fixed-shape batched prefill + decode with a shared KV cache.
///
/// The prefill graph processes `batch * prefill_len` tokens with causal
/// attention in one replay. The decode graph processes one token per batch
/// element at its own position and advances it. Both graphs are captured
/// once and replay without allocation.
pub struct BatchedStaticDecoder {
    config: DecoderConfig,
    batch: usize,
    prefill_len: usize,
    kv_dtype: KvDType,
    flash_prefill: bool,
    ctx: CudaStaticContext,
    buffers: BatchedStaticDecoderBuffers,
    prefill_graph: CudaGraphRunner,
    decode_graph: CudaGraphRunner,
    host_positions: Vec<usize>,
}

impl BatchedStaticDecoder {
    pub fn new(decoder: Decoder<CudaBackend>, batch: usize, prefill_len: usize) -> Result<Self> {
        Self::new_with_kv_dtype(decoder, batch, prefill_len, KvDType::F32)
    }

    pub fn new_with_kv_dtype(
        decoder: Decoder<CudaBackend>,
        batch: usize,
        prefill_len: usize,
        kv_dtype: KvDType,
    ) -> Result<Self> {
        Self::new_with_options(decoder, batch, prefill_len, kv_dtype, true)
    }

    pub fn new_with_options(
        decoder: Decoder<CudaBackend>,
        batch: usize,
        prefill_len: usize,
        kv_dtype: KvDType,
        flash_prefill: bool,
    ) -> Result<Self> {
        let config = decoder.config().clone();
        if batch == 0 || prefill_len == 0 {
            return Err(Error::invalid("batch and prefill_len must be positive"));
        }
        if prefill_len > config.max_seq {
            return Err(Error::invalid("prefill_len exceeds max_seq"));
        }
        let parameters = decoder.parameters();
        let backend = parameters
            .first()
            .map(|(_, tensor)| tensor.backend().clone())
            .ok_or_else(|| Error::invalid("decoder has no parameters"))?;
        let builder = StaticBuilder::new(backend.clone());
        let mut weights = HashMap::new();
        for (name, tensor) in &parameters {
            weights.insert(name.clone(), builder.from_tensor(tensor)?);
        }
        let head_dim = config.head_dim();
        let lowp = kv_dtype != KvDType::F32;
        let mut kv = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            kv.push((
                builder.zeros(
                    [batch, config.n_heads, config.max_seq, head_dim],
                    kv_dtype.dtype(),
                )?,
                builder.zeros(
                    [batch, config.n_heads, config.max_seq, head_dim],
                    kv_dtype.dtype(),
                )?,
            ));
        }
        let buffers = BatchedStaticDecoderBuffers {
            weights,
            kv,
            prefill_tokens: builder.i32_zeros(batch * prefill_len)?,
            prefill_lengths: builder.i32_zeros(batch)?,
            decode_tokens: builder.i32_zeros(batch)?,
            positions: builder.i32_zeros(batch)?,
            prefill: BatchTensors::allocate(
                &builder,
                batch * prefill_len,
                &config,
                Some((batch, prefill_len, lowp)),
                flash_prefill,
            )?,
            decode: BatchTensors::allocate(&builder, batch, &config, None, flash_prefill)?,
            decode_ws: if lowp && config.head_dim() <= 256 {
                None
            } else {
                Some(DecodeWorkspace::allocate(
                    &builder,
                    batch,
                    config.max_seq,
                    &config,
                    lowp,
                )?)
            },
        };
        let ctx = CudaStaticContext::new(backend.clone());
        let mut buffers = buffers;
        let prefill_graph = backend.capture(|| {
            enqueue_batched_prefill(
                &config,
                batch,
                prefill_len,
                kv_dtype,
                flash_prefill,
                &ctx,
                &mut buffers,
            )
        })?;
        let decode_graph = backend
            .capture(|| enqueue_batched_decode(&config, batch, kv_dtype, &ctx, &mut buffers))?;
        Ok(Self {
            config,
            batch,
            prefill_len,
            kv_dtype,
            flash_prefill,
            ctx,
            buffers,
            prefill_graph,
            decode_graph,
            host_positions: vec![0; batch],
        })
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    pub fn prefill_len(&self) -> usize {
        self.prefill_len
    }

    pub fn kv_dtype(&self) -> KvDType {
        self.kv_dtype
    }

    pub fn flash_prefill(&self) -> bool {
        self.flash_prefill
    }

    pub fn kv_bytes(&self) -> usize {
        let elements = self.batch
            * self.config.n_heads
            * self.config.max_seq
            * self.config.head_dim()
            * 2
            * self.config.n_layers;
        elements * self.kv_dtype.dtype().size_in_bytes()
    }

    pub fn config(&self) -> &DecoderConfig {
        &self.config
    }

    pub fn positions(&self) -> Result<Vec<usize>> {
        self.ctx
            .read_i32_vec(&self.buffers.positions)?
            .into_iter()
            .map(|position| {
                usize::try_from(position).map_err(|_| Error::invalid("negative static position"))
            })
            .collect()
    }

    pub fn position(&self) -> Result<usize> {
        self.positions()?
            .first()
            .copied()
            .ok_or_else(|| Error::invalid("empty static position"))
    }

    pub fn set_positions(&mut self, positions: &[usize]) -> Result<()> {
        if positions.len() != self.batch {
            return Err(Error::invalid("position count mismatch"));
        }
        let values: Vec<i32> = positions
            .iter()
            .map(|&position| {
                if position > self.config.max_seq {
                    return Err(Error::invalid("position exceeds max_seq"));
                }
                i32::try_from(position).map_err(|_| Error::invalid("position too large"))
            })
            .collect::<Result<_>>()?;
        self.ctx
            .write_i32_slice(&mut self.buffers.positions, &values)?;
        self.host_positions = positions.to_vec();
        Ok(())
    }

    /// Run one fixed-length prefill for every batch element and return the
    /// last-token logits as `[batch][vocab]`.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        if tokens.len() != self.batch * self.prefill_len {
            return Err(Error::invalid("prefill token count mismatch"));
        }
        let prompts: Vec<&[u32]> = tokens.chunks(self.prefill_len).collect();
        self.prefill_batch(&prompts)
    }

    /// Run a variable-length prefill. Each prompt must fit in `prefill_len`;
    /// shorter prompts are padded and their positions start at their length.
    pub fn prefill_batch(&mut self, prompts: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        if prompts.len() != self.batch {
            return Err(Error::invalid("prefill batch size mismatch"));
        }
        let mut tokens = vec![0i32; self.batch * self.prefill_len];
        let mut lengths = vec![0i32; self.batch];
        for (b, prompt) in prompts.iter().enumerate() {
            if prompt.is_empty() || prompt.len() > self.prefill_len {
                return Err(Error::invalid(format!(
                    "prompt {b} length {} is outside 1..={}",
                    prompt.len(),
                    self.prefill_len
                )));
            }
            lengths[b] = prompt.len() as i32;
            for (i, &token) in prompt.iter().enumerate() {
                tokens[b * self.prefill_len + i] = i32::try_from(token)
                    .map_err(|_| Error::invalid("token id does not fit i32"))?;
            }
        }
        self.ctx
            .write_i32_slice(&mut self.buffers.prefill_tokens, &tokens)?;
        self.ctx
            .write_i32_slice(&mut self.buffers.prefill_lengths, &lengths)?;
        self.prefill_graph.replay()?;
        self.host_positions = lengths
            .iter()
            .map(|&len| usize::try_from(len).map_err(|_| Error::invalid("negative length")))
            .collect::<Result<_>>()?;
        self.last_prefill_logits()
    }

    /// Run one batched decode step and return `[batch][vocab]` logits.
    pub fn decode(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        if tokens.len() != self.batch {
            return Err(Error::invalid("decode token count mismatch"));
        }
        if self
            .host_positions
            .iter()
            .any(|&position| position >= self.config.max_seq)
        {
            return Err(Error::invalid("static decode position exceeds max_seq"));
        }
        let tokens: Vec<i32> = tokens
            .iter()
            .map(|&token| {
                i32::try_from(token).map_err(|_| Error::invalid("token id does not fit i32"))
            })
            .collect::<Result<_>>()?;
        self.ctx
            .write_i32_slice(&mut self.buffers.decode_tokens, &tokens)?;
        self.decode_graph.replay()?;
        for position in &mut self.host_positions {
            *position += 1;
        }
        self.decode_logits()
    }

    pub fn last_prefill_logits(&self) -> Result<Vec<Vec<f32>>> {
        let logits = self.ctx.read_f32_vec(&self.buffers.prefill.logits)?;
        let vocab = self.config.vocab;
        Ok((0..self.batch)
            .map(|b| logits[b * vocab..(b + 1) * vocab].to_vec())
            .collect())
    }

    pub fn decode_logits(&self) -> Result<Vec<Vec<f32>>> {
        let logits = self.ctx.read_f32_vec(&self.buffers.decode.logits)?;
        let vocab = self.config.vocab;
        Ok((0..self.batch)
            .map(|b| logits[b * vocab..(b + 1) * vocab].to_vec())
            .collect())
    }

    pub fn synchronize(&self) -> Result<()> {
        self.ctx.backend().synchronize()
    }
}

fn weight<'a>(
    weights: &'a HashMap<String, Tensor<CudaBackend>>,
    name: &str,
) -> Result<&'a Tensor<CudaBackend>> {
    weights
        .get(name)
        .ok_or_else(|| Error::invalid(format!("missing decoder weight {name}")))
}

fn enqueue_batched_prefill(
    config: &DecoderConfig,
    batch: usize,
    prefill_len: usize,
    kv_dtype: KvDType,
    flash_prefill: bool,
    ctx: &CudaStaticContext,
    buffers: &mut BatchedStaticDecoderBuffers,
) -> Result<()> {
    let d_model = config.d_model;
    let head_dim = config.head_dim();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let rows = batch * prefill_len;
    let batch_heads = batch * config.n_heads;
    let lowp = kv_dtype != KvDType::F32;
    let BatchedStaticDecoderBuffers {
        weights,
        kv,
        prefill_tokens,
        prefill_lengths,
        positions,
        prefill,
        ..
    } = buffers;
    let BatchTensors {
        x,
        norm,
        q,
        k,
        v,
        attn,
        hidden,
        update,
        logits,
        qh,
        attn_h,
        scores,
        k_work,
        v_work,
        last_x,
        last_norm,
    } = prefill;

    ctx.embedding_batch(
        prefill_tokens,
        weight(weights, "tok_emb")?,
        x,
        rows,
        d_model,
    )?;
    for (layer, (k_cache, v_cache)) in kv.iter_mut().enumerate() {
        let prefix = format!("layers.{layer}");
        let name = |suffix: &str| format!("{prefix}.{suffix}");
        ctx.layer_norm_batch(
            x,
            weight(weights, &name("ln1_g"))?,
            weight(weights, &name("ln1_b"))?,
            norm,
            rows,
            d_model,
            config.eps as f32,
        )?;
        ctx.linear_into(
            norm,
            weight(weights, &name("wq"))?,
            Some(weight(weights, &name("bq"))?),
            q,
        )?;
        ctx.linear_into(
            norm,
            weight(weights, &name("wk"))?,
            Some(weight(weights, &name("bk"))?),
            k,
        )?;
        ctx.linear_into(
            norm,
            weight(weights, &name("wv"))?,
            Some(weight(weights, &name("bv"))?),
            v,
        )?;
        ctx.kv_prefill_typed(
            k,
            v,
            k_cache,
            v_cache,
            batch,
            prefill_len,
            config.n_heads,
            head_dim,
            config.max_seq,
            kv_dtype.dtype(),
        )?;

        let qh = qh
            .as_mut()
            .ok_or_else(|| Error::invalid("missing prefill Q workspace"))?;
        let attn_h = attn_h
            .as_mut()
            .ok_or_else(|| Error::invalid("missing prefill attention workspace"))?;
        ctx.split_heads_prefill(q, qh, batch, prefill_len, config.n_heads, head_dim)?;
        if flash_prefill && head_dim <= 128 {
            ctx.flash_attention_prefill(
                qh,
                k_cache,
                v_cache,
                attn_h,
                batch_heads,
                prefill_len,
                config.max_seq,
                head_dim,
                scale,
                kv_dtype.dtype(),
            )?;
        } else {
            let (k_used, v_used, kv_stride) = if lowp {
                let k_work = k_work
                    .as_mut()
                    .ok_or_else(|| Error::invalid("missing low-precision K workspace"))?;
                let v_work = v_work
                    .as_mut()
                    .ok_or_else(|| Error::invalid("missing low-precision V workspace"))?;
                ctx.kv_cache_to_f32_typed(
                    k_cache,
                    v_cache,
                    k_work,
                    v_work,
                    batch,
                    config.n_heads,
                    prefill_len,
                    config.max_seq,
                    head_dim,
                    kv_dtype.dtype(),
                )?;
                (
                    k_work as &Tensor<CudaBackend>,
                    v_work as &Tensor<CudaBackend>,
                    prefill_len * head_dim,
                )
            } else {
                (
                    k_cache as &Tensor<CudaBackend>,
                    v_cache as &Tensor<CudaBackend>,
                    config.max_seq * head_dim,
                )
            };
            let scores = scores
                .as_mut()
                .ok_or_else(|| Error::invalid("missing prefill score workspace"))?;
            ctx.prefill_scores(
                qh,
                k_used,
                scores,
                batch_heads,
                prefill_len,
                kv_stride,
                head_dim,
                scale,
            )?;
            ctx.causal_softmax(scores, batch_heads * prefill_len, prefill_len)?;
            ctx.prefill_out(
                scores,
                v_used,
                attn_h,
                batch_heads,
                prefill_len,
                kv_stride,
                head_dim,
            )?;
        }
        ctx.merge_heads_prefill(attn_h, attn, batch, prefill_len, config.n_heads, head_dim)?;

        ctx.linear_into(
            attn,
            weight(weights, &name("wo"))?,
            Some(weight(weights, &name("bo"))?),
            update,
        )?;
        ctx.residual_inplace(x, update, rows * d_model)?;

        ctx.layer_norm_batch(
            x,
            weight(weights, &name("ln2_g"))?,
            weight(weights, &name("ln2_b"))?,
            norm,
            rows,
            d_model,
            config.eps as f32,
        )?;
        ctx.linear_into(
            norm,
            weight(weights, &name("w1"))?,
            Some(weight(weights, &name("b1"))?),
            hidden,
        )?;
        ctx.gelu_inplace(hidden, rows * config.d_ff)?;
        ctx.linear_into(
            hidden,
            weight(weights, &name("w2"))?,
            Some(weight(weights, &name("b2"))?),
            update,
        )?;
        ctx.residual_inplace(x, update, rows * d_model)?;
    }
    let last_x = last_x
        .as_mut()
        .ok_or_else(|| Error::invalid("missing last-token workspace"))?;
    let last_norm = last_norm
        .as_mut()
        .ok_or_else(|| Error::invalid("missing last-token norm workspace"))?;
    ctx.gather_last_rows_var(x, prefill_lengths, last_x, batch, prefill_len, d_model)?;
    ctx.layer_norm_batch(
        last_x,
        weight(weights, "ln_f_g")?,
        weight(weights, "ln_f_b")?,
        last_norm,
        batch,
        d_model,
        config.eps as f32,
    )?;
    ctx.linear_into(last_norm, weight(weights, "lm_head")?, None, logits)?;
    ctx.copy_i32(prefill_lengths, positions, batch)?;
    Ok(())
}

fn enqueue_batched_decode(
    config: &DecoderConfig,
    batch: usize,
    kv_dtype: KvDType,
    ctx: &CudaStaticContext,
    buffers: &mut BatchedStaticDecoderBuffers,
) -> Result<()> {
    let d_model = config.d_model;
    let head_dim = config.head_dim();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let BatchedStaticDecoderBuffers {
        weights,
        kv,
        decode_tokens,
        positions,
        decode,
        decode_ws,
        ..
    } = buffers;

    ctx.embedding_batch(
        decode_tokens,
        weight(weights, "tok_emb")?,
        &mut decode.x,
        batch,
        d_model,
    )?;
    for (layer, (k_cache, v_cache)) in kv.iter_mut().enumerate() {
        let prefix = format!("layers.{layer}");
        let name = |suffix: &str| format!("{prefix}.{suffix}");
        ctx.layer_norm_batch(
            &decode.x,
            weight(weights, &name("ln1_g"))?,
            weight(weights, &name("ln1_b"))?,
            &mut decode.norm,
            batch,
            d_model,
            config.eps as f32,
        )?;
        ctx.linear_into(
            &decode.norm,
            weight(weights, &name("wq"))?,
            Some(weight(weights, &name("bq"))?),
            &mut decode.q,
        )?;
        ctx.linear_into(
            &decode.norm,
            weight(weights, &name("wk"))?,
            Some(weight(weights, &name("bk"))?),
            &mut decode.k,
        )?;
        ctx.linear_into(
            &decode.norm,
            weight(weights, &name("wv"))?,
            Some(weight(weights, &name("bv"))?),
            &mut decode.v,
        )?;
        ctx.kv_append_batch_typed(
            &decode.k,
            &decode.v,
            k_cache,
            v_cache,
            positions,
            batch,
            config.n_heads,
            head_dim,
            config.max_seq,
            kv_dtype.dtype(),
        )?;
        let lowp = kv_dtype != KvDType::F32;
        if lowp && head_dim <= 256 {
            ctx.flash_attention_decode_lowp(
                &decode.q,
                k_cache,
                v_cache,
                positions,
                &mut decode.attn,
                batch,
                config.n_heads,
                config.max_seq,
                head_dim,
                scale,
                kv_dtype.dtype(),
            )?;
        } else {
            let decode_ws = decode_ws
                .as_mut()
                .ok_or_else(|| Error::invalid("missing decode workspace"))?;
            let (k_used, v_used, kv_stride) = if lowp {
                let k_work = decode_ws
                    .k_work
                    .as_mut()
                    .ok_or_else(|| Error::invalid("missing decode K workspace"))?;
                let v_work = decode_ws
                    .v_work
                    .as_mut()
                    .ok_or_else(|| Error::invalid("missing decode V workspace"))?;
                ctx.kv_cache_to_f32_typed(
                    k_cache,
                    v_cache,
                    k_work,
                    v_work,
                    batch,
                    config.n_heads,
                    config.max_seq,
                    config.max_seq,
                    head_dim,
                    kv_dtype.dtype(),
                )?;
                (
                    k_work as &Tensor<CudaBackend>,
                    v_work as &Tensor<CudaBackend>,
                    config.max_seq * head_dim,
                )
            } else {
                (
                    k_cache as &Tensor<CudaBackend>,
                    v_cache as &Tensor<CudaBackend>,
                    config.max_seq * head_dim,
                )
            };
            ctx.split_heads_decode(
                &decode.q,
                &mut decode_ws.qh,
                batch,
                config.n_heads,
                head_dim,
            )?;
            ctx.decode_scores(
                &decode_ws.qh,
                k_used,
                &mut decode_ws.scores,
                batch * config.n_heads,
                config.max_seq,
                kv_stride,
                head_dim,
                scale,
            )?;
            ctx.decode_softmax(
                &mut decode_ws.scores,
                positions,
                batch,
                config.n_heads,
                config.max_seq,
            )?;
            ctx.decode_out(
                &decode_ws.scores,
                v_used,
                &mut decode_ws.attn_h,
                batch * config.n_heads,
                config.max_seq,
                kv_stride,
                head_dim,
            )?;
            ctx.merge_heads_decode(
                &decode_ws.attn_h,
                &mut decode.attn,
                batch,
                config.n_heads,
                head_dim,
            )?;
        }
        ctx.linear_into(
            &decode.attn,
            weight(weights, &name("wo"))?,
            Some(weight(weights, &name("bo"))?),
            &mut decode.update,
        )?;
        ctx.residual_inplace(&mut decode.x, &decode.update, batch * d_model)?;

        ctx.layer_norm_batch(
            &decode.x,
            weight(weights, &name("ln2_g"))?,
            weight(weights, &name("ln2_b"))?,
            &mut decode.norm,
            batch,
            d_model,
            config.eps as f32,
        )?;
        ctx.linear_into(
            &decode.norm,
            weight(weights, &name("w1"))?,
            Some(weight(weights, &name("b1"))?),
            &mut decode.hidden,
        )?;
        ctx.gelu_inplace(&mut decode.hidden, batch * config.d_ff)?;
        ctx.linear_into(
            &decode.hidden,
            weight(weights, &name("w2"))?,
            Some(weight(weights, &name("b2"))?),
            &mut decode.update,
        )?;
        ctx.residual_inplace(&mut decode.x, &decode.update, batch * d_model)?;
    }
    ctx.layer_norm_batch(
        &decode.x,
        weight(weights, "ln_f_g")?,
        weight(weights, "ln_f_b")?,
        &mut decode.norm,
        batch,
        d_model,
        config.eps as f32,
    )?;
    ctx.linear_into(
        &decode.norm,
        weight(weights, "lm_head")?,
        None,
        &mut decode.logits,
    )?;
    ctx.bump_positions(positions, batch)?;
    Ok(())
}
