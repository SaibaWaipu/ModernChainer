use std::sync::Arc;

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{Gemm, GemmConfig};
use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use modern_chainer_core::{Backend, DType, Error, Ops, Result, Shape, Tensor};
use modern_chainer_cpu::CpuBackend;

use crate::backend::{CudaBackend, CudaRuntime, CudaStorage};

#[derive(Clone)]
pub struct CudaOps {
    pub(crate) runtime: Arc<CudaRuntime>,
}

fn cuda_error(context: &str, error: impl std::fmt::Display) -> Error {
    Error::Backend(format!("cuda {context}: {error}"))
}

impl CudaOps {
    fn backend(&self) -> CudaBackend {
        CudaBackend {
            runtime: Arc::clone(&self.runtime),
            ops: self.clone(),
        }
    }

    fn f32<'a>(&self, tensor: &'a Tensor<CudaBackend>) -> Result<&'a CudaSlice<f32>> {
        if tensor.dtype() != DType::F32 {
            return Err(Error::UnsupportedDType(tensor.dtype()));
        }
        if !tensor.is_contiguous() {
            return Err(Error::invalid("cuda kernel requires a contiguous tensor"));
        }
        tensor.storage().f32()
    }

    fn out_f32(&self, shape: &Shape) -> Result<(CudaBackend, CudaStorage)> {
        let backend = self.backend();
        let storage = backend.alloc_zeroed_for(shape.numel() * 4, DType::F32)?;
        Ok((backend, storage))
    }

    fn tensor_from_storage(
        &self,
        backend: CudaBackend,
        storage: CudaStorage,
        shape: &Shape,
    ) -> Result<Tensor<CudaBackend>> {
        Tensor::from_storage(
            backend,
            storage,
            DType::F32,
            shape.clone(),
            shape.contiguous_strides(),
            0,
        )
    }

    fn binary(
        &self,
        function: &CudaFunction,
        a: &Tensor<CudaBackend>,
        b: &Tensor<CudaBackend>,
    ) -> Result<Tensor<CudaBackend>> {
        if a.shape() != b.shape() {
            return Err(Error::shape_mismatch(
                "cuda binary elementwise",
                a.shape().clone(),
                b.shape().clone(),
            ));
        }
        let a_slice = self.f32(a)?;
        let b_slice = self.f32(b)?;
        let n = a.numel();
        let (backend, mut storage) = self.out_f32(a.shape())?;
        {
            let out = storage.f32_mut()?;
            let n_i = i32::try_from(n).map_err(|_| Error::invalid("tensor too large"))?;
            let mut builder = self.runtime.stream.launch_builder(function);
            builder.arg(a_slice).arg(b_slice).arg(out).arg(&n_i);
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(n as u32))
                    .map_err(|error| cuda_error("launch", error))?;
            }
        }
        self.tensor_from_storage(backend, storage, a.shape())
    }

    fn unary(
        &self,
        function: &CudaFunction,
        a: &Tensor<CudaBackend>,
    ) -> Result<Tensor<CudaBackend>> {
        let a_slice = self.f32(a)?;
        let n = a.numel();
        let (backend, mut storage) = self.out_f32(a.shape())?;
        {
            let out = storage.f32_mut()?;
            let n_i = i32::try_from(n).map_err(|_| Error::invalid("tensor too large"))?;
            let mut builder = self.runtime.stream.launch_builder(function);
            builder.arg(a_slice).arg(out).arg(&n_i);
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(n as u32))
                    .map_err(|error| cuda_error("launch", error))?;
            }
        }
        self.tensor_from_storage(backend, storage, a.shape())
    }

    fn fill(&self, reference: &Tensor<CudaBackend>, value: f32) -> Result<Tensor<CudaBackend>> {
        let n = reference.numel();
        let (backend, mut storage) = self.out_f32(reference.shape())?;
        {
            let out = storage.f32_mut()?;
            let n_i = i32::try_from(n).map_err(|_| Error::invalid("tensor too large"))?;
            let mut builder = self
                .runtime
                .stream
                .launch_builder(&self.runtime.kernels.fill);
            builder.arg(out).arg(&value).arg(&n_i);
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(n as u32))
                    .map_err(|error| cuda_error("launch", error))?;
            }
        }
        self.tensor_from_storage(backend, storage, reference.shape())
    }

    fn to_cpu(&self, tensor: &Tensor<CudaBackend>) -> Result<Tensor<CpuBackend>> {
        let bytes = tensor.to_bytes()?;
        Tensor::from_bytes(
            CpuBackend::new(),
            &bytes,
            tensor.shape().clone(),
            tensor.dtype(),
        )
    }

    fn upload(&self, tensor: &Tensor<CpuBackend>) -> Result<Tensor<CudaBackend>> {
        let bytes = tensor.to_bytes()?;
        Tensor::from_bytes(
            self.backend(),
            &bytes,
            tensor.shape().clone(),
            tensor.dtype(),
        )
    }

    fn gemm(
        &self,
        config: GemmConfig<f32>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
    ) -> Result<()> {
        unsafe { self.runtime.blas.gemm(config, a, b, c) }
            .map_err(|error| cuda_error("cublas gemm", error))
    }
}

impl Ops<CudaBackend> for CudaOps {
    fn add(&self, a: &Tensor<CudaBackend>, b: &Tensor<CudaBackend>) -> Result<Tensor<CudaBackend>> {
        if a.shape() == b.shape() {
            self.binary(&self.runtime.kernels.add, a, b)
        } else {
            let result = CpuBackend::new()
                .ops()
                .add(&self.to_cpu(a)?, &self.to_cpu(b)?)?;
            self.upload(&result)
        }
    }

    fn sub(&self, a: &Tensor<CudaBackend>, b: &Tensor<CudaBackend>) -> Result<Tensor<CudaBackend>> {
        if a.shape() == b.shape() {
            self.binary(&self.runtime.kernels.sub, a, b)
        } else {
            let result = CpuBackend::new()
                .ops()
                .sub(&self.to_cpu(a)?, &self.to_cpu(b)?)?;
            self.upload(&result)
        }
    }

    fn mul(&self, a: &Tensor<CudaBackend>, b: &Tensor<CudaBackend>) -> Result<Tensor<CudaBackend>> {
        if a.shape() == b.shape() {
            self.binary(&self.runtime.kernels.mul, a, b)
        } else {
            let result = CpuBackend::new()
                .ops()
                .mul(&self.to_cpu(a)?, &self.to_cpu(b)?)?;
            self.upload(&result)
        }
    }

    fn div(&self, a: &Tensor<CudaBackend>, b: &Tensor<CudaBackend>) -> Result<Tensor<CudaBackend>> {
        if a.shape() == b.shape() {
            self.binary(&self.runtime.kernels.div, a, b)
        } else {
            let result = CpuBackend::new()
                .ops()
                .div(&self.to_cpu(a)?, &self.to_cpu(b)?)?;
            self.upload(&result)
        }
    }

    fn sqrt(&self, a: &Tensor<CudaBackend>) -> Result<Tensor<CudaBackend>> {
        self.unary(&self.runtime.kernels.sqrt, a)
    }

    fn relu(&self, a: &Tensor<CudaBackend>) -> Result<Tensor<CudaBackend>> {
        self.unary(&self.runtime.kernels.relu, a)
    }

    fn relu_backward(
        &self,
        x: &Tensor<CudaBackend>,
        grad: &Tensor<CudaBackend>,
    ) -> Result<Tensor<CudaBackend>> {
        self.binary(&self.runtime.kernels.relu_backward, x, grad)
    }

    fn silu(&self, a: &Tensor<CudaBackend>) -> Result<Tensor<CudaBackend>> {
        self.unary(&self.runtime.kernels.silu, a)
    }

    fn matmul(
        &self,
        a: &Tensor<CudaBackend>,
        b: &Tensor<CudaBackend>,
    ) -> Result<Tensor<CudaBackend>> {
        if a.rank() != 2 || b.rank() != 2 {
            return Err(Error::invalid("cuda matmul expects 2-D tensors"));
        }
        let (m, k) = (a.shape().dims()[0], a.shape().dims()[1]);
        let (k2, n) = (b.shape().dims()[0], b.shape().dims()[1]);
        if k != k2 {
            return Err(Error::shape_mismatch(
                "cuda matmul inner dimension",
                a.shape().clone(),
                b.shape().clone(),
            ));
        }
        let a_slice = self.f32(a)?;
        let b_slice = self.f32(b)?;
        let (backend, mut storage) = self.out_f32(&Shape::new([m, n]))?;
        {
            let out = storage.f32_mut()?;
            let config = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: n as i32,
                n: m as i32,
                k: k as i32,
                alpha: 1.0,
                lda: n as i32,
                ldb: k as i32,
                beta: 0.0,
                ldc: n as i32,
            };
            self.gemm(config, b_slice, a_slice, out)?;
        }
        self.tensor_from_storage(backend, storage, &Shape::new([m, n]))
    }

    fn linear(
        &self,
        x: &Tensor<CudaBackend>,
        weight: &Tensor<CudaBackend>,
        bias: Option<&Tensor<CudaBackend>>,
    ) -> Result<Tensor<CudaBackend>> {
        if x.rank() != 2 || weight.rank() != 2 {
            return Err(Error::invalid("cuda linear expects 2-D input and weight"));
        }
        let (batch, in_features) = (x.shape().dims()[0], x.shape().dims()[1]);
        let (out_features, k2) = (weight.shape().dims()[0], weight.shape().dims()[1]);
        if in_features != k2 {
            return Err(Error::shape_mismatch(
                "cuda linear inner dimension",
                x.shape().clone(),
                weight.shape().clone(),
            ));
        }
        let x_slice = self.f32(x)?;
        let weight_slice = self.f32(weight)?;
        let (backend, mut storage) = self.out_f32(&Shape::new([batch, out_features]))?;
        {
            let out = storage.f32_mut()?;
            let config = GemmConfig {
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
            };
            self.gemm(config, weight_slice, x_slice, out)?;
        }
        if let Some(bias) = bias {
            let bias_slice = self.f32(bias)?;
            let out = storage.f32_mut()?;
            let rows = i32::try_from(batch).map_err(|_| Error::invalid("tensor too large"))?;
            let cols =
                i32::try_from(out_features).map_err(|_| Error::invalid("tensor too large"))?;
            let total = u32::try_from(batch * out_features)
                .map_err(|_| Error::invalid("tensor too large"))?;
            let mut builder = self
                .runtime
                .stream
                .launch_builder(&self.runtime.kernels.bias_add);
            builder.arg(out).arg(bias_slice).arg(&rows).arg(&cols);
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(total))
                    .map_err(|error| cuda_error("launch", error))?;
            }
        }
        self.tensor_from_storage(backend, storage, &Shape::new([batch, out_features]))
    }

    fn linear_backward(
        &self,
        grad: &Tensor<CudaBackend>,
        x: &Tensor<CudaBackend>,
        weight: &Tensor<CudaBackend>,
    ) -> Result<(Tensor<CudaBackend>, Tensor<CudaBackend>)> {
        let grad_x = self.matmul(grad, weight)?;
        let grad_t = self.transpose2d(grad)?;
        let grad_w = self.matmul(&grad_t, x)?;
        Ok((grad_x, grad_w))
    }

    fn transpose2d(&self, a: &Tensor<CudaBackend>) -> Result<Tensor<CudaBackend>> {
        if a.rank() != 2 {
            return Err(Error::invalid("cuda transpose2d expects a 2-D tensor"));
        }
        let rows = a.shape().dims()[0];
        let cols = a.shape().dims()[1];
        let a_slice = self.f32(a)?;
        let (backend, mut storage) = self.out_f32(&Shape::new([cols, rows]))?;
        {
            let out = storage.f32_mut()?;
            let rows_i = i32::try_from(rows).map_err(|_| Error::invalid("tensor too large"))?;
            let cols_i = i32::try_from(cols).map_err(|_| Error::invalid("tensor too large"))?;
            let n = u32::try_from(rows * cols).map_err(|_| Error::invalid("tensor too large"))?;
            let mut builder = self
                .runtime
                .stream
                .launch_builder(&self.runtime.kernels.transpose2d);
            builder.arg(a_slice).arg(out).arg(&rows_i).arg(&cols_i);
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(n))
                    .map_err(|error| cuda_error("launch", error))?;
            }
        }
        self.tensor_from_storage(backend, storage, &Shape::new([cols, rows]))
    }

    fn sum_all(&self, a: &Tensor<CudaBackend>) -> Result<Tensor<CudaBackend>> {
        let result = CpuBackend::new().ops().sum_all(&self.to_cpu(a)?)?;
        self.upload(&result)
    }

    fn sum_to(&self, a: &Tensor<CudaBackend>, shape: &Shape) -> Result<Tensor<CudaBackend>> {
        let result = CpuBackend::new().ops().sum_to(&self.to_cpu(a)?, shape)?;
        self.upload(&result)
    }

    fn scale(&self, a: &Tensor<CudaBackend>, factor: f64) -> Result<Tensor<CudaBackend>> {
        let factor = factor as f32;
        let n = a.numel();
        let a_slice = self.f32(a)?;
        let (backend, mut storage) = self.out_f32(a.shape())?;
        {
            let out = storage.f32_mut()?;
            let n_i = i32::try_from(n).map_err(|_| Error::invalid("tensor too large"))?;
            let mut builder = self
                .runtime
                .stream
                .launch_builder(&self.runtime.kernels.scale);
            builder.arg(a_slice).arg(out).arg(&factor).arg(&n_i);
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(n as u32))
                    .map_err(|error| cuda_error("launch", error))?;
            }
        }
        self.tensor_from_storage(backend, storage, a.shape())
    }

    fn ones_like(&self, a: &Tensor<CudaBackend>) -> Result<Tensor<CudaBackend>> {
        self.fill(a, 1.0)
    }

    fn zeros_like(&self, a: &Tensor<CudaBackend>) -> Result<Tensor<CudaBackend>> {
        self.fill(a, 0.0)
    }

    fn full_like(&self, a: &Tensor<CudaBackend>, value: f64) -> Result<Tensor<CudaBackend>> {
        self.fill(a, value as f32)
    }

    fn to_scalar(&self, a: &Tensor<CudaBackend>) -> Result<f64> {
        if !a.is_scalar() {
            return Err(Error::invalid(format!(
                "expected scalar tensor, got {}",
                a.shape()
            )));
        }
        let slice = self.f32(a)?;
        let host = self
            .runtime
            .stream
            .clone_dtoh(slice)
            .map_err(|error| cuda_error("dtoh", error))?;
        host.first()
            .copied()
            .map(|value| value as f64)
            .ok_or_else(|| Error::invalid("empty scalar tensor"))
    }

    fn adam_update(
        &self,
        param: &Tensor<CudaBackend>,
        grad: &Tensor<CudaBackend>,
        m: &Tensor<CudaBackend>,
        v: &Tensor<CudaBackend>,
        cfg: &modern_chainer_core::AdamConfig,
    ) -> Result<(
        Tensor<CudaBackend>,
        Tensor<CudaBackend>,
        Tensor<CudaBackend>,
    )> {
        let n = param.numel();
        if grad.numel() != n || m.numel() != n || v.numel() != n {
            return Err(Error::invalid("cuda adam_update shape mismatch"));
        }
        let p_slice = self.f32(param)?;
        let g_slice = self.f32(grad)?;
        let m_slice = self.f32(m)?;
        let v_slice = self.f32(v)?;
        let (backend, mut p_storage) = self.out_f32(param.shape())?;
        let (_, mut m_storage) = self.out_f32(m.shape())?;
        let (_, mut v_storage) = self.out_f32(v.shape())?;
        {
            let new_p = p_storage.f32_mut()?;
            let new_m = m_storage.f32_mut()?;
            let new_v = v_storage.f32_mut()?;
            let n_i = i32::try_from(n).map_err(|_| Error::invalid("tensor too large"))?;
            let lr = cfg.lr as f32;
            let b1 = cfg.beta1 as f32;
            let b2 = cfg.beta2 as f32;
            let eps = cfg.eps as f32;
            let bc1 = 1.0 - b1.powi(cfg.step as i32);
            let bc2 = 1.0 - b2.powi(cfg.step as i32);
            let mut builder = self
                .runtime
                .stream
                .launch_builder(&self.runtime.kernels.adam_update);
            builder
                .arg(p_slice)
                .arg(g_slice)
                .arg(m_slice)
                .arg(v_slice)
                .arg(new_p)
                .arg(new_m)
                .arg(new_v)
                .arg(&n_i)
                .arg(&lr)
                .arg(&b1)
                .arg(&b2)
                .arg(&eps)
                .arg(&bc1)
                .arg(&bc2);
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(n as u32))
                    .map_err(|error| cuda_error("launch", error))?;
            }
        }
        Ok((
            self.tensor_from_storage(backend.clone(), p_storage, param.shape())?,
            self.tensor_from_storage(backend.clone(), m_storage, m.shape())?,
            self.tensor_from_storage(backend, v_storage, v.shape())?,
        ))
    }

    fn softmax_cross_entropy(
        &self,
        logits: &Tensor<CudaBackend>,
        target: &Tensor<CudaBackend>,
    ) -> Result<Tensor<CudaBackend>> {
        let result = CpuBackend::new()
            .ops()
            .softmax_cross_entropy(&self.to_cpu(logits)?, &self.to_cpu(target)?)?;
        self.upload(&result)
    }

    fn softmax_cross_entropy_backward(
        &self,
        logits: &Tensor<CudaBackend>,
        target: &Tensor<CudaBackend>,
    ) -> Result<Tensor<CudaBackend>> {
        let result = CpuBackend::new()
            .ops()
            .softmax_cross_entropy_backward(&self.to_cpu(logits)?, &self.to_cpu(target)?)?;
        self.upload(&result)
    }

    fn embedding(
        &self,
        ids: &Tensor<CudaBackend>,
        weight: &Tensor<CudaBackend>,
    ) -> Result<Tensor<CudaBackend>> {
        let result = CpuBackend::new()
            .ops()
            .embedding(&self.to_cpu(ids)?, &self.to_cpu(weight)?)?;
        self.upload(&result)
    }

    fn layer_norm(
        &self,
        x: &Tensor<CudaBackend>,
        gamma: &Tensor<CudaBackend>,
        beta: &Tensor<CudaBackend>,
        eps: f64,
    ) -> Result<Tensor<CudaBackend>> {
        let result = CpuBackend::new().ops().layer_norm(
            &self.to_cpu(x)?,
            &self.to_cpu(gamma)?,
            &self.to_cpu(beta)?,
            eps,
        )?;
        self.upload(&result)
    }

    fn softmax_last_dim(&self, x: &Tensor<CudaBackend>) -> Result<Tensor<CudaBackend>> {
        let result = CpuBackend::new().ops().softmax_last_dim(&self.to_cpu(x)?)?;
        self.upload(&result)
    }

    fn split_heads(
        &self,
        x: &Tensor<CudaBackend>,
        heads: usize,
        head_dim: usize,
    ) -> Result<Tensor<CudaBackend>> {
        let result = CpuBackend::new()
            .ops()
            .split_heads(&self.to_cpu(x)?, heads, head_dim)?;
        self.upload(&result)
    }

    fn merge_heads(&self, x: &Tensor<CudaBackend>, heads: usize) -> Result<Tensor<CudaBackend>> {
        let result = CpuBackend::new()
            .ops()
            .merge_heads(&self.to_cpu(x)?, heads)?;
        self.upload(&result)
    }

    fn attention(
        &self,
        q: &Tensor<CudaBackend>,
        k: &Tensor<CudaBackend>,
        v: &Tensor<CudaBackend>,
        scale: f64,
        causal: bool,
    ) -> Result<Tensor<CudaBackend>> {
        let result = CpuBackend::new().ops().attention(
            &self.to_cpu(q)?,
            &self.to_cpu(k)?,
            &self.to_cpu(v)?,
            scale,
            causal,
        )?;
        self.upload(&result)
    }

    fn concat_seq(
        &self,
        a: &Tensor<CudaBackend>,
        b: &Tensor<CudaBackend>,
    ) -> Result<Tensor<CudaBackend>> {
        let result = CpuBackend::new()
            .ops()
            .concat_seq(&self.to_cpu(a)?, &self.to_cpu(b)?)?;
        self.upload(&result)
    }

    fn last_row(&self, x: &Tensor<CudaBackend>) -> Result<Tensor<CudaBackend>> {
        let result = CpuBackend::new().ops().last_row(&self.to_cpu(x)?)?;
        self.upload(&result)
    }

    #[allow(clippy::too_many_arguments)]
    fn paged_attention(
        &self,
        q: &Tensor<CudaBackend>,
        k_blocks: &Tensor<CudaBackend>,
        v_blocks: &Tensor<CudaBackend>,
        block_table: &Tensor<CudaBackend>,
        seq_len: usize,
        scale: f64,
        causal: bool,
    ) -> Result<Tensor<CudaBackend>> {
        let result = CpuBackend::new().ops().paged_attention(
            &self.to_cpu(q)?,
            &self.to_cpu(k_blocks)?,
            &self.to_cpu(v_blocks)?,
            &self.to_cpu(block_table)?,
            seq_len,
            scale,
            causal,
        )?;
        self.upload(&result)
    }

    #[allow(clippy::too_many_arguments)]
    fn paged_append(
        &self,
        k_blocks: &Tensor<CudaBackend>,
        v_blocks: &Tensor<CudaBackend>,
        block_table: &Tensor<CudaBackend>,
        len: usize,
        k_new: &Tensor<CudaBackend>,
        v_new: &Tensor<CudaBackend>,
    ) -> Result<(Tensor<CudaBackend>, Tensor<CudaBackend>)> {
        let (k, v) = CpuBackend::new().ops().paged_append(
            &self.to_cpu(k_blocks)?,
            &self.to_cpu(v_blocks)?,
            &self.to_cpu(block_table)?,
            len,
            &self.to_cpu(k_new)?,
            &self.to_cpu(v_new)?,
        )?;
        Ok((self.upload(&k)?, self.upload(&v)?))
    }

    #[allow(clippy::too_many_arguments)]
    fn paged_attention_batch(
        &self,
        q: &Tensor<CudaBackend>,
        k_blocks: &Tensor<CudaBackend>,
        v_blocks: &Tensor<CudaBackend>,
        block_tables: &Tensor<CudaBackend>,
        seq_lens: &Tensor<CudaBackend>,
        scale: f64,
        causal: bool,
    ) -> Result<Tensor<CudaBackend>> {
        let result = CpuBackend::new().ops().paged_attention_batch(
            &self.to_cpu(q)?,
            &self.to_cpu(k_blocks)?,
            &self.to_cpu(v_blocks)?,
            &self.to_cpu(block_tables)?,
            &self.to_cpu(seq_lens)?,
            scale,
            causal,
        )?;
        self.upload(&result)
    }
}
