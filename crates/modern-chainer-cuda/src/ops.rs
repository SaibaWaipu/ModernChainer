use std::sync::Arc;

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{Gemm, GemmConfig};
use cudarc::cublaslt::{Matmul as BlasLtMatmul, MatmulConfig as BlasLtConfig};
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

    fn i64<'a>(&self, tensor: &'a Tensor<CudaBackend>) -> Result<&'a CudaSlice<i64>> {
        if tensor.dtype() != DType::I64 {
            return Err(Error::UnsupportedDType(tensor.dtype()));
        }
        if !tensor.is_contiguous() {
            return Err(Error::invalid("cuda kernel requires a contiguous tensor"));
        }
        tensor.storage().i64()
    }

    fn row_config(rows: usize) -> Result<LaunchConfig> {
        if rows == 0 {
            return Err(Error::invalid("empty tensor for row kernel"));
        }
        let grid = u32::try_from(rows).map_err(|_| Error::invalid("tensor too large"))?;
        Ok(LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
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
            match bias {
                Some(bias) => {
                    let bias_slice = self.f32(bias)?;
                    let config = BlasLtConfig {
                        transa: true,
                        transb: false,
                        transc: false,
                        m: out_features as u64,
                        n: batch as u64,
                        k: in_features as u64,
                        alpha: 1.0,
                        lda: in_features as i64,
                        ldb: in_features as i64,
                        beta: 0.0,
                        ldc: out_features as i64,
                        stride_a: None,
                        stride_b: None,
                        stride_c: None,
                        stride_bias: None,
                        batch_size: None,
                    };
                    unsafe {
                        self.runtime.blaslt.matmul(
                            config,
                            weight_slice,
                            x_slice,
                            out,
                            Some(bias_slice),
                            None,
                        )
                    }
                    .map_err(|error| cuda_error("cublaslt linear", error))?;
                }
                None => {
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
        let a_slice = self.f32(a)?;
        let n = a.numel();
        let (backend, mut storage) = self.out_f32(&Shape::scalar())?;
        if n > 0 {
            let out = storage.f32_mut()?;
            let n_i = i32::try_from(n).map_err(|_| Error::invalid("tensor too large"))?;
            let mut builder = self
                .runtime
                .stream
                .launch_builder(&self.runtime.kernels.sum_all);
            builder.arg(a_slice).arg(out).arg(&n_i);
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(n as u32))
                    .map_err(|error| cuda_error("launch", error))?;
            }
        }
        self.tensor_from_storage(backend, storage, &Shape::scalar())
    }

    fn sum_to(&self, a: &Tensor<CudaBackend>, shape: &Shape) -> Result<Tensor<CudaBackend>> {
        if a.rank() == 2 && shape.rank() == 1 && shape.dims()[0] == a.shape().dims()[1] {
            let rows = a.shape().dims()[0];
            let cols = a.shape().dims()[1];
            let a_slice = self.f32(a)?;
            let (backend, mut storage) = self.out_f32(shape)?;
            if cols > 0 {
                let out = storage.f32_mut()?;
                let rows_i = i32::try_from(rows).map_err(|_| Error::invalid("tensor too large"))?;
                let cols_i = i32::try_from(cols).map_err(|_| Error::invalid("tensor too large"))?;
                let mut builder = self
                    .runtime
                    .stream
                    .launch_builder(&self.runtime.kernels.sum_to_bias);
                builder.arg(a_slice).arg(out).arg(&rows_i).arg(&cols_i);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(cols as u32))
                        .map_err(|error| cuda_error("launch", error))?;
                }
            }
            return self.tensor_from_storage(backend, storage, shape);
        }
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
        if logits.rank() != 2 {
            return Err(Error::invalid(
                "cuda softmax_cross_entropy expects [rows, cols]",
            ));
        }
        let rows = logits.shape().dims()[0];
        let cols = logits.shape().dims()[1];
        if target.numel() != rows {
            return Err(Error::invalid("softmax_cross_entropy target size mismatch"));
        }
        let logits_slice = self.f32(logits)?;
        let target_slice = self.i64(target)?;
        let (backend, mut storage) = self.out_f32(&Shape::scalar())?;
        if rows > 0 {
            let out = storage.f32_mut()?;
            let rows_i = i32::try_from(rows).map_err(|_| Error::invalid("tensor too large"))?;
            let cols_i = i32::try_from(cols).map_err(|_| Error::invalid("tensor too large"))?;
            let mut builder = self
                .runtime
                .stream
                .launch_builder(&self.runtime.kernels.softmax_ce);
            builder
                .arg(logits_slice)
                .arg(target_slice)
                .arg(out)
                .arg(&rows_i)
                .arg(&cols_i);
            unsafe {
                builder
                    .launch(Self::row_config(rows)?)
                    .map_err(|error| cuda_error("launch", error))?;
            }
        }
        self.tensor_from_storage(backend, storage, &Shape::scalar())
    }

    fn softmax_cross_entropy_backward(
        &self,
        logits: &Tensor<CudaBackend>,
        target: &Tensor<CudaBackend>,
    ) -> Result<Tensor<CudaBackend>> {
        if logits.rank() != 2 {
            return Err(Error::invalid(
                "cuda softmax_cross_entropy expects [rows, cols]",
            ));
        }
        let rows = logits.shape().dims()[0];
        let cols = logits.shape().dims()[1];
        if target.numel() != rows {
            return Err(Error::invalid("softmax_cross_entropy target size mismatch"));
        }
        let logits_slice = self.f32(logits)?;
        let target_slice = self.i64(target)?;
        let (backend, mut storage) = self.out_f32(logits.shape())?;
        if rows > 0 {
            let out = storage.f32_mut()?;
            let rows_i = i32::try_from(rows).map_err(|_| Error::invalid("tensor too large"))?;
            let cols_i = i32::try_from(cols).map_err(|_| Error::invalid("tensor too large"))?;
            let mut builder = self
                .runtime
                .stream
                .launch_builder(&self.runtime.kernels.softmax_ce_backward);
            builder
                .arg(logits_slice)
                .arg(target_slice)
                .arg(out)
                .arg(&rows_i)
                .arg(&cols_i);
            unsafe {
                builder
                    .launch(Self::row_config(rows)?)
                    .map_err(|error| cuda_error("launch", error))?;
            }
        }
        self.tensor_from_storage(backend, storage, logits.shape())
    }

    fn embedding(
        &self,
        ids: &Tensor<CudaBackend>,
        weight: &Tensor<CudaBackend>,
    ) -> Result<Tensor<CudaBackend>> {
        if weight.rank() != 2 || ids.rank() == 0 || ids.rank() > 2 {
            return Err(Error::invalid(
                "cuda embedding expects [.., id] and [vocab, dim]",
            ));
        }
        let rows = ids.numel();
        let dim = weight.shape().dims()[1];
        let ids_slice = self.i64(ids)?;
        let weight_slice = self.f32(weight)?;
        let mut dims = ids.shape().dims().to_vec();
        dims.push(dim);
        let shape = Shape::new(dims);
        let (backend, mut storage) = self.out_f32(&shape)?;
        let total = rows * dim;
        if total > 0 {
            let out = storage.f32_mut()?;
            let rows_i = i32::try_from(rows).map_err(|_| Error::invalid("tensor too large"))?;
            let dim_i = i32::try_from(dim).map_err(|_| Error::invalid("tensor too large"))?;
            let mut builder = self
                .runtime
                .stream
                .launch_builder(&self.runtime.kernels.embedding);
            builder
                .arg(ids_slice)
                .arg(weight_slice)
                .arg(out)
                .arg(&rows_i)
                .arg(&dim_i);
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(total as u32))
                    .map_err(|error| cuda_error("launch", error))?;
            }
        }
        self.tensor_from_storage(backend, storage, &shape)
    }

    fn layer_norm(
        &self,
        x: &Tensor<CudaBackend>,
        gamma: &Tensor<CudaBackend>,
        beta: &Tensor<CudaBackend>,
        eps: f64,
    ) -> Result<Tensor<CudaBackend>> {
        if x.rank() < 2 {
            return Err(Error::invalid("cuda layer_norm expects rank >= 2"));
        }
        let dim = *x.shape().dims().last().unwrap();
        if gamma.numel() != dim || beta.numel() != dim {
            return Err(Error::invalid("layer_norm gamma/beta size mismatch"));
        }
        let rows = x.numel() / dim;
        let x_slice = self.f32(x)?;
        let gamma_slice = self.f32(gamma)?;
        let beta_slice = self.f32(beta)?;
        let (backend, mut storage) = self.out_f32(x.shape())?;
        if rows > 0 && dim > 0 {
            let out = storage.f32_mut()?;
            let rows_i = i32::try_from(rows).map_err(|_| Error::invalid("tensor too large"))?;
            let dim_i = i32::try_from(dim).map_err(|_| Error::invalid("tensor too large"))?;
            let eps = eps as f32;
            let mut builder = self
                .runtime
                .stream
                .launch_builder(&self.runtime.kernels.layer_norm);
            builder
                .arg(x_slice)
                .arg(gamma_slice)
                .arg(beta_slice)
                .arg(out)
                .arg(&rows_i)
                .arg(&dim_i)
                .arg(&eps);
            unsafe {
                builder
                    .launch(Self::row_config(rows)?)
                    .map_err(|error| cuda_error("launch", error))?;
            }
        }
        self.tensor_from_storage(backend, storage, x.shape())
    }

    fn softmax_last_dim(&self, x: &Tensor<CudaBackend>) -> Result<Tensor<CudaBackend>> {
        if x.rank() != 2 {
            return Err(Error::invalid("cuda softmax_last_dim expects a 2-D tensor"));
        }
        let rows = x.shape().dims()[0];
        let cols = x.shape().dims()[1];
        let x_slice = self.f32(x)?;
        let (backend, mut storage) = self.out_f32(x.shape())?;
        if rows > 0 {
            let out = storage.f32_mut()?;
            let rows_i = i32::try_from(rows).map_err(|_| Error::invalid("tensor too large"))?;
            let cols_i = i32::try_from(cols).map_err(|_| Error::invalid("tensor too large"))?;
            let mut builder = self
                .runtime
                .stream
                .launch_builder(&self.runtime.kernels.softmax_last_dim);
            builder.arg(x_slice).arg(out).arg(&rows_i).arg(&cols_i);
            unsafe {
                builder
                    .launch(Self::row_config(rows)?)
                    .map_err(|error| cuda_error("launch", error))?;
            }
        }
        self.tensor_from_storage(backend, storage, x.shape())
    }

    fn split_heads(
        &self,
        x: &Tensor<CudaBackend>,
        heads: usize,
        head_dim: usize,
    ) -> Result<Tensor<CudaBackend>> {
        if x.rank() != 2 || x.shape().dims()[1] != heads * head_dim {
            return Err(Error::invalid("cuda split_heads shape mismatch"));
        }
        let seq = x.shape().dims()[0];
        let x_slice = self.f32(x)?;
        let shape = Shape::new([heads, seq, head_dim]);
        let (backend, mut storage) = self.out_f32(&shape)?;
        let total = seq * heads * head_dim;
        if total > 0 {
            let out = storage.f32_mut()?;
            let seq_i = i32::try_from(seq).map_err(|_| Error::invalid("tensor too large"))?;
            let heads_i = i32::try_from(heads).map_err(|_| Error::invalid("tensor too large"))?;
            let dim_i = i32::try_from(head_dim).map_err(|_| Error::invalid("tensor too large"))?;
            let mut builder = self
                .runtime
                .stream
                .launch_builder(&self.runtime.kernels.split_heads);
            builder
                .arg(x_slice)
                .arg(out)
                .arg(&seq_i)
                .arg(&heads_i)
                .arg(&dim_i);
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(total as u32))
                    .map_err(|error| cuda_error("launch", error))?;
            }
        }
        self.tensor_from_storage(backend, storage, &shape)
    }

    fn merge_heads(&self, x: &Tensor<CudaBackend>, heads: usize) -> Result<Tensor<CudaBackend>> {
        if x.rank() != 3 || x.shape().dims()[0] != heads {
            return Err(Error::invalid("cuda merge_heads shape mismatch"));
        }
        let seq = x.shape().dims()[1];
        let head_dim = x.shape().dims()[2];
        let x_slice = self.f32(x)?;
        let shape = Shape::new([seq, heads * head_dim]);
        let (backend, mut storage) = self.out_f32(&shape)?;
        let total = heads * seq * head_dim;
        if total > 0 {
            let out = storage.f32_mut()?;
            let heads_i = i32::try_from(heads).map_err(|_| Error::invalid("tensor too large"))?;
            let seq_i = i32::try_from(seq).map_err(|_| Error::invalid("tensor too large"))?;
            let dim_i = i32::try_from(head_dim).map_err(|_| Error::invalid("tensor too large"))?;
            let mut builder = self
                .runtime
                .stream
                .launch_builder(&self.runtime.kernels.merge_heads);
            builder
                .arg(x_slice)
                .arg(out)
                .arg(&heads_i)
                .arg(&seq_i)
                .arg(&dim_i);
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(total as u32))
                    .map_err(|error| cuda_error("launch", error))?;
            }
        }
        self.tensor_from_storage(backend, storage, &shape)
    }

    fn attention(
        &self,
        q: &Tensor<CudaBackend>,
        k: &Tensor<CudaBackend>,
        v: &Tensor<CudaBackend>,
        scale: f64,
        causal: bool,
    ) -> Result<Tensor<CudaBackend>> {
        if q.rank() != 3 || k.rank() != 3 || v.rank() != 3 {
            return Err(Error::invalid(
                "cuda attention expects [heads, seq, head_dim]",
            ));
        }
        let (heads, q_seq, head_dim) = (
            q.shape().dims()[0],
            q.shape().dims()[1],
            q.shape().dims()[2],
        );
        let kv_seq = k.shape().dims()[1];
        if k.shape().dims()[0] != heads
            || v.shape().dims()[0] != heads
            || k.shape().dims()[2] != head_dim
            || v.shape().dims()[2] != head_dim
            || v.shape().dims()[1] != kv_seq
        {
            return Err(Error::invalid("cuda attention dimension mismatch"));
        }
        let q_slice = self.f32(q)?;
        let k_slice = self.f32(k)?;
        let v_slice = self.f32(v)?;
        let (backend, mut storage) = self.out_f32(q.shape())?;
        let total = heads * q_seq;
        if total > 0 {
            let out = storage.f32_mut()?;
            let heads_i = i32::try_from(heads).map_err(|_| Error::invalid("tensor too large"))?;
            let q_seq_i = i32::try_from(q_seq).map_err(|_| Error::invalid("tensor too large"))?;
            let kv_seq_i = i32::try_from(kv_seq).map_err(|_| Error::invalid("tensor too large"))?;
            let dim_i = i32::try_from(head_dim).map_err(|_| Error::invalid("tensor too large"))?;
            let scale = scale as f32;
            let causal = i32::from(causal);
            let mut builder = self
                .runtime
                .stream
                .launch_builder(&self.runtime.kernels.attention);
            builder
                .arg(q_slice)
                .arg(k_slice)
                .arg(v_slice)
                .arg(out)
                .arg(&heads_i)
                .arg(&q_seq_i)
                .arg(&kv_seq_i)
                .arg(&dim_i)
                .arg(&scale)
                .arg(&causal);
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(total as u32))
                    .map_err(|error| cuda_error("launch", error))?;
            }
        }
        self.tensor_from_storage(backend, storage, q.shape())
    }

    fn concat_seq(
        &self,
        a: &Tensor<CudaBackend>,
        b: &Tensor<CudaBackend>,
    ) -> Result<Tensor<CudaBackend>> {
        if a.rank() != 3 || b.rank() != 3 {
            return Err(Error::invalid(
                "cuda concat_seq expects [heads, seq, head_dim]",
            ));
        }
        let heads = a.shape().dims()[0];
        let s1 = a.shape().dims()[1];
        let head_dim = a.shape().dims()[2];
        let s2 = b.shape().dims()[1];
        if b.shape().dims()[0] != heads || b.shape().dims()[2] != head_dim {
            return Err(Error::invalid("cuda concat_seq dimension mismatch"));
        }
        let a_slice = self.f32(a)?;
        let b_slice = self.f32(b)?;
        let shape = Shape::new([heads, s1 + s2, head_dim]);
        let (backend, mut storage) = self.out_f32(&shape)?;
        let total = heads * (s1 + s2) * head_dim;
        if total > 0 {
            let out = storage.f32_mut()?;
            let heads_i = i32::try_from(heads).map_err(|_| Error::invalid("tensor too large"))?;
            let s1_i = i32::try_from(s1).map_err(|_| Error::invalid("tensor too large"))?;
            let s2_i = i32::try_from(s2).map_err(|_| Error::invalid("tensor too large"))?;
            let dim_i = i32::try_from(head_dim).map_err(|_| Error::invalid("tensor too large"))?;
            let mut builder = self
                .runtime
                .stream
                .launch_builder(&self.runtime.kernels.concat_seq);
            builder
                .arg(a_slice)
                .arg(b_slice)
                .arg(out)
                .arg(&heads_i)
                .arg(&s1_i)
                .arg(&s2_i)
                .arg(&dim_i);
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(total as u32))
                    .map_err(|error| cuda_error("launch", error))?;
            }
        }
        self.tensor_from_storage(backend, storage, &shape)
    }

    fn last_row(&self, x: &Tensor<CudaBackend>) -> Result<Tensor<CudaBackend>> {
        if x.rank() != 2 {
            return Err(Error::invalid("cuda last_row expects a 2-D tensor"));
        }
        let rows = x.shape().dims()[0];
        let cols = x.shape().dims()[1];
        if rows == 0 {
            return Err(Error::invalid("cuda last_row on an empty tensor"));
        }
        let x_slice = self.f32(x)?;
        let shape = Shape::new([cols]);
        let (backend, mut storage) = self.out_f32(&shape)?;
        if cols > 0 {
            let out = storage.f32_mut()?;
            let rows_i = i32::try_from(rows).map_err(|_| Error::invalid("tensor too large"))?;
            let cols_i = i32::try_from(cols).map_err(|_| Error::invalid("tensor too large"))?;
            let mut builder = self
                .runtime
                .stream
                .launch_builder(&self.runtime.kernels.last_row);
            builder.arg(x_slice).arg(out).arg(&rows_i).arg(&cols_i);
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(cols as u32))
                    .map_err(|error| cuda_error("launch", error))?;
            }
        }
        self.tensor_from_storage(backend, storage, &shape)
    }

    #[allow(clippy::too_many_arguments)]
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
        if q.rank() != 3 || k_blocks.rank() != 4 || v_blocks.rank() != 4 {
            return Err(Error::invalid("cuda paged_attention shape mismatch"));
        }
        let (heads, q_seq, head_dim) = (
            q.shape().dims()[0],
            q.shape().dims()[1],
            q.shape().dims()[2],
        );
        let (num_blocks, block_size, block_heads, block_dim) = (
            k_blocks.shape().dims()[0],
            k_blocks.shape().dims()[1],
            k_blocks.shape().dims()[2],
            k_blocks.shape().dims()[3],
        );
        if block_heads != heads
            || block_dim != head_dim
            || v_blocks.shape().dims() != k_blocks.shape().dims()
        {
            return Err(Error::invalid("cuda paged_attention dimension mismatch"));
        }
        if seq_len == 0 || block_table.numel() * block_size < seq_len {
            return Err(Error::invalid("cuda paged_attention block table too small"));
        }
        let q_slice = self.f32(q)?;
        let k_slice = self.f32(k_blocks)?;
        let v_slice = self.f32(v_blocks)?;
        let table = self.i64(block_table)?;
        let (backend, mut storage) = self.out_f32(q.shape())?;
        let total = heads * q_seq;
        if total > 0 {
            let out = storage.f32_mut()?;
            let heads_i = i32::try_from(heads).map_err(|_| Error::invalid("tensor too large"))?;
            let q_seq_i = i32::try_from(q_seq).map_err(|_| Error::invalid("tensor too large"))?;
            let seq_i = i32::try_from(seq_len).map_err(|_| Error::invalid("tensor too large"))?;
            let dim_i = i32::try_from(head_dim).map_err(|_| Error::invalid("tensor too large"))?;
            let block_i =
                i32::try_from(block_size).map_err(|_| Error::invalid("tensor too large"))?;
            let blocks_i =
                i32::try_from(num_blocks).map_err(|_| Error::invalid("tensor too large"))?;
            let scale = scale as f32;
            let causal = i32::from(causal);
            let mut builder = self
                .runtime
                .stream
                .launch_builder(&self.runtime.kernels.paged_attention);
            builder
                .arg(q_slice)
                .arg(k_slice)
                .arg(v_slice)
                .arg(table)
                .arg(out)
                .arg(&heads_i)
                .arg(&q_seq_i)
                .arg(&seq_i)
                .arg(&dim_i)
                .arg(&block_i)
                .arg(&blocks_i)
                .arg(&scale)
                .arg(&causal);
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(total as u32))
                    .map_err(|error| cuda_error("launch", error))?;
            }
        }
        self.tensor_from_storage(backend, storage, q.shape())
    }

    #[allow(clippy::too_many_arguments)]
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
        if k_blocks.rank() != 4 || v_blocks.rank() != 4 || k_new.rank() != 3 || v_new.rank() != 3 {
            return Err(Error::invalid("cuda paged_append shape mismatch"));
        }
        let (num_blocks, block_size, heads, head_dim) = (
            k_blocks.shape().dims()[0],
            k_blocks.shape().dims()[1],
            k_blocks.shape().dims()[2],
            k_blocks.shape().dims()[3],
        );
        let n = k_new.shape().dims()[1];
        if k_new.shape().dims()[0] != heads
            || k_new.shape().dims()[2] != head_dim
            || v_new.shape().dims() != k_new.shape().dims()
            || v_blocks.shape().dims() != k_blocks.shape().dims()
        {
            return Err(Error::invalid("cuda paged_append dimension mismatch"));
        }
        let old_k = self.f32(k_blocks)?;
        let old_v = self.f32(v_blocks)?;
        let new_k = self.f32(k_new)?;
        let new_v = self.f32(v_new)?;
        let table = self.i64(block_table)?;
        let (backend, mut k_storage) = self.out_f32(k_blocks.shape())?;
        let (_, mut v_storage) = self.out_f32(v_blocks.shape())?;
        {
            let dst = k_storage.f32_mut()?;
            self.runtime
                .stream
                .memcpy_dtod(old_k, dst)
                .map_err(|error| cuda_error("dtod", error))?;
        }
        {
            let dst = v_storage.f32_mut()?;
            self.runtime
                .stream
                .memcpy_dtod(old_v, dst)
                .map_err(|error| cuda_error("dtod", error))?;
        }
        let total = n * heads * head_dim;
        if total > 0 {
            let k_out = k_storage.f32_mut()?;
            let v_out = v_storage.f32_mut()?;
            let len_i = i32::try_from(len).map_err(|_| Error::invalid("tensor too large"))?;
            let n_i = i32::try_from(n).map_err(|_| Error::invalid("tensor too large"))?;
            let heads_i = i32::try_from(heads).map_err(|_| Error::invalid("tensor too large"))?;
            let dim_i = i32::try_from(head_dim).map_err(|_| Error::invalid("tensor too large"))?;
            let block_i =
                i32::try_from(block_size).map_err(|_| Error::invalid("tensor too large"))?;
            let blocks_i =
                i32::try_from(num_blocks).map_err(|_| Error::invalid("tensor too large"))?;
            let mut builder = self
                .runtime
                .stream
                .launch_builder(&self.runtime.kernels.paged_append);
            builder
                .arg(new_k)
                .arg(new_v)
                .arg(table)
                .arg(k_out)
                .arg(v_out)
                .arg(&len_i)
                .arg(&n_i)
                .arg(&heads_i)
                .arg(&dim_i)
                .arg(&block_i)
                .arg(&blocks_i);
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(total as u32))
                    .map_err(|error| cuda_error("launch", error))?;
            }
        }
        Ok((
            self.tensor_from_storage(backend.clone(), k_storage, k_blocks.shape())?,
            self.tensor_from_storage(backend, v_storage, v_blocks.shape())?,
        ))
    }

    #[allow(clippy::too_many_arguments)]
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
        if q.rank() != 4
            || k_blocks.rank() != 4
            || v_blocks.rank() != 4
            || block_tables.rank() != 2
            || seq_lens.rank() != 1
        {
            return Err(Error::invalid("cuda paged_attention_batch shape mismatch"));
        }
        let (batch, heads, q_seq, head_dim) = (
            q.shape().dims()[0],
            q.shape().dims()[1],
            q.shape().dims()[2],
            q.shape().dims()[3],
        );
        let (num_blocks, block_size, block_heads, block_dim) = (
            k_blocks.shape().dims()[0],
            k_blocks.shape().dims()[1],
            k_blocks.shape().dims()[2],
            k_blocks.shape().dims()[3],
        );
        let max_blocks = block_tables.shape().dims()[1];
        if block_heads != heads
            || block_dim != head_dim
            || v_blocks.shape().dims() != k_blocks.shape().dims()
            || block_tables.shape().dims()[0] != batch
            || seq_lens.numel() != batch
        {
            return Err(Error::invalid(
                "cuda paged_attention_batch dimension mismatch",
            ));
        }
        let q_slice = self.f32(q)?;
        let k_slice = self.f32(k_blocks)?;
        let v_slice = self.f32(v_blocks)?;
        let tables = self.i64(block_tables)?;
        let lens = self.i64(seq_lens)?;
        let (backend, mut storage) = self.out_f32(q.shape())?;
        let total = batch * heads * q_seq;
        if total > 0 {
            let out = storage.f32_mut()?;
            let batch_i = i32::try_from(batch).map_err(|_| Error::invalid("tensor too large"))?;
            let heads_i = i32::try_from(heads).map_err(|_| Error::invalid("tensor too large"))?;
            let q_seq_i = i32::try_from(q_seq).map_err(|_| Error::invalid("tensor too large"))?;
            let dim_i = i32::try_from(head_dim).map_err(|_| Error::invalid("tensor too large"))?;
            let block_i =
                i32::try_from(block_size).map_err(|_| Error::invalid("tensor too large"))?;
            let blocks_i =
                i32::try_from(num_blocks).map_err(|_| Error::invalid("tensor too large"))?;
            let max_i =
                i32::try_from(max_blocks).map_err(|_| Error::invalid("tensor too large"))?;
            let scale = scale as f32;
            let causal = i32::from(causal);
            let mut builder = self
                .runtime
                .stream
                .launch_builder(&self.runtime.kernels.paged_attention_batch);
            builder
                .arg(q_slice)
                .arg(k_slice)
                .arg(v_slice)
                .arg(tables)
                .arg(lens)
                .arg(out)
                .arg(&batch_i)
                .arg(&heads_i)
                .arg(&q_seq_i)
                .arg(&dim_i)
                .arg(&block_i)
                .arg(&blocks_i)
                .arg(&max_i)
                .arg(&scale)
                .arg(&causal);
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(total as u32))
                    .map_err(|error| cuda_error("launch", error))?;
            }
        }
        self.tensor_from_storage(backend, storage, q.shape())
    }
}
