use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{Gemm, GemmConfig};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use modern_chainer_core::{AdamConfig, Backend, DType, Error, Result, Shape, Tensor};

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

    pub fn i32_scalar(&self) -> Result<CudaSlice<i32>> {
        self.backend
            .runtime
            .stream
            .alloc_zeros::<i32>(1)
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
