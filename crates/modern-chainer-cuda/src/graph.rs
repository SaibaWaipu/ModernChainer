use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{Gemm, GemmConfig};
use cudarc::driver::{CudaGraph, CudaSlice, LaunchConfig, PushKernelArg, sys};

use modern_chainer_core::{AdamConfig, Backend, DType, Error, Result, Shape, Tensor};

use crate::backend::CudaBackend;

/// A replayed CUDA graph.  Captured closures must only enqueue work on the
/// backend stream; allocations and synchronization are not capturable.
pub struct CudaGraphRunner {
    graph: CudaGraph,
}

impl CudaGraphRunner {
    pub fn replay(&self) -> Result<()> {
        self.graph
            .launch()
            .map_err(|error| Error::Backend(format!("cuda graph launch: {error}")))
    }

    pub fn upload(&self) -> Result<()> {
        self.graph
            .upload()
            .map_err(|error| Error::Backend(format!("cuda graph upload: {error}")))
    }
}

impl CudaBackend {
    /// Capture a sequence of launches on this backend's stream.
    ///
    /// The closure must not allocate host/device memory or synchronize.  Use
    /// preallocated tensors and in-place kernels for capturable work.
    pub fn capture(&self, f: impl FnOnce() -> Result<()>) -> Result<CudaGraphRunner> {
        self.runtime
            .stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|error| Error::Backend(format!("cuda begin capture: {error}")))?;
        let result = f();
        let graph = self
            .runtime
            .stream
            .end_capture(sys::CUgraphInstantiate_flags(0))
            .map_err(|error| Error::Backend(format!("cuda end capture: {error}")))?;
        result?;
        let graph = graph.ok_or_else(|| Error::invalid("CUDA capture produced no graph"))?;
        Ok(CudaGraphRunner { graph })
    }
}

/// A captured matmul with a preallocated output buffer.
pub struct GraphMatmul {
    graph: CudaGraphRunner,
    out: Tensor<CudaBackend>,
    _a: Tensor<CudaBackend>,
    _b: Tensor<CudaBackend>,
}

impl GraphMatmul {
    pub fn replay(&self) -> Result<()> {
        self.graph.replay()
    }

    pub fn out(&self) -> &Tensor<CudaBackend> {
        &self.out
    }
}

impl CudaBackend {
    pub fn graph_matmul(
        &self,
        a: &Tensor<CudaBackend>,
        b: &Tensor<CudaBackend>,
    ) -> Result<GraphMatmul> {
        if a.rank() != 2 || b.rank() != 2 {
            return Err(Error::invalid("graph matmul expects 2-D tensors"));
        }
        let (m, k) = (a.shape().dims()[0], a.shape().dims()[1]);
        let (k2, n) = (b.shape().dims()[0], b.shape().dims()[1]);
        if k != k2 {
            return Err(Error::shape_mismatch(
                "graph matmul inner dimension",
                a.shape().clone(),
                b.shape().clone(),
            ));
        }
        let a_slice = a.storage().f32()?;
        let b_slice = b.storage().f32()?;
        let mut storage = self.alloc_zeroed_for(m * n * 4, DType::F32)?;
        let out_slice = storage.f32_mut()?;
        let graph = self.capture(|| {
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
            unsafe { self.runtime.blas.gemm(config, b_slice, a_slice, out_slice) }
                .map_err(|error| Error::Backend(format!("graph cublas: {error}")))
        })?;
        let shape = Shape::new([m, n]);
        let out = Tensor::from_storage(
            self.clone(),
            storage,
            DType::F32,
            shape.clone(),
            shape.contiguous_strides(),
            0,
        )?;
        Ok(GraphMatmul {
            graph,
            out,
            _a: a.clone(),
            _b: b.clone(),
        })
    }

    /// Capture a fused Adam update into preallocated buffers.
    pub fn graph_adam(
        &self,
        param: &Tensor<CudaBackend>,
        grad: &Tensor<CudaBackend>,
        m: &Tensor<CudaBackend>,
        v: &Tensor<CudaBackend>,
        cfg: &AdamConfig,
    ) -> Result<GraphAdam> {
        let n = param.numel();
        if grad.numel() != n || m.numel() != n || v.numel() != n {
            return Err(Error::invalid("graph adam shape mismatch"));
        }
        let p_slice = param.storage().f32()?;
        let g_slice = grad.storage().f32()?;
        let m_slice = m.storage().f32()?;
        let v_slice = v.storage().f32()?;
        let mut p_storage = self.alloc_zeroed_for(n * 4, DType::F32)?;
        let mut m_storage = self.alloc_zeroed_for(n * 4, DType::F32)?;
        let mut v_storage = self.alloc_zeroed_for(n * 4, DType::F32)?;
        let n_i = i32::try_from(n).map_err(|_| Error::invalid("tensor too large"))?;
        let lr = cfg.lr as f32;
        let b1 = cfg.beta1 as f32;
        let b2 = cfg.beta2 as f32;
        let eps = cfg.eps as f32;
        let bc1 = 1.0 - b1.powi(cfg.step as i32);
        let bc2 = 1.0 - b2.powi(cfg.step as i32);
        {
            let new_p = p_storage.f32_mut()?;
            let new_m = m_storage.f32_mut()?;
            let new_v = v_storage.f32_mut()?;
            let graph = self.capture(|| {
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
                        .map_err(|error| Error::Backend(format!("graph adam: {error}")))?;
                }
                Ok(())
            })?;
            let shape = param.shape().clone();
            let param_out = Tensor::from_storage(
                self.clone(),
                p_storage,
                DType::F32,
                shape.clone(),
                shape.contiguous_strides(),
                0,
            )?;
            let m_out = Tensor::from_storage(
                self.clone(),
                m_storage,
                DType::F32,
                shape.clone(),
                shape.contiguous_strides(),
                0,
            )?;
            let v_out = Tensor::from_storage(
                self.clone(),
                v_storage,
                DType::F32,
                shape.clone(),
                shape.contiguous_strides(),
                0,
            )?;
            Ok(GraphAdam {
                graph,
                param: param_out,
                m: m_out,
                v: v_out,
                _param_in: param.clone(),
                _grad: grad.clone(),
                _m_in: m.clone(),
                _v_in: v.clone(),
            })
        }
    }
}

pub struct GraphAdam {
    graph: CudaGraphRunner,
    param: Tensor<CudaBackend>,
    m: Tensor<CudaBackend>,
    v: Tensor<CudaBackend>,
    _param_in: Tensor<CudaBackend>,
    _grad: Tensor<CudaBackend>,
    _m_in: Tensor<CudaBackend>,
    _v_in: Tensor<CudaBackend>,
}

impl GraphAdam {
    pub fn replay(&self) -> Result<()> {
        self.graph.replay()
    }

    pub fn param(&self) -> &Tensor<CudaBackend> {
        &self.param
    }

    pub fn m(&self) -> &Tensor<CudaBackend> {
        &self.m
    }

    pub fn v(&self) -> &Tensor<CudaBackend> {
        &self.v
    }
}

#[derive(Clone, Debug)]
pub struct GraphTrainerConfig {
    pub batch: usize,
    pub in_features: usize,
    pub out_features: usize,
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
}

/// A capture-replayable Linear + MSE + Adam training step.
///
/// All parameters, activations, gradients and optimizer state live in fixed
/// device buffers.  The step counter is device-resident and incremented inside
/// the graph, so replay performs a complete Adam step without host arguments.
pub struct GraphLinearTrainer {
    backend: CudaBackend,
    config: GraphTrainerConfig,
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
    x: CudaSlice<f32>,
    t: CudaSlice<f32>,
    y: CudaSlice<f32>,
    diff: CudaSlice<f32>,
    loss: CudaSlice<f32>,
    grad_y: CudaSlice<f32>,
    grad_w: CudaSlice<f32>,
    grad_b: CudaSlice<f32>,
    m_w: CudaSlice<f32>,
    v_w: CudaSlice<f32>,
    m_b: CudaSlice<f32>,
    v_b: CudaSlice<f32>,
    step: CudaSlice<i32>,
    graph: Option<CudaGraphRunner>,
}

impl GraphLinearTrainer {
    pub fn new(backend: CudaBackend, config: GraphTrainerConfig, seed: u64) -> Result<Self> {
        if config.batch == 0
            || config.in_features == 0
            || config.out_features == 0
            || config.lr <= 0.0
        {
            return Err(Error::invalid("invalid graph trainer config"));
        }
        let stream = backend.runtime.stream.clone();
        let alloc = |len: usize| {
            stream
                .alloc_zeros::<f32>(len)
                .map_err(|error| Error::Backend(format!("cuda alloc: {error}")))
        };
        let mut w = alloc(config.out_features * config.in_features)?;
        let weight = modern_chainer_core::Rng::seed_from_u64(seed).normal_vec(
            config.out_features * config.in_features,
            0.0,
            (2.0 / config.in_features as f64).sqrt(),
        );
        stream
            .memcpy_htod(&weight, &mut w)
            .map_err(|error| Error::Backend(format!("htod: {error}")))?;
        let step = stream
            .alloc_zeros::<i32>(1)
            .map_err(|error| Error::Backend(format!("cuda alloc: {error}")))?;
        Ok(Self {
            w,
            b: alloc(config.out_features)?,
            x: alloc(config.batch * config.in_features)?,
            t: alloc(config.batch * config.out_features)?,
            y: alloc(config.batch * config.out_features)?,
            diff: alloc(config.batch * config.out_features)?,
            loss: alloc(1)?,
            grad_y: alloc(config.batch * config.out_features)?,
            grad_w: alloc(config.out_features * config.in_features)?,
            grad_b: alloc(config.out_features)?,
            m_w: alloc(config.out_features * config.in_features)?,
            v_w: alloc(config.out_features * config.in_features)?,
            m_b: alloc(config.out_features)?,
            v_b: alloc(config.out_features)?,
            step,
            backend,
            config,
            graph: None,
        })
    }

    pub fn set_input(&mut self, values: &[f32]) -> Result<()> {
        if values.len() != self.config.batch * self.config.in_features {
            return Err(Error::invalid("graph trainer input size mismatch"));
        }
        self.backend
            .runtime
            .stream
            .memcpy_htod(values, &mut self.x)
            .map_err(|error| Error::Backend(format!("htod: {error}")))
    }

    pub fn set_target(&mut self, values: &[f32]) -> Result<()> {
        if values.len() != self.config.batch * self.config.out_features {
            return Err(Error::invalid("graph trainer target size mismatch"));
        }
        self.backend
            .runtime
            .stream
            .memcpy_htod(values, &mut self.t)
            .map_err(|error| Error::Backend(format!("htod: {error}")))
    }

    pub fn capture(&mut self) -> Result<()> {
        let backend = self.backend.clone();
        let graph = {
            let this = &mut *self;
            backend.capture(|| this.enqueue_step())?
        };
        self.graph = Some(graph);
        Ok(())
    }

    pub fn replay(&self) -> Result<()> {
        self.graph
            .as_ref()
            .ok_or_else(|| Error::invalid("graph trainer is not captured"))?
            .replay()
    }

    pub fn synchronize(&self) -> Result<()> {
        self.backend.synchronize()
    }

    pub fn loss(&self) -> Result<f32> {
        let host = self
            .backend
            .runtime
            .stream
            .clone_dtoh(&self.loss)
            .map_err(|error| Error::Backend(format!("dtoh: {error}")))?;
        Ok(host[0])
    }

    pub fn weights(&self) -> Result<(Vec<f32>, Vec<f32>)> {
        let w = self
            .backend
            .runtime
            .stream
            .clone_dtoh(&self.w)
            .map_err(|error| Error::Backend(format!("dtoh: {error}")))?;
        let b = self
            .backend
            .runtime
            .stream
            .clone_dtoh(&self.b)
            .map_err(|error| Error::Backend(format!("dtoh: {error}")))?;
        Ok((w, b))
    }

    pub fn step_count(&self) -> Result<i32> {
        let host = self
            .backend
            .runtime
            .stream
            .clone_dtoh(&self.step)
            .map_err(|error| Error::Backend(format!("dtoh: {error}")))?;
        Ok(host[0])
    }

    fn enqueue_step(&mut self) -> Result<()> {
        let runtime = std::sync::Arc::clone(&self.backend.runtime);
        let n = self.config.batch * self.config.out_features;
        let n_i = i32::try_from(n).map_err(|_| Error::invalid("tensor too large"))?;
        let rows =
            i32::try_from(self.config.batch).map_err(|_| Error::invalid("tensor too large"))?;
        let cols = i32::try_from(self.config.out_features)
            .map_err(|_| Error::invalid("tensor too large"))?;
        let n_blocks = u32::try_from(n).map_err(|_| Error::invalid("tensor too large"))?;

        // loss = 0
        let zero = 0.0f32;
        let one = 1i32;
        {
            let mut builder = runtime.stream.launch_builder(&runtime.kernels.fill);
            builder.arg(&mut self.loss).arg(&zero).arg(&one);
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(1))
                    .map_err(|error| Error::Backend(format!("graph fill: {error}")))?;
            }
        }

        // y = x @ w^T
        unsafe {
            runtime.blas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_T,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: cols,
                    n: rows,
                    k: self.config.in_features as i32,
                    alpha: 1.0,
                    lda: self.config.in_features as i32,
                    ldb: self.config.in_features as i32,
                    beta: 0.0,
                    ldc: cols,
                },
                &self.w,
                &self.x,
                &mut self.y,
            )
        }
        .map_err(|error| Error::Backend(format!("graph cublas forward: {error}")))?;

        // y += b
        {
            let mut builder = runtime.stream.launch_builder(&runtime.kernels.bias_add);
            builder.arg(&mut self.y).arg(&self.b).arg(&rows).arg(&cols);
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(n_blocks))
                    .map_err(|error| Error::Backend(format!("graph bias: {error}")))?;
            }
        }

        // diff, loss
        {
            let mut builder = runtime.stream.launch_builder(&runtime.kernels.mse_forward);
            builder
                .arg(&self.y)
                .arg(&self.t)
                .arg(&mut self.diff)
                .arg(&mut self.loss)
                .arg(&n_i);
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(n_blocks))
                    .map_err(|error| Error::Backend(format!("graph mse: {error}")))?;
            }
        }

        // grad_y = 2 * diff / n
        {
            let scale = 2.0f32 / n as f32;
            let mut builder = runtime.stream.launch_builder(&runtime.kernels.grad_y);
            builder
                .arg(&self.diff)
                .arg(&mut self.grad_y)
                .arg(&n_i)
                .arg(&scale);
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(n_blocks))
                    .map_err(|error| Error::Backend(format!("graph grad_y: {error}")))?;
            }
        }

        // grad_w = grad_y^T @ x (row-major: C = A^T * B)
        unsafe {
            runtime.blas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_T,
                    m: self.config.in_features as i32,
                    n: cols,
                    k: rows,
                    alpha: 1.0,
                    lda: self.config.in_features as i32,
                    ldb: cols,
                    beta: 0.0,
                    ldc: self.config.in_features as i32,
                },
                &self.x,
                &self.grad_y,
                &mut self.grad_w,
            )
        }
        .map_err(|error| Error::Backend(format!("graph cublas grad_w: {error}")))?;

        // grad_b = sum rows
        {
            let mut builder = runtime.stream.launch_builder(&runtime.kernels.grad_b);
            builder
                .arg(&self.grad_y)
                .arg(&mut self.grad_b)
                .arg(&rows)
                .arg(&cols);
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(cols as u32))
                    .map_err(|error| Error::Backend(format!("graph grad_b: {error}")))?;
            }
        }

        // step += 1
        {
            let mut builder = runtime.stream.launch_builder(&runtime.kernels.bump_step);
            builder.arg(&mut self.step);
            unsafe {
                builder
                    .launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .map_err(|error| Error::Backend(format!("graph step: {error}")))?;
            }
        }

        let lr = self.config.lr as f32;
        let b1 = self.config.beta1 as f32;
        let b2 = self.config.beta2 as f32;
        let eps = self.config.eps as f32;
        let params = self.config.out_features * self.config.in_features;
        let params_i = i32::try_from(params).map_err(|_| Error::invalid("tensor too large"))?;
        {
            let mut builder = runtime.stream.launch_builder(&runtime.kernels.adam_step);
            builder
                .arg(&mut self.w)
                .arg(&self.grad_w)
                .arg(&mut self.m_w)
                .arg(&mut self.v_w)
                .arg(&params_i)
                .arg(&lr)
                .arg(&b1)
                .arg(&b2)
                .arg(&eps)
                .arg(&self.step);
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(
                        u32::try_from(params).map_err(|_| Error::invalid("tensor too large"))?,
                    ))
                    .map_err(|error| Error::Backend(format!("graph adam w: {error}")))?;
            }
        }
        {
            let mut builder = runtime.stream.launch_builder(&runtime.kernels.adam_step);
            builder
                .arg(&mut self.b)
                .arg(&self.grad_b)
                .arg(&mut self.m_b)
                .arg(&mut self.v_b)
                .arg(&cols)
                .arg(&lr)
                .arg(&b1)
                .arg(&b2)
                .arg(&eps)
                .arg(&self.step);
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(cols as u32))
                    .map_err(|error| Error::Backend(format!("graph adam b: {error}")))?;
            }
        }
        Ok(())
    }
}
