use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{Gemm, GemmConfig};
use cudarc::driver::{CudaGraph, LaunchConfig, PushKernelArg, sys};

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
