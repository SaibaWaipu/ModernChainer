//! Modern Chainer: a define-by-run deep learning library with a Rust core.
//!
//! ```no_run
//! use modern_chainer::prelude::*;
//!
//! # fn main() -> modern_chainer::Result<()> {
//! let model = L::Linear::new(4, 2, 0)?;
//! let x = Variable::new(Tensor::from_f32(CpuBackend::new(), &[1.0; 8], [2, 4])?);
//! let y = model.forward(&x)?;
//! let loss = F::sum(&y)?;
//! loss.backward()?;
//! # Ok(())
//! # }
//! ```

pub use modern_chainer_autograd::{
    NoGradGuard, Op, Variable, apply, gradcheck, is_grad_enabled, no_grad, set_grad_enabled,
};
pub use modern_chainer_core::{
    Activation, AdamConfig, Backend, DType, Device, Error, Ops, Result, Shape, Strides, Tensor,
    he_normal,
};
pub use modern_chainer_cpu::{CpuBackend, CpuOps, CpuStorage};
pub use modern_chainer_cuda::{
    ComputeType, CudaBackend, CudaGraphRunner, CudaOps, CudaStorage, GraphAdam, GraphMatmul,
};
pub use modern_chainer_llm::{
    BatchOutput, BatchPlan, BlockAllocator, ContinuousBatcher, Decoder, DecoderConfig,
    DecoderHandles, PagedBatchOutput, PagedBatchRequest, PagedBatchRunner, PagedKvCache, RequestId,
    RequestState, Sampler, argmax_token, generate, sample_token, speculative_generate,
};
pub use modern_chainer_nn::{Linear, Module, Parameter, Sequential};
pub use modern_chainer_optim::{Adam, MomentumSGD, Optimizer, SGD, zero_grad};
pub use modern_chainer_run::{
    Run, RunContext, RunMode, ScalarValue, StateBuilder, StateId, StateKey, StateSpec, StateStore,
    Step, Transaction,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Functional API (`F.*` in Python).
#[allow(non_snake_case)]
pub mod F {
    pub use modern_chainer_autograd::ops::*;
}

/// Links and modules (`L.*` in Python).
#[allow(non_snake_case)]
pub mod L {
    pub use modern_chainer_nn::*;
}

/// Optimizers (`chainer.optimizers.*` in Python).
pub mod optimizers {
    pub use modern_chainer_optim::*;
}

pub mod data {
    pub use modern_chainer_data::*;
}

pub mod serial {
    pub use modern_chainer_serial::*;
}

pub mod run {
    pub use modern_chainer_run::*;
}

pub mod llm {
    pub use modern_chainer_llm::*;
}

pub mod cuda {
    pub use modern_chainer_cuda::*;
}

pub mod vulkan {
    pub use modern_chainer_vulkan::*;
}

pub mod prelude {
    pub use crate::F;
    pub use crate::F as functional;
    pub use crate::L;
    pub use crate::{
        Activation, Adam, AdamConfig, Backend, BatchPlan, BlockAllocator, ContinuousBatcher,
        CpuBackend, CudaBackend, DType, Decoder, DecoderConfig, Device, Error, Linear, Module,
        MomentumSGD, Ops, Optimizer, PagedBatchRunner, Parameter, RequestId, Result, Run, RunMode,
        SGD, Sampler, Shape, StateId, StateKey, StateSpec, StateStore, Step, Tensor, Variable,
        ZeroGrad, generate, no_grad, speculative_generate, zero_grad,
    };
}

/// Convenience alias used by the prelude.
pub trait ZeroGrad {
    fn zero_grad(&self);
}

impl<B: Backend> ZeroGrad for [Parameter<B>] {
    fn zero_grad(&self) {
        zero_grad(self);
    }
}

impl<B: Backend> ZeroGrad for Vec<Parameter<B>> {
    fn zero_grad(&self) {
        zero_grad(self);
    }
}
