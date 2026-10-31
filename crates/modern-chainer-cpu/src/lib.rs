//! CPU backend: a small, dependency-light kernel set with rayon parallelism and
//! a BLAS-grade f32 GEMM via `matrixmultiply`.

mod backend;
mod kernels;
mod ops;

pub use backend::CpuTensorExt;
pub use backend::{CpuBackend, CpuOps, CpuStorage};
pub use kernels::broadcast_offset;
pub use ops::{softmax_cross_entropy, softmax_cross_entropy_backward};
