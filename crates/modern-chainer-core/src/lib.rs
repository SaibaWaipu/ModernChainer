//! Core primitives shared by every Modern Chainer backend.
//!
//! This crate deliberately contains no kernels.  It defines the vocabulary
//! (`Tensor`, `Shape`, `DType`, `Device`, `Backend`, `Ops`) that CPU, CUDA and
//! Vulkan backends implement.

pub mod activation;
pub mod autocast;
pub mod device;
pub mod dtype;
pub mod error;
pub mod init;
pub mod rng;
pub mod shape;
pub mod storage;
pub mod tensor;

pub use activation::Activation;
pub use autocast::{AutocastGuard, autocast, autocast_dtype, is_autocast_enabled};
pub use device::Device;
pub use dtype::DType;
pub use error::{Error, Result};
pub use init::he_normal;
pub use rng::Rng;
pub use shape::{Shape, Strides};
pub use storage::{AdamConfig, Backend, Ops, Storage};
pub use tensor::Tensor;
