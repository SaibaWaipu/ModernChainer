//! CUDA backend scaffold.
//!
//! The runtime backend will use `cudarc` (NVRTC for kernels, cuBLAS/cuBLASLt
//! for GEMM).  This crate currently provides capability probing so that
//! applications and the Python layer can degrade gracefully when CUDA is not
//! installed.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capability {
    pub available: bool,
    pub device_count: usize,
    pub reason: &'static str,
}

/// Whether the CUDA backend was compiled in.
pub const fn is_compiled() -> bool {
    cfg!(feature = "driver")
}

pub fn probe() -> Capability {
    Capability {
        available: false,
        device_count: 0,
        reason: "CUDA driver backend is not implemented in this build",
    }
}

pub fn is_available() -> bool {
    probe().available
}

pub fn device_count() -> usize {
    probe().device_count
}
