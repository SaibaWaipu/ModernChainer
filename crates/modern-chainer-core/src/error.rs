use std::fmt;

use thiserror::Error;

use crate::dtype::DType;
use crate::shape::Shape;

#[derive(Debug, Error)]
pub enum Error {
    #[error("shape mismatch: {context} (left={left}, right={right})")]
    ShapeMismatch {
        context: String,
        left: Shape,
        right: Shape,
    },

    #[error("cannot broadcast shapes {left} and {right}")]
    Broadcast { left: Shape, right: Shape },

    #[error("dtype mismatch: expected {expected}, got {actual}")]
    DTypeMismatch { expected: DType, actual: DType },

    #[error("unsupported dtype {0} for this operation")]
    UnsupportedDType(DType),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("index out of bounds: index={index}, len={len}")]
    OutOfBounds { index: usize, len: usize },

    #[error("unsupported operation: {0}")]
    Unsupported(String),

    #[error("backend error: {0}")]
    Backend(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

impl Error {
    pub fn unsupported(op: impl fmt::Display) -> Self {
        Error::Unsupported(op.to_string())
    }

    pub fn invalid(msg: impl fmt::Display) -> Self {
        Error::InvalidArgument(msg.to_string())
    }

    pub fn shape_mismatch(context: impl fmt::Display, left: Shape, right: Shape) -> Self {
        Error::ShapeMismatch {
            context: context.to_string(),
            left,
            right,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
