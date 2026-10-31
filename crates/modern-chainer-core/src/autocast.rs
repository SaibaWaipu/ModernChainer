use std::cell::Cell;

use crate::dtype::DType;
use crate::error::{Error, Result};

thread_local! {
    static AUTOCAST: Cell<Option<DType>> = const { Cell::new(None) };
}

/// The active autocast dtype, if any.
pub fn autocast_dtype() -> Option<DType> {
    AUTOCAST.with(Cell::get)
}

pub fn is_autocast_enabled() -> bool {
    autocast_dtype().is_some()
}

/// Scoped autocast policy.  Only f16/bf16 are accepted.
pub struct AutocastGuard {
    previous: Option<DType>,
}

impl AutocastGuard {
    pub fn new(dtype: DType) -> Result<Self> {
        if !matches!(dtype, DType::F16 | DType::BF16) {
            return Err(Error::UnsupportedDType(dtype));
        }
        let previous = AUTOCAST.with(|cell| cell.replace(Some(dtype)));
        Ok(Self { previous })
    }
}

impl Drop for AutocastGuard {
    fn drop(&mut self) {
        AUTOCAST.with(|cell| cell.set(self.previous));
    }
}

/// Run `f` with autocast enabled for `dtype`.
pub fn autocast<T>(dtype: DType, f: impl FnOnce() -> T) -> Result<T> {
    let _guard = AutocastGuard::new(dtype)?;
    Ok(f())
}
