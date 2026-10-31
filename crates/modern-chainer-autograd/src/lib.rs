//! Define-by-run automatic differentiation.
//!
//! The graph is built while `forward` runs, exactly like the original Chainer.
//! `Variable::backward` performs a reverse topological traversal and stores
//! gradients on every participating variable.  `no_grad` disables graph
//! construction for inference.

pub mod gradcheck;
pub mod node;
pub mod ops;
pub mod variable;

pub use gradcheck::gradcheck;
pub use node::{Op, apply};
pub use variable::Variable;

use std::cell::Cell;

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(Cell::get)
}

pub fn set_grad_enabled(enabled: bool) -> bool {
    GRAD_ENABLED.with(|cell| cell.replace(enabled))
}

pub struct NoGradGuard {
    previous: bool,
}

impl NoGradGuard {
    pub fn new() -> Self {
        Self {
            previous: set_grad_enabled(false),
        }
    }
}

impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        set_grad_enabled(self.previous);
    }
}

/// Run `f` with graph construction disabled.
pub fn no_grad<T>(f: impl FnOnce() -> T) -> T {
    let _guard = NoGradGuard::new();
    f()
}
