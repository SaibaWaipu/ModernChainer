//! Explicit run state for Modern Chainer.
//!
//! Define-by-Run builds the graph while `forward` executes.  State-by-Run
//! declares persistent state while a step executes, but the `Run` owns it:
//! KV caches, recurrent state, sampling state and scheduler-visible buffers are
//! introspectable, forkable and rollback-able instead of hiding inside modules.

pub mod key;
pub mod run;
pub mod spec;
pub mod store;

pub use key::{StateId, StateKey};
pub use run::{
    Run, RunContext, RunMetrics, RunMode, RunSnapshot, Step, Transaction, declare_dense,
};
pub use spec::{ForkPolicy, GradPolicy, PersistPolicy, StateLayout, StateSpec};
pub use store::{
    PagedState, ScalarValue, StateBuilder, StatePlan, StateSlot, StateStore, StateValue,
};
