use modern_chainer_core::{DType, Shape};

/// Whether a state slot participates in the autograd graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GradPolicy {
    /// Plain tensor state (inference, running statistics, caches).
    Detach,
    /// The slot holds a `Variable` so truncated BPTT / differentiable state works.
    Differentiable,
}

/// How a slot is copied when a run is forked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForkPolicy {
    /// Share immutable backing storage (dense tensors, `Arc`-backed).
    Share,
    /// Deep copy when forked.
    Copy,
    /// Copy-on-write blocks (paged state).
    Cow,
    /// Recompute from the parent on demand.
    Recompute,
}

/// Whether a slot is part of checkpoints or exported weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PersistPolicy {
    Ephemeral,
    Checkpoint,
    Export,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StateLayout {
    /// A dense tensor with an initial shape.  The shape may grow during a run
    /// (for example a KV cache), so the initial shape is a declaration hint.
    Dense(Shape),
    /// A single typed scalar.
    Scalar,
    /// Block-addressed storage (reserved for PagedAttention).
    Paged {
        block_size: usize,
        num_blocks: usize,
        trailing: Vec<usize>,
    },
}

impl StateLayout {
    pub fn kind(&self) -> &'static str {
        match self {
            StateLayout::Dense(_) => "dense",
            StateLayout::Scalar => "scalar",
            StateLayout::Paged { .. } => "paged",
        }
    }
}

/// The declaration of a state slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateSpec {
    pub layout: StateLayout,
    pub dtype: DType,
    pub grad: GradPolicy,
    pub fork: ForkPolicy,
    pub persist: PersistPolicy,
}

impl StateSpec {
    pub fn dense(shape: impl Into<Shape>, dtype: DType) -> Self {
        Self {
            layout: StateLayout::Dense(shape.into()),
            dtype,
            grad: GradPolicy::Detach,
            fork: ForkPolicy::Share,
            persist: PersistPolicy::Ephemeral,
        }
    }

    pub fn scalar(dtype: DType) -> Self {
        Self {
            layout: StateLayout::Scalar,
            dtype,
            grad: GradPolicy::Detach,
            fork: ForkPolicy::Share,
            persist: PersistPolicy::Ephemeral,
        }
    }

    pub fn paged(
        block_size: usize,
        num_blocks: usize,
        trailing: impl Into<Vec<usize>>,
        dtype: DType,
    ) -> Self {
        Self {
            layout: StateLayout::Paged {
                block_size,
                num_blocks,
                trailing: trailing.into(),
            },
            dtype,
            grad: GradPolicy::Detach,
            fork: ForkPolicy::Cow,
            persist: PersistPolicy::Checkpoint,
        }
    }

    pub fn with_grad(mut self, policy: GradPolicy) -> Self {
        self.grad = policy;
        self
    }

    pub fn with_fork(mut self, policy: ForkPolicy) -> Self {
        self.fork = policy;
        self
    }

    pub fn with_persist(mut self, policy: PersistPolicy) -> Self {
        self.persist = policy;
        self
    }

    pub fn is_compatible(&self, other: &Self) -> bool {
        self.layout == other.layout && self.dtype == other.dtype
    }
}
