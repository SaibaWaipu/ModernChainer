use modern_chainer_core::{Backend, DType, Result};
use modern_chainer_run::{ForkPolicy, PersistPolicy, StateBuilder, StateId, StateSpec};

#[derive(Clone, Debug)]
pub struct LayerKvHandles {
    pub k: StateId,
    pub v: StateId,
}

/// Explicit handles for a dense per-layer KV cache and its position counter.
#[derive(Clone, Debug)]
pub struct KvCacheHandles {
    pub layers: Vec<LayerKvHandles>,
    pub position: StateId,
}

pub fn declare_kv_cache<B: Backend>(
    builder: &mut StateBuilder<'_, B>,
    n_layers: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<KvCacheHandles> {
    let mut layers = Vec::with_capacity(n_layers);
    for layer in 0..n_layers {
        let spec = StateSpec::dense([n_heads, 0, head_dim], DType::F32)
            .with_fork(ForkPolicy::Share)
            .with_persist(PersistPolicy::Checkpoint);
        let k = builder.declare(format!("layers.{layer}.kv.k"), spec.clone())?;
        let v = builder.declare(format!("layers.{layer}.kv.v"), spec)?;
        layers.push(LayerKvHandles { k, v });
    }
    let position = builder.declare(
        "kv.position",
        StateSpec::scalar(DType::I64)
            .with_fork(ForkPolicy::Copy)
            .with_persist(PersistPolicy::Checkpoint),
    )?;
    Ok(KvCacheHandles { layers, position })
}
