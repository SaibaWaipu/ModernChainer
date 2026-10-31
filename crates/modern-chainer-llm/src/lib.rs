//! Dense KV-cache decoder and inference primitives.

pub mod batching;
pub mod decoder;
pub mod kv;
pub mod sampling;
pub mod speculative;

pub use batching::{BatchOutput, BatchPlan, ContinuousBatcher, RequestId, RequestState};
pub use decoder::{Decoder, DecoderConfig, DecoderHandles, generate};
pub use kv::{KvCacheHandles, LayerKvHandles};
pub use sampling::{Sampler, argmax_token, sample_token};
pub use speculative::speculative_generate;
