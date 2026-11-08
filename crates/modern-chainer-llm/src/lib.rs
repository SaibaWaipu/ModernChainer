//! Dense KV-cache decoder and inference primitives.

pub mod batched;
pub mod batching;
pub mod decoder;
pub mod kv;
pub mod paging;
pub mod runtime;
pub mod sampling;
pub mod speculative;

pub use batched::{PagedBatchOutput, PagedBatchRequest, PagedBatchRunner};
pub use batching::{BatchOutput, BatchPlan, ContinuousBatcher, RequestId, RequestState};
pub use decoder::{Decoder, DecoderConfig, DecoderHandles, generate};
pub use kv::{KvCacheHandles, LayerKvHandles};
pub use paging::{BlockAllocator, PagedKvCache};
pub use sampling::{
    Sampler, SamplingConfig, argmax_token, sample_token, sample_token_with_config, top_p_filter,
};
pub use speculative::speculative_generate;
