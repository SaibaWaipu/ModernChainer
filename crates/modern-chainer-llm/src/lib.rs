//! Dense KV-cache decoder and inference primitives.

pub mod batched;
pub mod batching;
pub mod decoder;
pub mod kv;
pub mod paging;
pub mod sampling;
pub mod speculative;

pub use batched::{PagedBatchOutput, PagedBatchRequest, PagedBatchRunner};
pub use batching::{BatchOutput, BatchPlan, ContinuousBatcher, RequestId, RequestState};
pub use decoder::{Decoder, DecoderConfig, DecoderHandles, generate};
pub use kv::{KvCacheHandles, LayerKvHandles};
pub use paging::{BlockAllocator, PagedKvCache};
pub use sampling::{Sampler, argmax_token, sample_token};
pub use speculative::speculative_generate;
