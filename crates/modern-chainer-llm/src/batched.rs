use modern_chainer_core::{Backend, Error, Result, Rng};
use modern_chainer_run::{Run, RunMode};

use crate::batching::RequestId;
use crate::decoder::{Decoder, select_batch_row};
use crate::paging::PagedKvCache;
use crate::sampling::{Sampler, sample_token};

pub struct PagedBatchRequest<B: Backend> {
    pub id: RequestId,
    pub caches: Vec<PagedKvCache<B>>,
    pub pending: Option<u32>,
    pub generated: Vec<u32>,
    pub max_new_tokens: usize,
    pub sampler: Sampler,
    pub finished: bool,
    rng: Rng,
}

impl<B: Backend> PagedBatchRequest<B> {
    pub fn position(&self) -> usize {
        self.caches.first().map(PagedKvCache::len).unwrap_or(0)
    }
}

#[derive(Clone, Debug)]
pub struct PagedBatchOutput {
    pub plan: Vec<RequestId>,
    pub tokens: Vec<(RequestId, u32)>,
}

/// Decode-batched runner backed by per-request paged KV caches.
///
/// Prefill uses the dense `Run` path and converts the resulting KV into paged
/// blocks.  Decode then appends one token per request and runs a single batched
/// paged-attention kernel over all active requests.
pub struct PagedBatchRunner<B: Backend> {
    model: Decoder<B>,
    backend: B,
    block_size: usize,
    capacity_blocks: usize,
    seed: u64,
    requests: Vec<PagedBatchRequest<B>>,
    next_id: u64,
    last_plan: Vec<RequestId>,
}

impl<B: Backend> PagedBatchRunner<B> {
    pub fn new(
        model: Decoder<B>,
        backend: B,
        block_size: usize,
        capacity_blocks: usize,
        seed: u64,
    ) -> Result<Self> {
        if block_size == 0 || capacity_blocks == 0 {
            return Err(Error::invalid(
                "paged batch runner needs positive block_size and capacity",
            ));
        }
        Ok(Self {
            model,
            backend,
            block_size,
            capacity_blocks,
            seed,
            requests: Vec::new(),
            next_id: 0,
            last_plan: Vec::new(),
        })
    }

    pub fn add_request(
        &mut self,
        prompt: Vec<u32>,
        max_new_tokens: usize,
        sampler: Sampler,
    ) -> Result<RequestId> {
        if prompt.is_empty() {
            return Err(Error::invalid("request prompt must not be empty"));
        }
        if max_new_tokens == 0 {
            return Err(Error::invalid("max_new_tokens must be positive"));
        }
        let needed = (prompt.len() + max_new_tokens).div_ceil(self.block_size);
        if needed > self.capacity_blocks {
            return Err(Error::invalid(format!(
                "request needs {needed} blocks but capacity is {}",
                self.capacity_blocks
            )));
        }
        let id = RequestId(self.next_id);
        self.next_id += 1;

        // Prefill on the dense path, then move the KV into paged blocks.
        let mut run = Run::new(
            self.model.clone(),
            self.backend.clone(),
            RunMode::Inference,
            self.seed.wrapping_add(id.0),
        )?;
        let logits = run.step(prompt.clone())?;
        let handles = run.handles().clone();
        let mut caches = Vec::with_capacity(self.model.config().n_layers);
        for layer in 0..self.model.config().n_layers {
            let k = run.state().tensor(handles.kv.layers[layer].k)?;
            let v = run.state().tensor(handles.kv.layers[layer].v)?;
            caches.push(PagedKvCache::from_dense_with_capacity(
                self.backend.clone(),
                &k,
                &v,
                self.block_size,
                self.capacity_blocks,
            )?);
        }
        let mut rng = run.rng().clone();
        let first = sample_token(&logits, &sampler, &mut rng)?;
        self.requests.push(PagedBatchRequest {
            id,
            caches,
            pending: Some(first),
            generated: Vec::new(),
            max_new_tokens,
            sampler,
            finished: false,
            rng,
        });
        Ok(id)
    }

    /// One batched decode step over every active request.
    pub fn step(&mut self) -> Result<PagedBatchOutput> {
        let indices: Vec<usize> = self
            .requests
            .iter()
            .enumerate()
            .filter(|(_, request)| !request.finished && request.pending.is_some())
            .map(|(index, _)| index)
            .collect();
        if indices.is_empty() {
            return Ok(PagedBatchOutput {
                plan: Vec::new(),
                tokens: Vec::new(),
            });
        }

        let mut active_caches = Vec::with_capacity(indices.len());
        let mut tokens = Vec::with_capacity(indices.len());
        for &index in &indices {
            let request = &mut self.requests[index];
            active_caches.push(std::mem::take(&mut request.caches));
            tokens.push(request.pending.expect("filtered above"));
        }

        let logits = self
            .model
            .forward_decode_batch(&mut active_caches, &tokens)?;

        let mut emitted = Vec::with_capacity(indices.len());
        let mut plan = Vec::with_capacity(indices.len());
        for (slot, &index) in indices.iter().enumerate() {
            let request = &mut self.requests[index];
            request.caches = std::mem::take(&mut active_caches[slot]);
            let token = request.pending.take().expect("filtered above");
            request.generated.push(token);
            emitted.push((request.id, token));
            plan.push(request.id);
            if request.generated.len() >= request.max_new_tokens {
                request.finished = true;
                continue;
            }
            let row = select_batch_row(&logits, slot)?;
            let next = sample_token(&row, &request.sampler, &mut request.rng)?;
            request.pending = Some(next);
        }

        self.last_plan = plan.clone();
        Ok(PagedBatchOutput {
            plan,
            tokens: emitted,
        })
    }

    pub fn run_until_finished(&mut self) -> Result<Vec<(RequestId, Vec<u32>)>> {
        while self.active() > 0 {
            self.step()?;
        }
        Ok(self
            .requests
            .iter()
            .map(|request| (request.id, request.generated.clone()))
            .collect())
    }

    pub fn last_plan(&self) -> &[RequestId] {
        &self.last_plan
    }

    pub fn active(&self) -> usize {
        self.requests
            .iter()
            .filter(|request| !request.finished)
            .count()
    }

    pub fn len(&self) -> usize {
        self.requests.len()
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    pub fn state(&self, id: RequestId) -> Option<&PagedBatchRequest<B>> {
        self.requests.iter().find(|request| request.id == id)
    }

    pub fn generated(&self, id: RequestId) -> Option<&[u32]> {
        self.state(id).map(|request| request.generated.as_slice())
    }

    pub fn requests(&self) -> impl Iterator<Item = &PagedBatchRequest<B>> {
        self.requests.iter()
    }

    pub fn remove_finished(&mut self) {
        self.requests.retain(|request| !request.finished);
    }
}
