use modern_chainer_core::{Activation, Backend, DType, Error, Ops, Result, Rng, Shape, Tensor};
use modern_chainer_run::{Run, RunContext, ScalarValue, StateBuilder, Step};

use crate::kv::{KvCacheHandles, declare_kv_cache};
use crate::paging::PagedKvCache;
use crate::sampling::{Sampler, sample_token};

#[derive(Clone, Debug)]
pub struct DecoderConfig {
    pub vocab: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub max_seq: usize,
    pub eps: f64,
}

impl DecoderConfig {
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads.max(1)
    }

    pub fn validate(&self) -> Result<()> {
        if self.vocab == 0 || self.d_model == 0 || self.n_layers == 0 {
            return Err(Error::invalid("decoder config has zero dimensions"));
        }
        if self.n_heads == 0 || !self.d_model.is_multiple_of(self.n_heads) {
            return Err(Error::invalid(
                "d_model must be divisible by a positive n_heads",
            ));
        }
        if self.d_ff == 0 || self.max_seq == 0 {
            return Err(Error::invalid("d_ff and max_seq must be positive"));
        }
        Ok(())
    }
}

#[derive(Clone)]
struct DecoderLayer<B: Backend> {
    ln1_g: Tensor<B>,
    ln1_b: Tensor<B>,
    wq: Tensor<B>,
    bq: Tensor<B>,
    wk: Tensor<B>,
    bk: Tensor<B>,
    wv: Tensor<B>,
    bv: Tensor<B>,
    wo: Tensor<B>,
    bo: Tensor<B>,
    ln2_g: Tensor<B>,
    ln2_b: Tensor<B>,
    w1: Tensor<B>,
    b1: Tensor<B>,
    w2: Tensor<B>,
    b2: Tensor<B>,
}

#[derive(Clone)]
pub struct Decoder<B: Backend> {
    config: DecoderConfig,
    tok_emb: Tensor<B>,
    layers: Vec<DecoderLayer<B>>,
    ln_f_g: Tensor<B>,
    ln_f_b: Tensor<B>,
    lm_head: Tensor<B>,
}

#[derive(Clone, Debug)]
pub struct DecoderHandles {
    pub kv: KvCacheHandles,
}

fn matrix<B: Backend>(
    backend: &B,
    rng: &mut Rng,
    rows: usize,
    cols: usize,
    std: f64,
) -> Result<Tensor<B>> {
    Tensor::from_f32(
        backend.clone(),
        &rng.normal_vec(rows * cols, 0.0, std),
        [rows, cols],
    )
}

fn vector<B: Backend>(backend: &B, len: usize, value: f32) -> Result<Tensor<B>> {
    Tensor::from_f32(backend.clone(), &vec![value; len], [len])
}

fn matrix_std(fan_in: usize) -> f64 {
    (2.0 / fan_in.max(1) as f64).sqrt()
}

impl<B: Backend> Decoder<B> {
    pub fn new(backend: B, config: DecoderConfig, seed: u64) -> Result<Self> {
        config.validate()?;
        let mut rng = Rng::seed_from_u64(seed);
        let d = config.d_model;
        let d_ff = config.d_ff;
        let tok_emb = matrix(&backend, &mut rng, config.vocab, d, 0.02)?;

        let mut layers = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            let std = matrix_std(d);
            layers.push(DecoderLayer {
                ln1_g: vector(&backend, d, 1.0)?,
                ln1_b: vector(&backend, d, 0.0)?,
                wq: matrix(&backend, &mut rng, d, d, std)?,
                bq: vector(&backend, d, 0.0)?,
                wk: matrix(&backend, &mut rng, d, d, std)?,
                bk: vector(&backend, d, 0.0)?,
                wv: matrix(&backend, &mut rng, d, d, std)?,
                bv: vector(&backend, d, 0.0)?,
                wo: matrix(&backend, &mut rng, d, d, std)?,
                bo: vector(&backend, d, 0.0)?,
                ln2_g: vector(&backend, d, 1.0)?,
                ln2_b: vector(&backend, d, 0.0)?,
                w1: matrix(&backend, &mut rng, d_ff, d, matrix_std(d))?,
                b1: vector(&backend, d_ff, 0.0)?,
                w2: matrix(&backend, &mut rng, d, d_ff, matrix_std(d_ff))?,
                b2: vector(&backend, d, 0.0)?,
            });
        }

        let ln_f_g = vector(&backend, d, 1.0)?;
        let ln_f_b = vector(&backend, d, 0.0)?;
        let lm_head = matrix(&backend, &mut rng, config.vocab, d, 0.02)?;

        Ok(Self {
            config,
            tok_emb,
            layers,
            ln_f_g,
            ln_f_b,
            lm_head,
        })
    }

    pub fn config(&self) -> &DecoderConfig {
        &self.config
    }

    pub fn parameters(&self) -> Vec<(String, Tensor<B>)> {
        let mut out = vec![("tok_emb".to_string(), self.tok_emb.clone())];
        for (index, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{index}");
            for (name, tensor) in [
                ("ln1_g", &layer.ln1_g),
                ("ln1_b", &layer.ln1_b),
                ("wq", &layer.wq),
                ("bq", &layer.bq),
                ("wk", &layer.wk),
                ("bk", &layer.bk),
                ("wv", &layer.wv),
                ("bv", &layer.bv),
                ("wo", &layer.wo),
                ("bo", &layer.bo),
                ("ln2_g", &layer.ln2_g),
                ("ln2_b", &layer.ln2_b),
                ("w1", &layer.w1),
                ("b1", &layer.b1),
                ("w2", &layer.w2),
                ("b2", &layer.b2),
            ] {
                out.push((format!("{prefix}.{name}"), tensor.clone()));
            }
        }
        out.push(("ln_f_g".to_string(), self.ln_f_g.clone()));
        out.push(("ln_f_b".to_string(), self.ln_f_b.clone()));
        out.push(("lm_head".to_string(), self.lm_head.clone()));
        out
    }

    pub fn forward(
        &self,
        ctx: &mut RunContext<'_, B>,
        handles: &DecoderHandles,
        tokens: &[u32],
    ) -> Result<Tensor<B>> {
        let config = &self.config;
        let seq = tokens.len();
        if seq == 0 {
            return Err(Error::invalid("decoder step requires at least one token"));
        }
        let position = ctx.scalar(handles.kv.position)?.as_i64()? as usize;
        if position + seq > config.max_seq {
            return Err(Error::invalid(format!(
                "sequence length {} exceeds max_seq {}",
                position + seq,
                config.max_seq
            )));
        }
        if let Some(&token) = tokens.iter().find(|&&token| token as usize >= config.vocab) {
            return Err(Error::invalid(format!(
                "token {token} out of range for vocab {}",
                config.vocab
            )));
        }

        let backend = self.tok_emb.backend().clone();
        let ops = backend.ops();
        let ids: Vec<i64> = tokens.iter().map(|&token| token as i64).collect();
        let ids = Tensor::from_slice(backend.clone(), &ids, [seq], DType::I64)?;
        let mut x = ops.embedding(&ids, &self.tok_emb)?;
        let heads = config.n_heads;
        let head_dim = config.head_dim();
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut pending = Vec::with_capacity(self.layers.len());

        for (index, layer) in self.layers.iter().enumerate() {
            let norm = ops.layer_norm(&x, &layer.ln1_g, &layer.ln1_b, config.eps)?;
            let q = ops.linear(&norm, &layer.wq, Some(&layer.bq))?;
            let k_new = ops.linear(&norm, &layer.wk, Some(&layer.bk))?;
            let v_new = ops.linear(&norm, &layer.wv, Some(&layer.bv))?;
            let q = ops.split_heads(&q, heads, head_dim)?;
            let k_new = ops.split_heads(&k_new, heads, head_dim)?;
            let v_new = ops.split_heads(&v_new, heads, head_dim)?;

            let k_old = ctx.tensor(handles.kv.layers[index].k)?;
            let v_old = ctx.tensor(handles.kv.layers[index].v)?;
            let k_all = ops.concat_seq(&k_old, &k_new)?;
            let v_all = ops.concat_seq(&v_old, &v_new)?;
            let attn = ops.attention(&q, &k_all, &v_all, scale, true)?;
            let attn = ops.merge_heads(&attn, heads)?;
            let attn = ops.linear(&attn, &layer.wo, Some(&layer.bo))?;
            x = ops.add(&x, &attn)?;

            let norm = ops.layer_norm(&x, &layer.ln2_g, &layer.ln2_b, config.eps)?;
            let hidden =
                ops.linear_activation(&norm, &layer.w1, Some(&layer.b1), Activation::Gelu)?;
            let hidden = ops.linear(&hidden, &layer.w2, Some(&layer.b2))?;
            x = ops.add(&x, &hidden)?;

            pending.push((k_all, v_all));
        }

        let x = ops.layer_norm(&x, &self.ln_f_g, &self.ln_f_b, config.eps)?;
        let logits = ops.linear(&x, &self.lm_head, None)?;
        let last = ops.last_row(&logits)?;

        for (index, (k, v)) in pending.into_iter().enumerate() {
            ctx.set_tensor(handles.kv.layers[index].k, k)?;
            ctx.set_tensor(handles.kv.layers[index].v, v)?;
        }
        ctx.set_scalar(
            handles.kv.position,
            ScalarValue::I64((position + seq) as i64),
        )?;
        Ok(last)
    }
}

pub(crate) fn select_batch_row<B: Backend>(x: &Tensor<B>, index: usize) -> Result<Tensor<B>> {
    let dims = x.shape().dims();
    if dims.is_empty() || index >= dims[0] {
        return Err(Error::invalid(format!(
            "batch index {index} out of range for {:?}",
            dims
        )));
    }
    let row_numel = x.numel() / dims[0];
    let element = x.dtype().size_in_bytes();
    let bytes = x.to_bytes()?;
    let start = index * row_numel * element;
    let end = start + row_numel * element;
    Tensor::from_bytes(
        x.backend().clone(),
        &bytes[start..end],
        Shape::new(&dims[1..]),
        x.dtype(),
    )
}

#[allow(clippy::type_complexity)]
fn stack_paged_layer<B: Backend>(
    caches: &[Vec<PagedKvCache<B>>],
    layer: usize,
) -> Result<(Tensor<B>, Tensor<B>, Tensor<B>, Tensor<B>)> {
    let first = caches
        .first()
        .and_then(|cache| cache.get(layer))
        .ok_or_else(|| Error::invalid("no paged caches for batch"))?;
    let num_blocks = first.num_blocks();
    let block_size = first.block_size();
    let dims = first.k_blocks().shape().dims();
    let heads = dims[2];
    let head_dim = dims[3];
    let mut k_bytes = Vec::new();
    let mut v_bytes = Vec::new();
    let mut tables = Vec::with_capacity(caches.len() * num_blocks);
    let mut lens = Vec::with_capacity(caches.len());
    for (batch_index, cache_vec) in caches.iter().enumerate() {
        let cache = cache_vec
            .get(layer)
            .ok_or_else(|| Error::invalid("missing paged cache for layer"))?;
        if cache.num_blocks() != num_blocks || cache.block_size() != block_size {
            return Err(Error::invalid("paged cache capacities differ in a batch"));
        }
        k_bytes.extend_from_slice(&cache.k_blocks().to_bytes()?);
        v_bytes.extend_from_slice(&cache.v_blocks().to_bytes()?);
        let table_bytes = cache.block_table().to_bytes()?;
        let (chunks, _) = table_bytes.as_chunks::<8>();
        for chunk in chunks {
            let physical = i64::from_ne_bytes(*chunk);
            tables.push(if physical >= 0 {
                physical + (batch_index * num_blocks) as i64
            } else {
                physical
            });
        }
        lens.push(cache.len() as i64);
    }
    let backend = first.k_blocks().backend().clone();
    let total_blocks = caches.len() * num_blocks;
    let k_blocks = Tensor::from_bytes(
        backend.clone(),
        &k_bytes,
        Shape::new([total_blocks, block_size, heads, head_dim]),
        DType::F32,
    )?;
    let v_blocks = Tensor::from_bytes(
        backend.clone(),
        &v_bytes,
        Shape::new([total_blocks, block_size, heads, head_dim]),
        DType::F32,
    )?;
    let block_tables = Tensor::from_slice(
        backend.clone(),
        &tables,
        Shape::new([caches.len(), num_blocks]),
        DType::I64,
    )?;
    let seq_lens = Tensor::from_slice(backend, &lens, Shape::new([caches.len()]), DType::I64)?;
    Ok((k_blocks, v_blocks, block_tables, seq_lens))
}

impl<B: Backend> Decoder<B> {
    /// Batched decode step (q_seq = 1) over per-request paged KV caches.
    ///
    /// This is the real batched path: every request appends one K/V token and a
    /// single paged-attention kernel runs over the whole batch.  Prefill is
    /// handled by the dense `Run` path and converted into paged caches by the
    /// caller.
    pub fn forward_decode_batch(
        &self,
        caches: &mut [Vec<PagedKvCache<B>>],
        tokens: &[u32],
    ) -> Result<Tensor<B>> {
        if tokens.is_empty() {
            return Err(Error::invalid("batched decode requires at least one token"));
        }
        if caches.len() != tokens.len() {
            return Err(Error::invalid("cache/token batch size mismatch"));
        }
        let config = &self.config;
        if let Some(&token) = tokens.iter().find(|&&token| token as usize >= config.vocab) {
            return Err(Error::invalid(format!(
                "token {token} out of range for vocab {}",
                config.vocab
            )));
        }
        let backend = self.tok_emb.backend().clone();
        let ops = backend.ops();
        let batch = tokens.len();
        let ids: Vec<i64> = tokens.iter().map(|&token| token as i64).collect();
        let ids = Tensor::from_slice(backend.clone(), &ids, [batch], DType::I64)?;
        let mut x = ops.embedding(&ids, &self.tok_emb)?;
        let heads = config.n_heads;
        let head_dim = config.head_dim();
        let scale = 1.0 / (head_dim as f64).sqrt();

        for (layer_index, layer) in self.layers.iter().enumerate() {
            let norm = ops.layer_norm(&x, &layer.ln1_g, &layer.ln1_b, config.eps)?;
            let q = ops.linear(&norm, &layer.wq, Some(&layer.bq))?;
            let k_new = ops.linear(&norm, &layer.wk, Some(&layer.bk))?;
            let v_new = ops.linear(&norm, &layer.wv, Some(&layer.bv))?;
            let q = q.reshape([batch, heads, 1, head_dim])?;
            let k_new = k_new.reshape([batch, heads, 1, head_dim])?;
            let v_new = v_new.reshape([batch, heads, 1, head_dim])?;

            for (batch_index, cache_vec) in caches.iter_mut().enumerate() {
                let k_row = select_batch_row(&k_new, batch_index)?;
                let v_row = select_batch_row(&v_new, batch_index)?;
                let cache = cache_vec
                    .get_mut(layer_index)
                    .ok_or_else(|| Error::invalid("missing paged cache for layer"))?;
                cache.append(&k_row, &v_row)?;
            }

            let (k_blocks, v_blocks, block_tables, seq_lens) =
                stack_paged_layer(caches, layer_index)?;
            let attn = ops.paged_attention_batch(
                &q,
                &k_blocks,
                &v_blocks,
                &block_tables,
                &seq_lens,
                scale,
                true,
            )?;
            let attn = attn.reshape([batch, config.d_model])?;
            let attn = ops.linear(&attn, &layer.wo, Some(&layer.bo))?;
            x = ops.add(&x, &attn)?;

            let norm = ops.layer_norm(&x, &layer.ln2_g, &layer.ln2_b, config.eps)?;
            let hidden =
                ops.linear_activation(&norm, &layer.w1, Some(&layer.b1), Activation::Gelu)?;
            let hidden = ops.linear(&hidden, &layer.w2, Some(&layer.b2))?;
            x = ops.add(&x, &hidden)?;
        }

        let x = ops.layer_norm(&x, &self.ln_f_g, &self.ln_f_b, config.eps)?;
        ops.linear(&x, &self.lm_head, None)
    }
}

impl<B: Backend> Step<B> for Decoder<B> {
    type Input = Vec<u32>;
    type Output = Tensor<B>;
    type Handles = DecoderHandles;

    fn declare_state(&self, builder: &mut StateBuilder<'_, B>) -> Result<Self::Handles> {
        Ok(DecoderHandles {
            kv: declare_kv_cache(
                builder,
                self.config.n_layers,
                self.config.n_heads,
                self.config.head_dim(),
            )?,
        })
    }

    fn step(
        &self,
        ctx: &mut RunContext<'_, B>,
        handles: &Self::Handles,
        tokens: Self::Input,
    ) -> Result<Self::Output> {
        self.forward(ctx, handles, &tokens)
    }
}

/// Autoregressive greedy/temperature generation over an explicit `Run`.
pub fn generate<B: Backend>(
    run: &mut Run<B, Decoder<B>>,
    prompt: &[u32],
    max_new_tokens: usize,
    sampler: Sampler,
) -> Result<Vec<u32>> {
    if prompt.is_empty() {
        return Err(Error::invalid("prompt must not be empty"));
    }
    let mut logits = run.step(prompt.to_vec())?;
    let mut output = Vec::with_capacity(max_new_tokens);
    for _ in 0..max_new_tokens {
        let token = sample_token(&logits, &sampler, run.rng_mut())?;
        output.push(token);
        logits = run.step(vec![token])?;
    }
    Ok(output)
}

/// Convenience shape helper used by tests and callers.
pub fn kv_shape(config: &DecoderConfig, seq: usize) -> Shape {
    Shape::new([config.n_heads, seq, config.head_dim()])
}
