use std::time::Instant;

use modern_chainer::prelude::*;
use modern_chainer_llm::DecoderConfig;

fn argmax(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index as u32)
        .unwrap_or(0)
}

fn main() -> Result<()> {
    let config = DecoderConfig {
        vocab: 256,
        d_model: 64,
        n_layers: 4,
        n_heads: 4,
        d_ff: 128,
        max_seq: 128,
        eps: 1e-5,
    };
    let batch = 8;
    let prefill_len = 16;
    let decode_tokens = 32;

    let backend = CudaBackend::new(0)?;
    let decoder = Decoder::new(backend, config.clone(), 42)?;
    let mut model = BatchedStaticDecoder::new(decoder, batch, prefill_len)?;

    let prompts: Vec<u32> = (0..batch * prefill_len)
        .map(|i| ((i * 37 + 11) % config.vocab) as u32)
        .collect();

    for _ in 0..3 {
        let logits = model.prefill(&prompts)?;
        let tokens: Vec<u32> = logits.iter().map(|row| argmax(row)).collect();
        model.decode(&tokens)?;
    }

    let mut ttft_samples = Vec::new();
    for _ in 0..20 {
        let start = Instant::now();
        model.prefill(&prompts)?;
        ttft_samples.push(start.elapsed().as_secs_f64());
    }
    ttft_samples.sort_by(|a, b| a.total_cmp(b));
    let ttft = ttft_samples[ttft_samples.len() / 2];

    let decode_runs = 5;
    let mut tokens: Vec<u32>;
    let mut decode_samples = Vec::new();
    for _ in 0..decode_runs {
        model.prefill(&prompts)?;
        tokens = model
            .last_prefill_logits()?
            .iter()
            .map(|row| argmax(row))
            .collect();
        let start = Instant::now();
        for _ in 0..decode_tokens {
            let logits = model.decode(&tokens)?;
            tokens = logits.iter().map(|row| argmax(row)).collect();
        }
        decode_samples.push(start.elapsed().as_secs_f64());
    }
    decode_samples.sort_by(|a, b| a.total_cmp(b));
    let decode_time = decode_samples[decode_samples.len() / 2];
    let generated = batch * decode_tokens;

    let prefill_tokens = batch * prefill_len;
    let decode_ms = decode_time / decode_tokens as f64 * 1e3;
    println!("batch: {batch}");
    println!("prefill_len: {prefill_len}");
    println!("max_seq: {}", config.max_seq);
    println!("prefill tokens: {prefill_tokens}");
    println!("TTFT (batched prefill): {:.3} ms", ttft * 1e3);
    println!(
        "prefill throughput: {:.1} tok/s",
        prefill_tokens as f64 / ttft
    );
    println!(
        "decode throughput: {:.1} tok/s ({:.3} ms per batched step)",
        generated as f64 / decode_time,
        decode_ms
    );
    println!(
        "per-sequence decode: {:.1} tok/s",
        generated as f64 / decode_time / batch as f64
    );
    println!("final position: {}", model.position()?);
    Ok(())
}
