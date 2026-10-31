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

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<()> {
    let batch = env_usize("BATCH", 8);
    let prefill_len = env_usize("PREFILL_LEN", 16);
    let decode_tokens = env_usize("DECODE_TOKENS", 32);
    let max_seq = env_usize("MAX_SEQ", (prefill_len + decode_tokens + 16).max(128));
    let variable = std::env::var("VARIABLE").as_deref() == Ok("1");
    let d_model = env_usize("D_MODEL", 64);
    let n_heads = env_usize("N_HEADS", 4);
    let d_ff = env_usize("D_FF", 128);
    let flash_prefill = std::env::var("FLASH_PREFILL").as_deref() != Ok("0");
    let kv_dtype = match std::env::var("KV_DTYPE").as_deref() {
        Ok("f16") => KvDType::F16,
        Ok("bf16") => KvDType::Bf16,
        _ => KvDType::F32,
    };
    let config = DecoderConfig {
        vocab: 256,
        d_model,
        n_layers: 4,
        n_heads,
        d_ff,
        max_seq,
        eps: 1e-5,
    };

    let backend = CudaBackend::new(0)?;
    let decoder = Decoder::new(backend, config.clone(), 42)?;
    let mut model = BatchedStaticDecoder::new_with_options(
        decoder,
        batch,
        prefill_len,
        kv_dtype,
        flash_prefill,
    )?;

    let lengths: Vec<usize> = (0..batch)
        .map(|b| {
            if variable {
                (prefill_len - b * (prefill_len / (batch * 2))).max(1)
            } else {
                prefill_len
            }
        })
        .collect();
    let prompts: Vec<Vec<u32>> = lengths
        .iter()
        .enumerate()
        .map(|(b, &len)| {
            (0..len)
                .map(|i| ((b * prefill_len + i) * 37 + 11) as u32 % config.vocab as u32)
                .collect()
        })
        .collect();
    let prompt_refs: Vec<&[u32]> = prompts.iter().map(Vec::as_slice).collect();

    for _ in 0..3 {
        let logits = model.prefill_batch(&prompt_refs)?;
        let tokens: Vec<u32> = logits.iter().map(|row| argmax(row)).collect();
        model.decode(&tokens)?;
    }

    let mut ttft_samples = Vec::new();
    for _ in 0..20 {
        let start = Instant::now();
        model.prefill_batch(&prompt_refs)?;
        ttft_samples.push(start.elapsed().as_secs_f64());
    }
    ttft_samples.sort_by(|a, b| a.total_cmp(b));
    let ttft = ttft_samples[ttft_samples.len() / 2];

    let decode_runs = 5;
    let mut tokens: Vec<u32>;
    let mut decode_samples = Vec::new();
    for _ in 0..decode_runs {
        model.prefill_batch(&prompt_refs)?;
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

    let prefill_tokens: usize = lengths.iter().sum();
    let decode_ms = decode_time / decode_tokens as f64 * 1e3;
    println!("batch: {batch}");
    println!("prefill_len: {prefill_len}");
    println!("variable: {variable}");
    println!("max_seq: {}", config.max_seq);
    println!(
        "kv dtype: {kv_dtype:?} ({} MiB cache), flash prefill: {}",
        model.kv_bytes() / (1024 * 1024),
        model.flash_prefill()
    );
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
