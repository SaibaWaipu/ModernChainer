use modern_chainer_core::{Backend, Ops, Shape, Tensor};
use modern_chainer_cpu::{CpuBackend, CpuTensorExt};

fn values(tensor: &Tensor<CpuBackend>) -> Vec<f32> {
    tensor.as_slice::<f32>().unwrap().to_vec()
}

#[allow(clippy::too_many_arguments)]
fn fill_blocks(
    backend: CpuBackend,
    dense: &[f32],
    seq_len: usize,
    block_size: usize,
    num_blocks: usize,
    table: &[i64],
    heads: usize,
    head_dim: usize,
) -> Tensor<CpuBackend> {
    let mut blocks = vec![0f32; num_blocks * block_size * heads * head_dim];
    for (logical, &physical) in table.iter().enumerate() {
        let physical = physical as usize;
        for within in 0..block_size {
            let position = logical * block_size + within;
            if position >= seq_len {
                continue;
            }
            for head in 0..heads {
                for d in 0..head_dim {
                    let src = (head * seq_len + position) * head_dim + d;
                    let dst = ((physical * block_size + within) * heads + head) * head_dim + d;
                    blocks[dst] = dense[src];
                }
            }
        }
    }
    Tensor::from_f32(backend, &blocks, [num_blocks, block_size, heads, head_dim]).unwrap()
}

#[test]
fn paged_attention_matches_dense_prefill_and_decode() {
    let backend = CpuBackend::new();
    let heads = 1;
    let seq = 5;
    let head_dim = 2;
    let block_size = 2;
    let num_blocks = 3;
    let table = [1i64, 0, 2];

    let k = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let q_full = vec![1.0, 0.5, 0.2, 0.4, 0.3, 0.1, 0.6, 0.2, 0.1, 0.9];
    let q_last = q_full[8..10].to_vec();

    let k_dense = Tensor::from_f32(backend, &k, [heads, seq, head_dim]).unwrap();
    let v_dense = Tensor::from_f32(backend, &v, [heads, seq, head_dim]).unwrap();
    let q_dense = Tensor::from_f32(backend, &q_full, [heads, seq, head_dim]).unwrap();
    let q_last_dense = Tensor::from_f32(backend, &q_last, [heads, 1, head_dim]).unwrap();
    let k_blocks = fill_blocks(
        backend, &k, seq, block_size, num_blocks, &table, heads, head_dim,
    );
    let v_blocks = fill_blocks(
        backend, &v, seq, block_size, num_blocks, &table, heads, head_dim,
    );
    let block_table = Tensor::from_slice(
        backend,
        &table,
        [num_blocks],
        modern_chainer_core::DType::I64,
    )
    .unwrap();

    let scale = 1.0 / (head_dim as f64).sqrt();
    let dense_prefill = backend
        .ops()
        .attention(&q_dense, &k_dense, &v_dense, scale, true)
        .unwrap();
    let paged_prefill = backend
        .ops()
        .paged_attention(
            &q_dense,
            &k_blocks,
            &v_blocks,
            &block_table,
            seq,
            scale,
            true,
        )
        .unwrap();
    assert_eq!(paged_prefill.shape(), &Shape::new([heads, seq, head_dim]));
    for (a, b) in values(&dense_prefill)
        .iter()
        .zip(values(&paged_prefill).iter())
    {
        assert!((a - b).abs() < 1e-5, "prefill mismatch: {a} vs {b}");
    }

    let dense_decode = backend
        .ops()
        .attention(&q_last_dense, &k_dense, &v_dense, scale, true)
        .unwrap();
    let paged_decode = backend
        .ops()
        .paged_attention(
            &q_last_dense,
            &k_blocks,
            &v_blocks,
            &block_table,
            seq,
            scale,
            true,
        )
        .unwrap();
    for (a, b) in values(&dense_decode)
        .iter()
        .zip(values(&paged_decode).iter())
    {
        assert!((a - b).abs() < 1e-5, "decode mismatch: {a} vs {b}");
    }
}

#[test]
fn gelu_matches_formula() {
    use modern_chainer_core::Backend;
    let backend = CpuBackend::new();
    let x = Tensor::from_f32(backend, &[-1.0, 0.0, 0.5, 2.0], [4]).unwrap();
    let y = backend.ops().gelu(&x).unwrap();
    let values = y.as_slice::<f32>().unwrap();
    for (input, output) in [-1.0f32, 0.0, 0.5, 2.0].iter().zip(values) {
        let expected = 0.5
            * input
            * (1.0
                + ((2.0f32 / std::f32::consts::PI).sqrt() * (input + 0.044715 * input.powi(3)))
                    .tanh());
        assert!((output - expected).abs() < 1e-5);
    }
}
