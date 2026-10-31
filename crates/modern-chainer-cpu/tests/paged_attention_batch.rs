use modern_chainer_core::{Backend, DType, Ops, Tensor};
use modern_chainer_cpu::{CpuBackend, CpuTensorExt};

#[allow(clippy::too_many_arguments)]
fn write_positions(
    blocks: &mut [f32],
    table: &[i64],
    block_size: usize,
    heads: usize,
    head_dim: usize,
    seq_len: usize,
    dense: &[f32],
    num_blocks: usize,
) {
    for position in 0..seq_len {
        let logical = position / block_size;
        let physical = table[logical] as usize;
        let within = position % block_size;
        for head in 0..heads {
            for d in 0..head_dim {
                let src = (head * seq_len + position) * head_dim + d;
                let dst = ((physical * block_size + within) * heads + head) * head_dim + d;
                assert!(physical < num_blocks);
                blocks[dst] = dense[src];
            }
        }
    }
}

#[test]
fn batched_paged_attention_matches_dense_per_request() {
    let backend = CpuBackend::new();
    let heads = 1;
    let head_dim = 2;
    let block_size = 2;
    let num_blocks = 6;
    let seq_lens = [3usize, 5];
    let max_blocks = 3;
    let tables = [vec![0i64, 1, -1], vec![2i64, 3, 4]];

    let k0 = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let v0 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let k1 = vec![0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.8, 0.7, 1.0, 0.9];
    let v1 = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0, 10.0, 9.0];

    let mut k_blocks = vec![0f32; num_blocks * block_size * heads * head_dim];
    let mut v_blocks = vec![0f32; num_blocks * block_size * heads * head_dim];
    write_positions(
        &mut k_blocks,
        &tables[0],
        block_size,
        heads,
        head_dim,
        seq_lens[0],
        &k0,
        num_blocks,
    );
    write_positions(
        &mut v_blocks,
        &tables[0],
        block_size,
        heads,
        head_dim,
        seq_lens[0],
        &v0,
        num_blocks,
    );
    write_positions(
        &mut k_blocks,
        &tables[1],
        block_size,
        heads,
        head_dim,
        seq_lens[1],
        &k1,
        num_blocks,
    );
    write_positions(
        &mut v_blocks,
        &tables[1],
        block_size,
        heads,
        head_dim,
        seq_lens[1],
        &v1,
        num_blocks,
    );

    let k_blocks_t = Tensor::from_f32(
        backend,
        &k_blocks,
        [num_blocks, block_size, heads, head_dim],
    )
    .unwrap();
    let v_blocks_t = Tensor::from_f32(
        backend,
        &v_blocks,
        [num_blocks, block_size, heads, head_dim],
    )
    .unwrap();
    let mut table_values = Vec::new();
    for table in &tables {
        table_values.extend_from_slice(table);
    }
    let block_tables =
        Tensor::from_slice(backend, &table_values, [2, max_blocks], DType::I64).unwrap();
    let lens = Tensor::from_slice(
        backend,
        &[seq_lens[0] as i64, seq_lens[1] as i64],
        [2],
        DType::I64,
    )
    .unwrap();

    let q0 = vec![0.5, -0.5, 0.2, 0.1];
    let q1 = vec![0.3, 0.8, -0.2, 0.4];
    let mut q_values = q0.clone();
    q_values.extend_from_slice(&q1);
    let q = Tensor::from_f32(backend, &q_values, [2, heads, 2, head_dim]).unwrap();
    let scale = 1.0 / (head_dim as f64).sqrt();

    let batched = backend
        .ops()
        .paged_attention_batch(
            &q,
            &k_blocks_t,
            &v_blocks_t,
            &block_tables,
            &lens,
            scale,
            true,
        )
        .unwrap();
    let batched_values = batched.as_slice::<f32>().unwrap();

    for (request, (k, v, qv, seq_len)) in [
        (k0.as_slice(), v0.as_slice(), q0.as_slice(), seq_lens[0]),
        (k1.as_slice(), v1.as_slice(), q1.as_slice(), seq_lens[1]),
    ]
    .into_iter()
    .enumerate()
    {
        let dense_k = Tensor::from_f32(backend, k, [heads, seq_len, head_dim]).unwrap();
        let dense_v = Tensor::from_f32(backend, v, [heads, seq_len, head_dim]).unwrap();
        let dense_q = Tensor::from_f32(backend, qv, [heads, 2, head_dim]).unwrap();
        let expected = backend
            .ops()
            .attention(&dense_q, &dense_k, &dense_v, scale, true)
            .unwrap();
        let expected_values = expected.as_slice::<f32>().unwrap();
        let offset = request * expected_values.len();
        for (index, (a, b)) in batched_values[offset..offset + expected_values.len()]
            .iter()
            .zip(expected_values)
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-5,
                "request {request} element {index}: {a} vs {b}"
            );
        }
    }
}
