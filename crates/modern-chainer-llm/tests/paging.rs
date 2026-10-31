use modern_chainer_core::{Backend, Ops, Tensor};
use modern_chainer_cpu::CpuBackend;
use modern_chainer_llm::{BlockAllocator, PagedKvCache};

fn values(tensor: &Tensor<CpuBackend>) -> Vec<f32> {
    let bytes = tensor.to_bytes().unwrap();
    let (chunks, _) = bytes.as_chunks::<4>();
    chunks
        .iter()
        .map(|chunk| f32::from_ne_bytes(*chunk))
        .collect()
}

fn assert_close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (index, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() < 1e-5, "element {index}: {x} vs {y}");
    }
}

#[test]
fn allocator_reuses_released_blocks() {
    let mut allocator = BlockAllocator::new(2, 4).unwrap();
    let a = allocator.allocate().unwrap();
    let b = allocator.allocate().unwrap();
    assert_eq!(allocator.free_blocks(), 0);
    assert!(allocator.allocate().is_err());
    allocator.retain(a).unwrap();
    assert_eq!(allocator.refcount(a), Some(2));
    allocator.release(a).unwrap();
    allocator.release(a).unwrap();
    assert_eq!(allocator.free_blocks(), 1);
    let reused = allocator.allocate().unwrap();
    assert_eq!(reused, a);
    allocator.release(b).unwrap();
    allocator.release(reused).unwrap();
    assert_eq!(allocator.free_blocks(), 2);
}

#[test]
fn paged_cache_matches_dense_and_supports_append_truncate() {
    let backend = CpuBackend::new();
    let heads = 1;
    let head_dim = 2;
    let seq = 3;
    let k = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let q = vec![0.5, -0.5, 0.2, 0.1, -0.3, 0.7];
    let scale = 1.0 / (head_dim as f64).sqrt();

    let k_dense = Tensor::from_f32(backend, &k, [heads, seq, head_dim]).unwrap();
    let v_dense = Tensor::from_f32(backend, &v, [heads, seq, head_dim]).unwrap();
    let q_dense = Tensor::from_f32(backend, &q, [heads, seq, head_dim]).unwrap();
    let mut cache =
        PagedKvCache::from_dense_with_capacity(backend, &k_dense, &v_dense, 2, 4).unwrap();
    assert_eq!(cache.len(), seq);
    assert_eq!(cache.block_table().shape().dims(), &[4]);

    let dense = backend
        .ops()
        .attention(&q_dense, &k_dense, &v_dense, scale, true)
        .unwrap();
    let paged = backend
        .ops()
        .paged_attention(
            &q_dense,
            cache.k_blocks(),
            cache.v_blocks(),
            cache.block_table(),
            cache.len(),
            scale,
            true,
        )
        .unwrap();
    assert_close(&values(&dense), &values(&paged));

    let k_new = Tensor::from_f32(backend, &[0.7, 0.8], [heads, 1, head_dim]).unwrap();
    let v_new = Tensor::from_f32(backend, &[7.0, 8.0], [heads, 1, head_dim]).unwrap();
    cache.append(&k_new, &v_new).unwrap();
    assert_eq!(cache.len(), seq + 1);

    let k_all = Tensor::from_f32(
        backend,
        &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [heads, 4, head_dim],
    )
    .unwrap();
    let v_all = Tensor::from_f32(
        backend,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [heads, 4, head_dim],
    )
    .unwrap();
    let q_all = Tensor::from_f32(
        backend,
        &[0.5, -0.5, 0.2, 0.1, -0.3, 0.7, 0.4, 0.2],
        [heads, 4, head_dim],
    )
    .unwrap();
    let dense_all = backend
        .ops()
        .attention(&q_all, &k_all, &v_all, scale, true)
        .unwrap();
    let paged_all = backend
        .ops()
        .paged_attention(
            &q_all,
            cache.k_blocks(),
            cache.v_blocks(),
            cache.block_table(),
            cache.len(),
            scale,
            true,
        )
        .unwrap();
    assert_close(&values(&dense_all), &values(&paged_all));

    cache.truncate(2).unwrap();
    assert_eq!(cache.len(), 2);
    cache.append(&k_new, &v_new).unwrap();
    assert_eq!(cache.len(), 3);
}

#[test]
fn fork_isolates_paged_writes() {
    let backend = CpuBackend::new();
    let k = Tensor::from_f32(backend, &[0.1, 0.2, 0.3, 0.4], [1, 2, 2]).unwrap();
    let v = Tensor::from_f32(backend, &[1.0, 2.0, 3.0, 4.0], [1, 2, 2]).unwrap();
    let parent = PagedKvCache::from_dense_with_capacity(backend, &k, &v, 2, 3).unwrap();
    let mut child = parent.fork();
    let k_new = Tensor::from_f32(backend, &[0.5, 0.6], [1, 1, 2]).unwrap();
    let v_new = Tensor::from_f32(backend, &[5.0, 6.0], [1, 1, 2]).unwrap();
    child.append(&k_new, &v_new).unwrap();
    assert_eq!(parent.len(), 2);
    assert_eq!(child.len(), 3);
    assert_eq!(parent.k_blocks().shape().dims(), &[3, 2, 1, 2]);
    assert_eq!(child.k_blocks().shape().dims(), &[3, 2, 1, 2]);
}
