use modern_chainer_core::{Backend, DType, Ops, Shape, Tensor};
use modern_chainer_cpu::{CpuBackend, CpuTensorExt};
use modern_chainer_cuda::{CudaBackend, is_available, probe};
use serial_test::serial;

fn values<B: Backend>(tensor: &Tensor<B>) -> Vec<f32> {
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
        assert!((x - y).abs() < 1e-4, "element {index}: {x} vs {y}");
    }
}

#[test]
#[serial]
fn cuda_core_ops_match_cpu() {
    let capability = probe();
    eprintln!("cuda capability: {capability:?}");
    if !is_available() {
        eprintln!("skipping: CUDA is not available");
        return;
    }
    let backend = CudaBackend::new(0).expect("cuda backend");
    eprintln!("cuda device: {}", backend.name());
    let cpu = CpuBackend::new();

    let a_data = [0.5, -1.0, 2.0, 3.0, -0.5, 0.25];
    let b_data = [1.5, 0.5, -2.0, 0.25, 4.0, -1.0];
    let a = Tensor::from_f32(backend.clone(), &a_data, [2, 3]).unwrap();
    let b = Tensor::from_f32(backend.clone(), &b_data, [2, 3]).unwrap();
    let ca = Tensor::from_f32(cpu, &a_data, [2, 3]).unwrap();
    let cb = Tensor::from_f32(cpu, &b_data, [2, 3]).unwrap();

    assert_close(
        &values(&backend.ops().add(&a, &b).unwrap()),
        &values(&cpu.ops().add(&ca, &cb).unwrap()),
    );
    assert_close(
        &values(&backend.ops().mul(&a, &b).unwrap()),
        &values(&cpu.ops().mul(&ca, &cb).unwrap()),
    );
    assert_close(
        &values(&backend.ops().relu(&a).unwrap()),
        &values(&cpu.ops().relu(&ca).unwrap()),
    );
    assert_close(
        &values(&backend.ops().scale(&a, -2.5).unwrap()),
        &values(&cpu.ops().scale(&ca, -2.5).unwrap()),
    );

    let m_data: Vec<f32> = (0..35).map(|i| i as f32 * 0.1 - 1.0).collect();
    let n_data: Vec<f32> = (0..21).map(|i| 1.0 - i as f32 * 0.07).collect();
    let m = Tensor::from_f32(backend.clone(), &m_data, [5, 7]).unwrap();
    let n = Tensor::from_f32(backend.clone(), &n_data, [7, 3]).unwrap();
    let cm = Tensor::from_f32(cpu, &m_data, [5, 7]).unwrap();
    let cn = Tensor::from_f32(cpu, &n_data, [7, 3]).unwrap();
    assert_close(
        &values(&backend.ops().matmul(&m, &n).unwrap()),
        &values(&cpu.ops().matmul(&cm, &cn).unwrap()),
    );

    let x_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let w_data: Vec<f32> = (0..6).map(|i| (i as f32 * 0.13).sin()).collect();
    let bias_data = [0.1, -0.2, 0.3];
    let x = Tensor::from_f32(backend.clone(), &x_data, [4, 2]).unwrap();
    let w = Tensor::from_f32(backend.clone(), &w_data, [3, 2]).unwrap();
    let bias = Tensor::from_f32(backend.clone(), &bias_data, [3]).unwrap();
    let cx = Tensor::from_f32(cpu, &x_data, [4, 2]).unwrap();
    let cw = Tensor::from_f32(cpu, &w_data, [3, 2]).unwrap();
    let cbias = Tensor::from_f32(cpu, &bias_data, [3]).unwrap();
    assert_close(
        &values(&backend.ops().linear(&x, &w, Some(&bias)).unwrap()),
        &values(&cpu.ops().linear(&cx, &cw, Some(&cbias)).unwrap()),
    );

    let scalar = backend.ops().sum_all(&x).unwrap();
    assert!((backend.ops().to_scalar(&scalar).unwrap() - 36.0).abs() < 1e-4);
    assert_eq!(scalar.shape(), &Shape::scalar());

    let bias_grad = backend.ops().sum_to(&x, &Shape::new([2])).unwrap();
    let cpu_bias_grad = cpu.ops().sum_to(&cx, &Shape::new([2])).unwrap();
    assert_close(&values(&bias_grad), &values(&cpu_bias_grad));

    let target =
        Tensor::from_slice(backend.clone(), &[0i64, 1i64, 2i64, 1i64], [4], DType::I64).unwrap();
    let logits_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.1 - 0.5).collect();
    let logits = Tensor::from_f32(backend.clone(), &logits_data, [4, 3]).unwrap();
    let closs = cpu
        .ops()
        .softmax_cross_entropy(
            &Tensor::from_f32(cpu, &logits_data, [4, 3]).unwrap(),
            &Tensor::from_slice(cpu, &[0i64, 1, 2, 1], [4], DType::I64).unwrap(),
        )
        .unwrap();
    let loss = backend
        .ops()
        .softmax_cross_entropy(&logits, &target)
        .unwrap();
    assert!((backend.ops().to_scalar(&loss).unwrap() - closs.scalar_f64().unwrap()).abs() < 1e-4);
}

#[test]
#[serial]
fn cuda_linear_and_adam_training_step() {
    use modern_chainer_autograd::Variable;
    use modern_chainer_autograd::ops as F;
    use modern_chainer_nn::{Linear, Module};
    use modern_chainer_optim::{Adam, Optimizer, zero_grad};

    if !is_available() {
        eprintln!("skipping: CUDA is not available");
        return;
    }
    let backend = CudaBackend::new(0).unwrap();
    let weight = Variable::new(Tensor::from_f32(backend.clone(), &[0.2, -0.4], [1, 2]).unwrap());
    let bias = Variable::new(Tensor::from_f32(backend.clone(), &[0.1], [1]).unwrap());
    let model = Linear::from_parts(weight, Some(bias), 2, 1).unwrap();
    let x = Variable::new(
        Tensor::from_f32(
            backend.clone(),
            &[1.0, 2.0, 2.0, 1.0, 3.0, 1.0, 4.0, 2.0],
            [4, 2],
        )
        .unwrap(),
    );
    let target = Variable::constant(
        Tensor::from_f32(backend.clone(), &[3.0, 4.0, 5.0, 6.0], [4, 1]).unwrap(),
    );
    let mut optimizer = Adam::new(0.05);
    let mut first = None;
    let mut last = 0.0;
    for _ in 0..40 {
        let prediction = model.forward(&x).unwrap();
        let diff = F::sub(&prediction, &target).unwrap();
        let loss = F::mean(&F::mul(&diff, &diff).unwrap()).unwrap();
        last = loss.data().backend().ops().to_scalar(&loss.data()).unwrap();
        first.get_or_insert(last);
        zero_grad(&model.parameters());
        loss.backward().unwrap();
        optimizer.update(&model.parameters()).unwrap();
    }
    let first = first.unwrap();
    eprintln!("cuda adam loss: {first} -> {last}");
    assert!(last < first);
}

#[test]
#[serial]
fn cuda_llm_kernels_match_cpu() {
    use modern_chainer_llm::PagedKvCache;

    if !is_available() {
        eprintln!("skipping: CUDA is not available");
        return;
    }
    let backend = CudaBackend::new(0).unwrap();
    let cpu = CpuBackend::new();
    let scale = std::f64::consts::FRAC_1_SQRT_2;

    let ids_data = [1i64, 3i64];
    let weight_data: Vec<f32> = (0..8).map(|i| i as f32 * 0.11).collect();
    let ids = Tensor::from_slice(backend.clone(), &ids_data, [2], DType::I64).unwrap();
    let weight = Tensor::from_f32(backend.clone(), &weight_data, [4, 2]).unwrap();
    assert_close(
        &values(&backend.ops().embedding(&ids, &weight).unwrap()),
        &values(
            &cpu.ops()
                .embedding(
                    &Tensor::from_slice(cpu, &ids_data, [2], DType::I64).unwrap(),
                    &Tensor::from_f32(cpu, &weight_data, [4, 2]).unwrap(),
                )
                .unwrap(),
        ),
    );

    let x_data: Vec<f32> = (0..6).map(|i| (i as f32 * 0.3).sin()).collect();
    let gamma_data = [1.0f32, 0.5, 1.5];
    let beta_data = [0.0f32, 0.1, -0.2];
    let x = Tensor::from_f32(backend.clone(), &x_data, [2, 3]).unwrap();
    let gamma = Tensor::from_f32(backend.clone(), &gamma_data, [3]).unwrap();
    let beta = Tensor::from_f32(backend.clone(), &beta_data, [3]).unwrap();
    assert_close(
        &values(&backend.ops().layer_norm(&x, &gamma, &beta, 1e-5).unwrap()),
        &values(
            &cpu.ops()
                .layer_norm(
                    &Tensor::from_f32(cpu, &x_data, [2, 3]).unwrap(),
                    &Tensor::from_f32(cpu, &gamma_data, [3]).unwrap(),
                    &Tensor::from_f32(cpu, &beta_data, [3]).unwrap(),
                    1e-5,
                )
                .unwrap(),
        ),
    );

    let softmax_input: Vec<f32> = (0..6).map(|i| i as f32 * 0.4 - 1.0).collect();
    assert_close(
        &values(
            &backend
                .ops()
                .softmax_last_dim(
                    &Tensor::from_f32(backend.clone(), &softmax_input, [2, 3]).unwrap(),
                )
                .unwrap(),
        ),
        &values(
            &cpu.ops()
                .softmax_last_dim(&Tensor::from_f32(cpu, &softmax_input, [2, 3]).unwrap())
                .unwrap(),
        ),
    );

    let heads_data: Vec<f32> = (0..8).map(|i| i as f32 * 0.25).collect();
    let heads_tensor = Tensor::from_f32(backend.clone(), &heads_data, [2, 4]).unwrap();
    let split = backend.ops().split_heads(&heads_tensor, 2, 2).unwrap();
    assert_eq!(split.shape().dims(), &[2, 2, 2]);
    let merged = backend.ops().merge_heads(&split, 2).unwrap();
    assert_close(&values(&merged), &heads_data);
    let _ = cpu;

    let seq_a: Vec<f32> = (0..4).map(|i| i as f32).collect();
    let seq_b: Vec<f32> = (0..2).map(|i| 10.0 + i as f32).collect();
    let a = Tensor::from_f32(backend.clone(), &seq_a, [1, 2, 2]).unwrap();
    let b = Tensor::from_f32(backend.clone(), &seq_b, [1, 1, 2]).unwrap();
    let concatenated = backend.ops().concat_seq(&a, &b).unwrap();
    assert_eq!(concatenated.shape().dims(), &[1, 3, 2]);
    assert_close(&values(&concatenated), &[0.0, 1.0, 2.0, 3.0, 10.0, 11.0]);

    let logits_data: Vec<f32> = (0..9).map(|i| i as f32 * 0.2 - 0.5).collect();
    let logits = Tensor::from_f32(backend.clone(), &logits_data, [3, 3]).unwrap();
    assert_close(
        &values(&backend.ops().last_row(&logits).unwrap()),
        &[0.7, 0.9, 1.1],
    );

    let q_data = [0.5f32, -0.5, 0.2, 0.1];
    let k_data: Vec<f32> = (0..6).map(|i| i as f32 * 0.3 - 0.4).collect();
    let v_data: Vec<f32> = (0..6).map(|i| 1.0 + i as f32 * 0.2).collect();
    let q = Tensor::from_f32(backend.clone(), &q_data, [1, 2, 2]).unwrap();
    let k = Tensor::from_f32(backend.clone(), &k_data, [1, 3, 2]).unwrap();
    let v = Tensor::from_f32(backend.clone(), &v_data, [1, 3, 2]).unwrap();
    assert_close(
        &values(&backend.ops().attention(&q, &k, &v, scale, true).unwrap()),
        &values(
            &cpu.ops()
                .attention(
                    &Tensor::from_f32(cpu, &q_data, [1, 2, 2]).unwrap(),
                    &Tensor::from_f32(cpu, &k_data, [1, 3, 2]).unwrap(),
                    &Tensor::from_f32(cpu, &v_data, [1, 3, 2]).unwrap(),
                    scale,
                    true,
                )
                .unwrap(),
        ),
    );

    let mut cache = PagedKvCache::from_dense_with_capacity(backend.clone(), &k, &v, 2, 4).unwrap();
    let paged = backend
        .ops()
        .paged_attention(
            &q,
            cache.k_blocks(),
            cache.v_blocks(),
            cache.block_table(),
            cache.len(),
            scale,
            true,
        )
        .unwrap();
    let dense = cpu
        .ops()
        .attention(
            &Tensor::from_f32(cpu, &q_data, [1, 2, 2]).unwrap(),
            &Tensor::from_f32(cpu, &k_data, [1, 3, 2]).unwrap(),
            &Tensor::from_f32(cpu, &v_data, [1, 3, 2]).unwrap(),
            scale,
            true,
        )
        .unwrap();
    assert_close(&values(&paged), &values(&dense));

    let k_new = Tensor::from_f32(backend.clone(), &[0.7, 0.8], [1, 1, 2]).unwrap();
    let v_new = Tensor::from_f32(backend.clone(), &[7.0, 8.0], [1, 1, 2]).unwrap();
    cache.append(&k_new, &v_new).unwrap();
    assert_eq!(cache.len(), 4);

    let q_all = Tensor::from_f32(
        backend.clone(),
        &[0.5, -0.5, 0.2, 0.1, -0.3, 0.7, 0.4, 0.2],
        [1, 4, 2],
    )
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
    let q_all_cpu =
        Tensor::from_f32(cpu, &[0.5, -0.5, 0.2, 0.1, -0.3, 0.7, 0.4, 0.2], [1, 4, 2]).unwrap();
    let k_all_cpu =
        Tensor::from_f32(cpu, &[-0.4, -0.1, 0.2, 0.5, 0.8, 1.1, 0.7, 0.8], [1, 4, 2]).unwrap();
    let v_all_cpu =
        Tensor::from_f32(cpu, &[1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 7.0, 8.0], [1, 4, 2]).unwrap();
    assert_close(
        &values(&paged_all),
        &values(
            &cpu.ops()
                .attention(&q_all_cpu, &k_all_cpu, &v_all_cpu, scale, true)
                .unwrap(),
        ),
    );

    let q_batch = q_all.reshape([1, 1, 4, 2]).unwrap();
    let table_bytes = cache.block_table().to_bytes().unwrap();
    let table_values: Vec<i64> = table_bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| i64::from_ne_bytes(*chunk))
        .collect();
    let tables = Tensor::from_slice(backend.clone(), &table_values, [1, 4], DType::I64).unwrap();
    let lens = Tensor::from_slice(backend.clone(), &[4i64], [1], DType::I64).unwrap();
    let batched = backend
        .ops()
        .paged_attention_batch(
            &q_batch,
            cache.k_blocks(),
            cache.v_blocks(),
            &tables,
            &lens,
            scale,
            true,
        )
        .unwrap();
    assert_close(&values(&batched), &values(&paged_all));
}

#[test]
#[serial]
fn cuda_decoder_matches_cpu() {
    use modern_chainer_llm::{Decoder, DecoderConfig};
    use modern_chainer_run::{Run, RunMode};

    if !is_available() {
        eprintln!("skipping: CUDA is not available");
        return;
    }
    let backend = CudaBackend::new(0).unwrap();
    let config = DecoderConfig {
        vocab: 32,
        d_model: 8,
        n_layers: 2,
        n_heads: 2,
        d_ff: 16,
        max_seq: 24,
        eps: 1e-5,
    };
    let cuda_decoder = Decoder::new(backend.clone(), config.clone(), 7).unwrap();
    let cpu_decoder = Decoder::new(CpuBackend::new(), config, 7).unwrap();
    let mut cuda_run = Run::new(cuda_decoder, backend, RunMode::Inference, 11).unwrap();
    let mut cpu_run = Run::new(cpu_decoder, CpuBackend::new(), RunMode::Inference, 11).unwrap();
    let prompt = vec![1u32, 5, 9, 3];
    let cuda_logits = cuda_run.step(prompt.clone()).unwrap();
    let cpu_logits = cpu_run.step(prompt).unwrap();
    let cuda_values = values(&cuda_logits);
    let cpu_values = values(&cpu_logits);
    for (index, (a, b)) in cuda_values.iter().zip(&cpu_values).enumerate() {
        assert!(
            (a - b).abs() < 2e-3,
            "decoder logit {index}: cuda {a} cpu {b}"
        );
    }
}

#[test]
#[serial]
fn cuda_paged_batch_matches_cpu() {
    use modern_chainer_llm::{Decoder, DecoderConfig, PagedBatchRunner, Sampler};

    if !is_available() {
        eprintln!("skipping: CUDA is not available");
        return;
    }
    let backend = CudaBackend::new(0).unwrap();
    let config = DecoderConfig {
        vocab: 32,
        d_model: 8,
        n_layers: 2,
        n_heads: 2,
        d_ff: 16,
        max_seq: 24,
        eps: 1e-5,
    };
    let cuda_decoder = Decoder::new(backend.clone(), config.clone(), 3).unwrap();
    let cpu_decoder = Decoder::new(CpuBackend::new(), config, 3).unwrap();
    let mut cuda_runner = PagedBatchRunner::new(cuda_decoder, backend, 2, 16, 7).unwrap();
    let mut cpu_runner = PagedBatchRunner::new(cpu_decoder, CpuBackend::new(), 2, 16, 7).unwrap();
    let specs = vec![
        (vec![1u32, 2, 3], 4usize),
        (vec![4u32, 5], 3),
        (vec![6u32], 5),
    ];
    let mut cuda_ids = Vec::new();
    let mut cpu_ids = Vec::new();
    for (prompt, max_new) in &specs {
        cuda_ids.push(
            cuda_runner
                .add_request(prompt.clone(), *max_new, Sampler::Greedy)
                .unwrap(),
        );
        cpu_ids.push(
            cpu_runner
                .add_request(prompt.clone(), *max_new, Sampler::Greedy)
                .unwrap(),
        );
    }
    cuda_runner.run_until_finished().unwrap();
    cpu_runner.run_until_finished().unwrap();
    for (cuda_id, cpu_id) in cuda_ids.iter().zip(cpu_ids.iter()) {
        assert_eq!(
            cuda_runner.generated(*cuda_id).unwrap(),
            cpu_runner.generated(*cpu_id).unwrap()
        );
    }
}

#[test]
#[serial]
fn cuda_paged_attention_batch_matches_cpu_batch() {
    if !is_available() {
        eprintln!("skipping: CUDA is not available");
        return;
    }
    let backend = CudaBackend::new(0).unwrap();
    let cpu = CpuBackend::new();
    let heads = 1;
    let head_dim = 2;
    let block_size = 2;
    let num_blocks = 6;
    let max_blocks = 3;
    let tables = [vec![0i64, 1, -1], vec![2i64, 3, 4]];
    let seq_lens = [3usize, 5];

    let k0 = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
    let v0 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let k1 = vec![0.2f32, 0.1, 0.4, 0.3, 0.6, 0.5, 0.8, 0.7, 1.0, 0.9];
    let v1 = vec![2.0f32, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0, 10.0, 9.0];
    let mut k_blocks = vec![0f32; num_blocks * block_size * heads * head_dim];
    let mut v_blocks = vec![0f32; num_blocks * block_size * heads * head_dim];
    for (request, (k, v, seq_len, table)) in [
        (
            k0.as_slice(),
            v0.as_slice(),
            seq_lens[0],
            tables[0].as_slice(),
        ),
        (
            k1.as_slice(),
            v1.as_slice(),
            seq_lens[1],
            tables[1].as_slice(),
        ),
    ]
    .into_iter()
    .enumerate()
    {
        for position in 0..seq_len {
            let logical = position / block_size;
            let physical = table[logical] as usize;
            let within = position % block_size;
            for d in 0..head_dim {
                let src = position * head_dim + d;
                let dst = ((physical * block_size + within) * heads) * head_dim + d;
                k_blocks[dst] = k[src];
                v_blocks[dst] = v[src];
            }
        }
        let _ = request;
    }

    let mut table_values = Vec::new();
    for table in &tables {
        table_values.extend_from_slice(table);
    }
    let q_values = [0.5f32, -0.5, 0.2, 0.1, 0.3, 0.8, -0.2, 0.4];
    let cuda_q = Tensor::from_f32(backend.clone(), &q_values, [2, 1, 2, 2]).unwrap();
    let cuda_k =
        Tensor::from_f32(backend.clone(), &k_blocks, [num_blocks, block_size, 1, 2]).unwrap();
    let cuda_v =
        Tensor::from_f32(backend.clone(), &v_blocks, [num_blocks, block_size, 1, 2]).unwrap();
    let cuda_tables =
        Tensor::from_slice(backend.clone(), &table_values, [2, max_blocks], DType::I64).unwrap();
    let cuda_lens = Tensor::from_slice(backend.clone(), &[3i64, 5], [2], DType::I64).unwrap();

    let cuda_out = backend
        .ops()
        .paged_attention_batch(
            &cuda_q,
            &cuda_k,
            &cuda_v,
            &cuda_tables,
            &cuda_lens,
            std::f64::consts::FRAC_1_SQRT_2,
            true,
        )
        .unwrap();

    let cpu_q = Tensor::from_f32(cpu, &q_values, [2, 1, 2, 2]).unwrap();
    let cpu_k = Tensor::from_f32(cpu, &k_blocks, [num_blocks, block_size, 1, 2]).unwrap();
    let cpu_v = Tensor::from_f32(cpu, &v_blocks, [num_blocks, block_size, 1, 2]).unwrap();
    let cpu_tables = Tensor::from_slice(cpu, &table_values, [2, max_blocks], DType::I64).unwrap();
    let cpu_lens = Tensor::from_slice(cpu, &[3i64, 5], [2], DType::I64).unwrap();
    let cpu_out = cpu
        .ops()
        .paged_attention_batch(
            &cpu_q,
            &cpu_k,
            &cpu_v,
            &cpu_tables,
            &cpu_lens,
            std::f64::consts::FRAC_1_SQRT_2,
            true,
        )
        .unwrap();
    assert_close(&values(&cuda_out), &values(&cpu_out));
}

#[test]
#[serial]
fn cuda_graph_matmul_and_adam_match_eager() {
    use modern_chainer_core::AdamConfig;

    if !is_available() {
        eprintln!("skipping: CUDA is not available");
        return;
    }
    let backend = CudaBackend::new(0).unwrap();
    let a_data: Vec<f32> = (0..64 * 64).map(|i| (i as f32 * 0.001).sin()).collect();
    let b_data: Vec<f32> = (0..64 * 64).map(|i| (i as f32 * 0.002).cos()).collect();
    let a = Tensor::from_f32(backend.clone(), &a_data, [64, 64]).unwrap();
    let b = Tensor::from_f32(backend.clone(), &b_data, [64, 64]).unwrap();
    let graph = backend.graph_matmul(&a, &b).unwrap();
    graph.replay().unwrap();
    backend.synchronize().unwrap();
    let eager = backend.ops().matmul(&a, &b).unwrap();
    assert_close(&values(graph.out()), &values(&eager));

    let n = 1024;
    let p_data = vec![0.1f32; n];
    let g_data = vec![0.01f32; n];
    let zero = vec![0.0f32; n];
    let p = Tensor::from_f32(backend.clone(), &p_data, [n]).unwrap();
    let g = Tensor::from_f32(backend.clone(), &g_data, [n]).unwrap();
    let m = Tensor::from_f32(backend.clone(), &zero, [n]).unwrap();
    let v = Tensor::from_f32(backend.clone(), &zero, [n]).unwrap();
    let cfg = AdamConfig {
        lr: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        step: 1,
    };
    let graph_adam = backend.graph_adam(&p, &g, &m, &v, &cfg).unwrap();
    graph_adam.replay().unwrap();
    backend.synchronize().unwrap();
    let (eager_p, eager_m, eager_v) = backend.ops().adam_update(&p, &g, &m, &v, &cfg).unwrap();
    assert_close(&values(graph_adam.param()), &values(&eager_p));
    assert_close(&values(graph_adam.m()), &values(&eager_m));
    assert_close(&values(graph_adam.v()), &values(&eager_v));
}

#[test]
#[serial]
fn cuda_memory_pool_reuses_buffers() {
    if !is_available() {
        eprintln!("skipping: CUDA is not available");
        return;
    }
    let backend = CudaBackend::new(0).unwrap();
    assert_eq!(backend.pooled_blocks(), 0);
    {
        let tensor = Tensor::from_f32(backend.clone(), &vec![1.0f32; 1024], [1024]).unwrap();
        backend.synchronize().unwrap();
        let _ = tensor;
    }
    let pooled = backend.pooled_blocks();
    assert!(pooled >= 1);
    {
        let zeroed = Tensor::zeros(backend.clone(), [1024], DType::F32).unwrap();
        backend.synchronize().unwrap();
        assert!(values(&zeroed).iter().all(|value| *value == 0.0));
    }
    assert!(backend.pooled_blocks() <= pooled + 1);
}

#[test]
#[serial]
fn cuda_compute_types_match_f32() {
    use modern_chainer_core::Activation;
    use modern_chainer_cuda::ComputeType;

    if !is_available() {
        eprintln!("skipping: CUDA is not available");
        return;
    }
    let backend = CudaBackend::new(0).unwrap();
    let batch = 16;
    let input = 32;
    let output = 8;
    let x_data: Vec<f32> = (0..batch * input)
        .map(|i| (i as f32 * 0.017).sin())
        .collect();
    let w_data: Vec<f32> = (0..output * input)
        .map(|i| (i as f32 * 0.013).cos() * 0.2)
        .collect();
    let bias_data: Vec<f32> = (0..output).map(|i| i as f32 * 0.01).collect();
    let x = Tensor::from_f32(backend.clone(), &x_data, [batch, input]).unwrap();
    let w = Tensor::from_f32(backend.clone(), &w_data, [output, input]).unwrap();
    let bias = Tensor::from_f32(backend.clone(), &bias_data, [output]).unwrap();

    backend.set_compute_type(ComputeType::F32);
    let reference = backend
        .ops()
        .linear_activation(&x, &w, Some(&bias), Activation::Gelu)
        .unwrap();
    let reference_values = values(&reference);

    for (compute, tolerance) in [(ComputeType::F16, 0.08f32), (ComputeType::Bf16, 0.15)] {
        backend.set_compute_type(compute);
        let mixed = backend
            .ops()
            .linear_activation(&x, &w, Some(&bias), Activation::Gelu)
            .unwrap();
        for (index, (actual, expected)) in values(&mixed).iter().zip(&reference_values).enumerate()
        {
            assert!(
                (actual - expected).abs() < tolerance,
                "{compute:?} element {index}: {actual} vs {expected}"
            );
        }
    }

    let a = Tensor::from_f32(backend.clone(), &x_data, [batch, input]).unwrap();
    let b = Tensor::from_f32(
        backend.clone(),
        &(0..input * output)
            .map(|i| (i as f32 * 0.007).sin())
            .collect::<Vec<_>>(),
        [input, output],
    )
    .unwrap();
    backend.set_compute_type(ComputeType::F32);
    let reference = values(&backend.ops().matmul(&a, &b).unwrap());
    backend.set_compute_type(ComputeType::F16);
    let mixed = values(&backend.ops().matmul(&a, &b).unwrap());
    for (index, (actual, expected)) in mixed.iter().zip(&reference).enumerate() {
        assert!(
            (actual - expected).abs() < 0.1,
            "matmul {index}: {actual} vs {expected}"
        );
    }
    backend.set_compute_type(ComputeType::Tf32);
}

#[test]
#[serial]
fn cuda_bf16_decoder_matches_f32_cpu() {
    use modern_chainer_cuda::ComputeType;
    use modern_chainer_llm::{Decoder, DecoderConfig};
    use modern_chainer_run::{Run, RunMode};

    if !is_available() {
        eprintln!("skipping: CUDA is not available");
        return;
    }
    let backend = CudaBackend::new(0).unwrap();
    backend.set_compute_type(ComputeType::Bf16);
    let config = DecoderConfig {
        vocab: 32,
        d_model: 8,
        n_layers: 2,
        n_heads: 2,
        d_ff: 16,
        max_seq: 24,
        eps: 1e-5,
    };
    let cuda_decoder = Decoder::new(backend.clone(), config.clone(), 7).unwrap();
    let cpu_decoder = Decoder::new(CpuBackend::new(), config, 7).unwrap();
    let mut cuda_run = Run::new(cuda_decoder, backend, RunMode::Inference, 11).unwrap();
    let mut cpu_run = Run::new(cpu_decoder, CpuBackend::new(), RunMode::Inference, 11).unwrap();
    let prompt = vec![1u32, 5, 9, 3];
    let cuda_logits = values(&cuda_run.step(prompt.clone()).unwrap());
    let cpu_logits = values(&cpu_run.step(prompt).unwrap());
    for (index, (a, b)) in cuda_logits.iter().zip(&cpu_logits).enumerate() {
        assert!(
            (a - b).abs() < 0.3,
            "bf16 decoder logit {index}: {a} vs {b}"
        );
    }
}

#[test]
#[serial]
fn cuda_autocast_decoder_and_amp_training() {
    use modern_chainer_autograd::Variable;
    use modern_chainer_autograd::ops as F;
    use modern_chainer_core::DType;
    use modern_chainer_core::autocast::autocast;
    use modern_chainer_llm::{Decoder, DecoderConfig};
    use modern_chainer_nn::{Linear, Module};
    use modern_chainer_optim::{Adam, GradScaler, zero_grad};
    use modern_chainer_run::{Run, RunMode};

    if !is_available() {
        eprintln!("skipping: CUDA is not available");
        return;
    }
    let backend = CudaBackend::new(0).unwrap();

    let config = DecoderConfig {
        vocab: 32,
        d_model: 8,
        n_layers: 2,
        n_heads: 2,
        d_ff: 16,
        max_seq: 24,
        eps: 1e-5,
    };
    let cuda_decoder = Decoder::new(backend.clone(), config.clone(), 7).unwrap();
    let cpu_decoder = Decoder::new(CpuBackend::new(), config, 7).unwrap();
    let mut cuda_run = Run::new(cuda_decoder, backend.clone(), RunMode::Inference, 11).unwrap();
    let mut cpu_run = Run::new(cpu_decoder, CpuBackend::new(), RunMode::Inference, 11).unwrap();
    let prompt = vec![1u32, 5, 9, 3];
    let cuda_logits = autocast(DType::F16, || cuda_run.step(prompt.clone()))
        .unwrap()
        .unwrap();
    let cpu_logits = cpu_run.step(prompt).unwrap();
    for (index, (a, b)) in values(&cuda_logits)
        .iter()
        .zip(values(&cpu_logits))
        .enumerate()
    {
        assert!(
            (a - b).abs() < 0.2,
            "autocast decoder logit {index}: {a} vs {b}"
        );
    }

    let weight = Variable::new(Tensor::from_f32(backend.clone(), &[0.2, -0.4], [1, 2]).unwrap());
    let bias = Variable::new(Tensor::from_f32(backend.clone(), &[0.1], [1]).unwrap());
    let model = Linear::from_parts(weight, Some(bias), 2, 1).unwrap();
    let x = Variable::new(
        Tensor::from_f32(
            backend.clone(),
            &[1.0, 2.0, 2.0, 1.0, 3.0, 1.0, 4.0, 2.0],
            [4, 2],
        )
        .unwrap(),
    );
    let target = Variable::constant(
        Tensor::from_f32(backend.clone(), &[3.0, 4.0, 5.0, 6.0], [4, 1]).unwrap(),
    );
    let params = model.parameters();
    let mut optimizer = Adam::new(0.01);
    let mut scaler = GradScaler::new()
        .with_scale(1024.0)
        .with_growth_interval(10);
    let mut first = None;
    let mut last = 0.0;
    for _ in 0..40 {
        zero_grad(&params);
        let loss = autocast(DType::F16, || {
            let prediction = model.forward(&x)?;
            let diff = F::sub(&prediction, &target)?;
            F::mean(&F::mul(&diff, &diff)?)
        })
        .unwrap()
        .unwrap();
        last = loss.data().backend().ops().to_scalar(&loss.data()).unwrap();
        first.get_or_insert(last);
        let scaled = scaler.scale_loss(&loss).unwrap();
        scaled.backward().unwrap();
        assert!(scaler.step(&mut optimizer, &params).unwrap());
    }
    let first = first.unwrap();
    eprintln!("cuda amp loss: {first} -> {last}, scale {}", scaler.scale());
    assert!(last < first);
}

#[test]
#[serial]
fn cuda_graph_linear_trainer_replays_training_steps() {
    use modern_chainer_cuda::{GraphLinearTrainer, GraphTrainerConfig};

    if !is_available() {
        eprintln!("skipping: CUDA is not available");
        return;
    }
    let backend = CudaBackend::new(0).unwrap();
    let config = GraphTrainerConfig {
        batch: 32,
        in_features: 16,
        out_features: 4,
        lr: 0.05,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
    };
    let mut trainer = GraphLinearTrainer::new(backend, config, 7).unwrap();
    let mut x = Vec::with_capacity(32 * 16);
    let mut t = Vec::with_capacity(32 * 4);
    for row in 0..32 {
        let mut features = Vec::with_capacity(16);
        for col in 0..16 {
            features.push(((row * 16 + col) as f32 * 0.01).sin());
        }
        for out in 0..4 {
            let target: f32 = features
                .iter()
                .enumerate()
                .map(|(i, value)| value * ((i + out) as f32 * 0.03).cos())
                .sum();
            t.push(target);
        }
        x.extend_from_slice(&features);
    }
    trainer.set_input(&x).unwrap();
    trainer.set_target(&t).unwrap();
    trainer.capture().unwrap();
    trainer.replay().unwrap();
    trainer.synchronize().unwrap();
    let first = trainer.loss().unwrap();
    for _ in 0..200 {
        trainer.replay().unwrap();
    }
    trainer.synchronize().unwrap();
    let last = trainer.loss().unwrap();
    assert_eq!(trainer.step_count().unwrap(), 201);
    eprintln!("graph trainer loss: {first} -> {last}");
    assert!(
        last < first * 0.1,
        "loss did not decrease: {first} -> {last}"
    );
    let (w, b) = trainer.weights().unwrap();
    assert_eq!(w.len(), 4 * 16);
    assert_eq!(b.len(), 4);
}

#[test]
#[serial]
fn cuda_static_run_linear_training_generalizes_graph_trainer() {
    use modern_chainer_cuda::{
        LinearMseAdamStep, StaticLinearConfig, StaticLinearInput, StaticRun,
    };

    if !is_available() {
        eprintln!("skipping: CUDA is not available");
        return;
    }
    let backend = CudaBackend::new(0).unwrap();
    let config = StaticLinearConfig {
        batch: 32,
        in_features: 16,
        out_features: 4,
        lr: 0.05,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
    };
    let model = LinearMseAdamStep::new(config, 7).unwrap();
    let mut run = StaticRun::new(model, backend).unwrap();
    let mut x = Vec::with_capacity(32 * 16);
    let mut target = Vec::with_capacity(32 * 4);
    for row in 0..32 {
        let features: Vec<f32> = (0..16)
            .map(|col| ((row * 16 + col) as f32 * 0.01).sin())
            .collect();
        for out in 0..4 {
            let value: f32 = features
                .iter()
                .enumerate()
                .map(|(i, v)| v * ((i + out) as f32 * 0.03).cos())
                .sum();
            target.push(value);
        }
        x.extend_from_slice(&features);
    }
    let input = StaticLinearInput { x, target };
    run.replay(&input).unwrap();
    let first = run.output().unwrap().loss;
    for _ in 0..100 {
        run.replay(&input).unwrap();
    }
    let output = run.output().unwrap();
    eprintln!("static run loss: {first} -> {}", output.loss);
    assert!(output.loss < first * 0.2);
    assert_eq!(output.step, 101);
    assert_eq!(output.weight.len(), 4 * 16);
    assert_eq!(output.bias.len(), 4);
}

#[test]
#[serial]
fn cuda_static_run_linear_forward_replays_new_input() {
    use modern_chainer_cuda::{
        LinearForwardStep, StaticForwardConfig, StaticForwardInput, StaticRun,
    };

    if !is_available() {
        eprintln!("skipping: CUDA is not available");
        return;
    }
    let backend = CudaBackend::new(0).unwrap();
    let config = StaticForwardConfig {
        batch: 4,
        in_features: 8,
        out_features: 3,
    };
    let model = LinearForwardStep::new(config, 3).unwrap();
    let mut run = StaticRun::new(model, backend).unwrap();
    let x1: Vec<f32> = (0..32).map(|i| i as f32 * 0.01).collect();
    let x2: Vec<f32> = (0..32).map(|i| 1.0 - i as f32 * 0.01).collect();
    run.replay(&StaticForwardInput { x: x1 }).unwrap();
    let first = run.output().unwrap();
    run.replay(&StaticForwardInput { x: x2 }).unwrap();
    let second = run.output().unwrap();
    assert_eq!(first.len(), 12);
    assert_eq!(second.len(), 12);
    assert_ne!(first, second);
}

#[test]
#[serial]
fn cuda_static_decoder_replays_tokens() {
    use modern_chainer_cuda::{StaticDecoderInput, StaticDecoderStep, StaticRun};
    use modern_chainer_llm::{Decoder, DecoderConfig};
    use modern_chainer_run::{Run, RunMode};

    if !is_available() {
        eprintln!("skipping: CUDA is not available");
        return;
    }
    let backend = CudaBackend::new(0).unwrap();
    let config = DecoderConfig {
        vocab: 32,
        d_model: 8,
        n_layers: 2,
        n_heads: 2,
        d_ff: 16,
        max_seq: 24,
        eps: 1e-5,
    };
    let cuda_decoder = Decoder::new(backend.clone(), config.clone(), 7).unwrap();
    let model = StaticDecoderStep::from_decoder(cuda_decoder).unwrap();
    let mut run = StaticRun::new(model, backend).unwrap();
    let cpu_decoder = Decoder::new(CpuBackend::new(), config, 7).unwrap();
    let mut cpu_run = Run::new(cpu_decoder, CpuBackend::new(), RunMode::Inference, 11).unwrap();

    let prompt = [1u32, 5, 9, 3, 7];
    for token in prompt {
        run.replay(&StaticDecoderInput { token }).unwrap();
        let output = run.output().unwrap();
        let cpu_logits = values(&cpu_run.step(vec![token]).unwrap());
        for (index, (a, b)) in output.logits.iter().zip(&cpu_logits).enumerate() {
            assert!(
                (a - b).abs() < 1e-2,
                "token {token} logit {index}: static {a} vs dense {b}"
            );
        }
    }
    let output = run.output().unwrap();
    assert_eq!(output.position, 5);
    assert_eq!(output.logits.len(), 32);
}
