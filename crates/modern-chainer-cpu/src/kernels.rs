use modern_chainer_core::{DType, Error, Result, Shape, Tensor};
use rayon::prelude::*;

use crate::backend::{CpuBackend, CpuTensorExt, f32_slice, make_f32};

pub(crate) const PAR_THRESHOLD: usize = 1 << 15;

fn gemm_parallelism() -> gemm::Parallelism {
    match std::env::var("MC_GEMM_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        Some(0) => gemm::Parallelism::None,
        Some(n) => gemm::Parallelism::Rayon(n),
        None => gemm::Parallelism::Rayon(rayon::current_num_threads()),
    }
}

/// Offset into a contiguous input for broadcast element `linear` of `out_dims`.
pub fn broadcast_offset(
    out_dims: &[usize],
    in_dims: &[usize],
    in_strides: &[isize],
    mut linear: usize,
) -> usize {
    let rank = out_dims.len();
    let in_rank = in_dims.len();
    let shift = rank.saturating_sub(in_rank);
    let mut offset = 0isize;
    for d in (0..rank).rev() {
        let dim = out_dims[d];
        let coord = if dim == 0 { 0 } else { linear % dim };
        linear /= dim.max(1);
        if d >= shift {
            let id = d - shift;
            if in_dims[id] != 1 {
                offset += coord as isize * in_strides[id];
            }
        }
    }
    offset as usize
}

fn out_shape_for(a: &Tensor<CpuBackend>, b: &Tensor<CpuBackend>) -> Result<Shape> {
    a.shape().broadcast_with(b.shape())
}

pub(crate) fn binary_map(
    a: &Tensor<CpuBackend>,
    b: &Tensor<CpuBackend>,
    f: impl Fn(f32, f32) -> f32 + Sync + Send,
) -> Result<Tensor<CpuBackend>> {
    let av = f32_slice(a)?;
    let bv = f32_slice(b)?;
    let shape = out_shape_for(a, b)?;
    let n = shape.numel();

    if a.shape() == b.shape() {
        let out: Vec<f32> = if n >= PAR_THRESHOLD {
            (0..n).into_par_iter().map(|i| f(av[i], bv[i])).collect()
        } else {
            (0..n).map(|i| f(av[i], bv[i])).collect()
        };
        return make_f32(a, out, shape);
    }

    let b_single = bv.len() == 1;
    let a_single = av.len() == 1;
    if b_single || a_single {
        let value = if b_single { bv[0] } else { av[0] };
        let src = if b_single { av } else { bv };
        let scalar_on_right = b_single;
        let out: Vec<f32> = if n >= PAR_THRESHOLD {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    if scalar_on_right {
                        f(src[i], value)
                    } else {
                        f(value, src[i])
                    }
                })
                .collect()
        } else {
            (0..n)
                .map(|i| {
                    if scalar_on_right {
                        f(src[i], value)
                    } else {
                        f(value, src[i])
                    }
                })
                .collect()
        };
        return make_f32(a, out, shape);
    }

    if a.rank() == 2 && b.rank() == 1 && b.shape().dims()[0] == a.shape().dims()[1] {
        let rows = a.shape().dims()[0];
        let cols = a.shape().dims()[1];
        let f_ref = &f;
        let out: Vec<f32> = if n >= PAR_THRESHOLD {
            (0..rows)
                .into_par_iter()
                .flat_map_iter(move |row| {
                    let base = row * cols;
                    (0..cols).map(move |col| f_ref(av[base + col], bv[col]))
                })
                .collect()
        } else {
            let mut out = Vec::with_capacity(n);
            for row in 0..rows {
                let base = row * cols;
                for col in 0..cols {
                    out.push(f(av[base + col], bv[col]));
                }
            }
            out
        };
        return make_f32(a, out, shape);
    }

    let out_dims = shape.dims().to_vec();
    let a_dims = a.shape().dims().to_vec();
    let b_dims = b.shape().dims().to_vec();
    let a_strides = a.strides().as_slice().to_vec();
    let b_strides = b.strides().as_slice().to_vec();
    let compute = |i: usize| {
        let ia = broadcast_offset(&out_dims, &a_dims, &a_strides, i);
        let ib = broadcast_offset(&out_dims, &b_dims, &b_strides, i);
        f(av[ia], bv[ib])
    };
    let out: Vec<f32> = if n >= PAR_THRESHOLD {
        (0..n).into_par_iter().map(compute).collect()
    } else {
        (0..n).map(compute).collect()
    };
    make_f32(a, out, shape)
}

pub(crate) fn unary_map(
    a: &Tensor<CpuBackend>,
    f: impl Fn(f32) -> f32 + Sync + Send,
) -> Result<Tensor<CpuBackend>> {
    let av = f32_slice(a)?;
    let out: Vec<f32> = if av.len() >= PAR_THRESHOLD {
        av.par_iter().map(|&x| f(x)).collect()
    } else {
        av.iter().map(|&x| f(x)).collect()
    };
    make_f32(a, out, a.shape().clone())
}

pub(crate) fn matmul(a: &Tensor<CpuBackend>, b: &Tensor<CpuBackend>) -> Result<Tensor<CpuBackend>> {
    if a.rank() != 2 || b.rank() != 2 {
        return Err(Error::invalid(format!(
            "matmul expects 2-D tensors, got {} and {}",
            a.shape(),
            b.shape()
        )));
    }
    let (m, k) = (a.shape().dims()[0], a.shape().dims()[1]);
    let (k2, n) = (b.shape().dims()[0], b.shape().dims()[1]);
    if k != k2 {
        return Err(Error::shape_mismatch(
            "matmul inner dimension",
            a.shape().clone(),
            b.shape().clone(),
        ));
    }
    let av = f32_slice(a)?;
    let bv = f32_slice(b)?;
    let mut out = vec![0f32; m * n];
    if m == 0 || n == 0 || k == 0 {
        return make_f32(a, out, Shape::new([m, n]));
    }
    let work = m.saturating_mul(n).saturating_mul(k);
    if work < 1_000_000 {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for d in 0..k {
                    acc += av[i * k + d] * bv[d * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        return make_f32(a, out, Shape::new([m, n]));
    }
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            out.as_mut_ptr(),
            1,
            n as isize,
            true,
            av.as_ptr(),
            1,
            k as isize,
            bv.as_ptr(),
            1,
            n as isize,
            1.0f32,
            1.0f32,
            false,
            false,
            false,
            gemm_parallelism(),
        );
    }
    make_f32(a, out, Shape::new([m, n]))
}

/// Fused `x @ W^T + b` used by `L.Linear`.
pub(crate) fn linear(
    x: &Tensor<CpuBackend>,
    weight: &Tensor<CpuBackend>,
    bias: Option<&Tensor<CpuBackend>>,
) -> Result<Tensor<CpuBackend>> {
    if x.rank() != 2 || weight.rank() != 2 {
        return Err(Error::invalid(format!(
            "linear expects 2-D input and weight, got {} and {}",
            x.shape(),
            weight.shape()
        )));
    }
    let (m, k) = (x.shape().dims()[0], x.shape().dims()[1]);
    let (out_features, k2) = (weight.shape().dims()[0], weight.shape().dims()[1]);
    if k != k2 {
        return Err(Error::shape_mismatch(
            "linear inner dimension",
            x.shape().clone(),
            weight.shape().clone(),
        ));
    }
    if let Some(bias) = bias
        && bias.shape().dims() != [out_features]
    {
        return Err(Error::invalid(format!(
            "linear bias must be [{out_features}], got {}",
            bias.shape()
        )));
    }
    let xv = f32_slice(x)?;
    let wv = f32_slice(weight)?;
    let mut out = vec![0f32; m * out_features];
    let work = m.saturating_mul(out_features).saturating_mul(k);
    if work > 0 && work < 1_000_000 {
        for i in 0..m {
            for j in 0..out_features {
                let mut acc = 0f32;
                for d in 0..k {
                    acc += xv[i * k + d] * wv[j * k + d];
                }
                out[i * out_features + j] = acc;
            }
        }
    } else if m > 0 && out_features > 0 && k > 0 {
        unsafe {
            gemm::gemm(
                m,
                out_features,
                k,
                out.as_mut_ptr(),
                1,
                out_features as isize,
                true,
                xv.as_ptr(),
                1,
                k as isize,
                // weight is [out, in]; present it transposed as [in, out]
                wv.as_ptr(),
                k as isize,
                1,
                1.0f32,
                1.0f32,
                false,
                false,
                false,
                gemm_parallelism(),
            );
        }
    }
    if let Some(bias) = bias {
        let bv = f32_slice(bias)?;
        if m * out_features >= PAR_THRESHOLD {
            out.par_chunks_mut(out_features).for_each(|row| {
                for (value, &bias) in row.iter_mut().zip(bv) {
                    *value += bias;
                }
            });
        } else {
            for row in out.chunks_mut(out_features) {
                for (value, &bias) in row.iter_mut().zip(bv) {
                    *value += bias;
                }
            }
        }
    }
    make_f32(x, out, Shape::new([m, out_features]))
}

pub(crate) fn transpose2d(a: &Tensor<CpuBackend>) -> Result<Tensor<CpuBackend>> {
    if a.rank() != 2 {
        return Err(Error::invalid(format!(
            "transpose2d expects a 2-D tensor, got {}",
            a.shape()
        )));
    }
    let (r, c) = (a.shape().dims()[0], a.shape().dims()[1]);
    let av = f32_slice(a)?;
    let out: Vec<f32> = if r * c >= PAR_THRESHOLD {
        (0..c)
            .into_par_iter()
            .flat_map_iter(|j| (0..r).map(move |i| av[i * c + j]))
            .collect()
    } else {
        let mut out = Vec::with_capacity(r * c);
        for j in 0..c {
            for i in 0..r {
                out.push(av[i * c + j]);
            }
        }
        out
    };
    make_f32(a, out, Shape::new([c, r]))
}

pub(crate) fn sum_all(a: &Tensor<CpuBackend>) -> Result<Tensor<CpuBackend>> {
    let av = f32_slice(a)?;
    let sum: f64 = if av.len() >= PAR_THRESHOLD {
        av.par_iter().map(|&x| x as f64).sum()
    } else {
        av.iter().map(|&x| x as f64).sum()
    };
    Tensor::from_f32(*a.backend(), &[sum as f32], Shape::scalar())
}

pub(crate) fn sum_to(a: &Tensor<CpuBackend>, target: &Shape) -> Result<Tensor<CpuBackend>> {
    let av = f32_slice(a)?;
    if target.rank() > a.rank() {
        return Err(Error::invalid(format!(
            "sum_to target rank {} exceeds source rank {}",
            target.rank(),
            a.rank()
        )));
    }
    let out_strides = target.contiguous_strides();
    let out_strides = out_strides.as_slice();
    let out_dims = target.dims();
    let src_dims = a.shape().dims();
    let shift = a.rank() - target.rank();
    let mut out = vec![0f32; target.numel()];
    if out.is_empty() {
        return make_f32(a, out, target.clone());
    }
    if src_dims.len() == 2 && target.rank() == 1 && target.dims()[0] == src_dims[1] {
        let rows = src_dims[0];
        let cols = src_dims[1];
        for row in 0..rows {
            let base = row * cols;
            for col in 0..cols {
                out[col] += av[base + col];
            }
        }
        return make_f32(a, out, target.clone());
    }
    for (linear, &value) in av.iter().enumerate() {
        let mut rest = linear;
        let mut offset = 0isize;
        for d in (0..src_dims.len()).rev() {
            let dim = src_dims[d];
            let coord = if dim == 0 { 0 } else { rest % dim };
            rest /= dim.max(1);
            if d >= shift {
                let od = d - shift;
                if out_dims[od] != 1 {
                    offset += coord as isize * out_strides[od];
                }
            }
        }
        out[offset as usize] += value;
    }
    make_f32(a, out, target.clone())
}

pub(crate) fn scale(a: &Tensor<CpuBackend>, factor: f64) -> Result<Tensor<CpuBackend>> {
    let factor = factor as f32;
    unary_map(a, move |x| x * factor)
}

pub(crate) fn relu(a: &Tensor<CpuBackend>) -> Result<Tensor<CpuBackend>> {
    unary_map(a, |x| x.max(0.0))
}

pub(crate) fn relu_backward(
    x: &Tensor<CpuBackend>,
    grad: &Tensor<CpuBackend>,
) -> Result<Tensor<CpuBackend>> {
    if x.shape() != grad.shape() {
        return Err(Error::shape_mismatch(
            "relu backward",
            x.shape().clone(),
            grad.shape().clone(),
        ));
    }
    let xv = f32_slice(x)?;
    let gv = f32_slice(grad)?;
    let out: Vec<f32> = if xv.len() >= PAR_THRESHOLD {
        (0..xv.len())
            .into_par_iter()
            .map(|i| if xv[i] > 0.0 { gv[i] } else { 0.0 })
            .collect()
    } else {
        (0..xv.len())
            .map(|i| if xv[i] > 0.0 { gv[i] } else { 0.0 })
            .collect()
    };
    make_f32(x, out, x.shape().clone())
}

fn target_indices(target: &Tensor<CpuBackend>) -> Result<Vec<usize>> {
    let values: Vec<i64> = match target.dtype() {
        DType::I64 => target.as_slice::<i64>()?.to_vec(),
        DType::I32 => target
            .as_slice::<i32>()?
            .iter()
            .map(|&x| x as i64)
            .collect(),
        DType::U8 => target.as_slice::<u8>()?.iter().map(|&x| x as i64).collect(),
        other => return Err(Error::UnsupportedDType(other)),
    };
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        if value < 0 {
            return Err(Error::invalid(format!("negative class index {value}")));
        }
        out.push(value as usize);
    }
    Ok(out)
}

fn logits_2d(logits: &Tensor<CpuBackend>) -> Result<(usize, usize)> {
    if logits.rank() != 2 {
        return Err(Error::invalid(format!(
            "softmax_cross_entropy expects [batch, classes] logits, got {}",
            logits.shape()
        )));
    }
    Ok((logits.shape().dims()[0], logits.shape().dims()[1]))
}

pub(crate) fn softmax_cross_entropy(
    logits: &Tensor<CpuBackend>,
    target: &Tensor<CpuBackend>,
) -> Result<Tensor<CpuBackend>> {
    let (n, c) = logits_2d(logits)?;
    let x = f32_slice(logits)?;
    let labels = target_indices(target)?;
    if labels.len() != n {
        return Err(Error::invalid(format!(
            "expected {n} targets, got {}",
            labels.len()
        )));
    }
    if n == 0 {
        return Tensor::from_f32(*logits.backend(), &[0.0], Shape::scalar());
    }
    let loss: f64 = if n * c >= PAR_THRESHOLD {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let label = labels[i];
                let row = &x[i * c..(i + 1) * c];
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f64;
                for &v in row {
                    sum += ((v - max) as f64).exp();
                }
                (max as f64 + sum.ln()) - row[label] as f64
            })
            .sum()
    } else {
        let mut loss = 0f64;
        for i in 0..n {
            let label = labels[i];
            let row = &x[i * c..(i + 1) * c];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f64;
            for &v in row {
                sum += ((v - max) as f64).exp();
            }
            loss += (max as f64 + sum.ln()) - row[label] as f64;
        }
        loss
    };
    Tensor::from_f32(
        *logits.backend(),
        &[(loss / n as f64) as f32],
        Shape::scalar(),
    )
}

pub(crate) fn softmax_cross_entropy_backward(
    logits: &Tensor<CpuBackend>,
    target: &Tensor<CpuBackend>,
) -> Result<Tensor<CpuBackend>> {
    let (n, c) = logits_2d(logits)?;
    let x = f32_slice(logits)?;
    let labels = target_indices(target)?;
    if labels.len() != n {
        return Err(Error::invalid(format!(
            "expected {n} targets, got {}",
            labels.len()
        )));
    }
    let mut out = vec![0f32; n * c];
    if n == 0 {
        return make_f32(logits, out, logits.shape().clone());
    }
    let inv_n = 1.0 / n as f32;
    for i in 0..n {
        let label = labels[i];
        if label >= c {
            return Err(Error::invalid(format!(
                "class index {label} out of range for {c} classes"
            )));
        }
        let src = &x[i * c..(i + 1) * c];
        let dst = &mut out[i * c..(i + 1) * c];
        let max = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f64;
        for &v in src {
            sum += ((v - max) as f64).exp();
        }
        let inv_sum = 1.0 / sum;
        for j in 0..c {
            dst[j] = (((src[j] - max) as f64).exp() * inv_sum) as f32;
        }
        dst[label] -= 1.0;
        for value in dst.iter_mut() {
            *value *= inv_n;
        }
    }
    make_f32(logits, out, logits.shape().clone())
}

/// Gradients of the fused linear op, without materialising `grad^T`.
pub(crate) fn linear_backward(
    grad: &Tensor<CpuBackend>,
    x: &Tensor<CpuBackend>,
    weight: &Tensor<CpuBackend>,
) -> Result<(Tensor<CpuBackend>, Tensor<CpuBackend>)> {
    let (batch, out_features) = (grad.shape().dims()[0], grad.shape().dims()[1]);
    let in_features = weight.shape().dims()[1];
    let gv = f32_slice(grad)?;
    let xv = f32_slice(x)?;
    let wv = f32_slice(weight)?;

    let work = batch
        .saturating_mul(out_features)
        .saturating_mul(in_features);
    let mut grad_x = vec![0f32; batch * in_features];
    let mut grad_w = vec![0f32; out_features * in_features];
    if work > 0 && work < 1_000_000 {
        for i in 0..batch {
            for d in 0..in_features {
                let mut acc = 0f32;
                for j in 0..out_features {
                    acc += gv[i * out_features + j] * wv[j * in_features + d];
                }
                grad_x[i * in_features + d] = acc;
            }
        }
        for j in 0..out_features {
            for d in 0..in_features {
                let mut acc = 0f32;
                for i in 0..batch {
                    acc += gv[i * out_features + j] * xv[i * in_features + d];
                }
                grad_w[j * in_features + d] = acc;
            }
        }
    } else if batch > 0 && in_features > 0 && out_features > 0 {
        unsafe {
            gemm::gemm(
                batch,
                in_features,
                out_features,
                grad_x.as_mut_ptr(),
                1,
                in_features as isize,
                true,
                gv.as_ptr(),
                1,
                out_features as isize,
                wv.as_ptr(),
                1,
                in_features as isize,
                1.0f32,
                1.0f32,
                false,
                false,
                false,
                gemm_parallelism(),
            );
        }
    }

    if work >= 1_000_000 && out_features > 0 && in_features > 0 && batch > 0 {
        let grad_t = transpose2d(grad)?;
        let gt = f32_slice(&grad_t)?;
        unsafe {
            gemm::gemm(
                out_features,
                in_features,
                batch,
                grad_w.as_mut_ptr(),
                1,
                in_features as isize,
                true,
                gt.as_ptr(),
                1,
                batch as isize,
                xv.as_ptr(),
                1,
                in_features as isize,
                1.0f32,
                1.0f32,
                false,
                false,
                false,
                gemm_parallelism(),
            );
        }
    }

    Ok((
        make_f32(grad, grad_x, Shape::new([batch, in_features]))?,
        make_f32(grad, grad_w, Shape::new([out_features, in_features]))?,
    ))
}

/// Fused Adam update in a single pass over the parameter.
pub(crate) fn adam_update(
    param: &Tensor<CpuBackend>,
    grad: &Tensor<CpuBackend>,
    m: &Tensor<CpuBackend>,
    v: &Tensor<CpuBackend>,
    cfg: &modern_chainer_core::AdamConfig,
) -> Result<(Tensor<CpuBackend>, Tensor<CpuBackend>, Tensor<CpuBackend>)> {
    let pv = f32_slice(param)?;
    let gv = f32_slice(grad)?;
    let mv = f32_slice(m)?;
    let vv = f32_slice(v)?;
    let n = pv.len();
    if gv.len() != n || mv.len() != n || vv.len() != n {
        return Err(Error::invalid("adam_update shape mismatch"));
    }
    let b1 = cfg.beta1 as f32;
    let b2 = cfg.beta2 as f32;
    let one_minus_b1 = 1.0 - b1;
    let one_minus_b2 = 1.0 - b2;
    let bc1 = 1.0 - b1.powi(cfg.step as i32);
    let bc2 = 1.0 - b2.powi(cfg.step as i32);
    let lr = cfg.lr as f32;
    let eps = cfg.eps as f32;

    let mut new_p: Vec<f32> = Vec::with_capacity(n);
    let mut new_m: Vec<f32> = Vec::with_capacity(n);
    let mut new_v: Vec<f32> = Vec::with_capacity(n);

    let fill = |out_p: &mut [std::mem::MaybeUninit<f32>],
                out_m: &mut [std::mem::MaybeUninit<f32>],
                out_v: &mut [std::mem::MaybeUninit<f32>],
                base: usize| {
        for (j, ((p, m), v)) in out_p
            .iter_mut()
            .zip(out_m.iter_mut())
            .zip(out_v.iter_mut())
            .enumerate()
        {
            let i = base + j;
            let g = gv[i];
            let nm = b1 * mv[i] + one_minus_b1 * g;
            let nv = b2 * vv[i] + one_minus_b2 * g * g;
            let m_hat = nm / bc1;
            let v_hat = nv / bc2;
            m.write(nm);
            v.write(nv);
            p.write(pv[i] - lr * m_hat / (v_hat.sqrt() + eps));
        }
    };

    if n >= PAR_THRESHOLD {
        let chunk = 4096;
        let p_chunks = new_p.spare_capacity_mut().par_chunks_mut(chunk);
        let m_chunks = new_m.spare_capacity_mut().par_chunks_mut(chunk);
        let v_chunks = new_v.spare_capacity_mut().par_chunks_mut(chunk);
        p_chunks
            .zip(m_chunks)
            .zip(v_chunks)
            .enumerate()
            .for_each(|(ci, ((p, m), v))| fill(p, m, v, ci * chunk));
    } else {
        fill(
            new_p.spare_capacity_mut(),
            new_m.spare_capacity_mut(),
            new_v.spare_capacity_mut(),
            0,
        );
    }
    // SAFETY: every element up to `n` was written above.
    unsafe {
        new_p.set_len(n);
        new_m.set_len(n);
        new_v.set_len(n);
    }

    Ok((
        make_f32(param, new_p, param.shape().clone())?,
        make_f32(param, new_m, m.shape().clone())?,
        make_f32(param, new_v, v.shape().clone())?,
    ))
}
