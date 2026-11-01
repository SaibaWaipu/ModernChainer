//! Correctness-first host implementations for shape-heavy neural-network ops.
//!
//! Convolution/pooling/normalization/attention kernels are prototyped on the
//! host and uploaded to the tensor's backend. This keeps the public API and
//! gradients usable on CPU/CUDA/Vulkan while native kernels are developed.

use modern_chainer_core::{Backend, Error, Result, Shape, Tensor};

use crate::node::{Op, apply, apply1};
use crate::variable::Variable;

fn host<B: Backend>(tensor: &Tensor<B>) -> Result<Vec<f32>> {
    let bytes = tensor.to_bytes()?;
    let (chunks, rest) = bytes.as_chunks::<4>();
    if !rest.is_empty() {
        return Err(Error::invalid("host op expects f32 tensors"));
    }
    Ok(chunks
        .iter()
        .map(|chunk| f32::from_ne_bytes(*chunk))
        .collect())
}

fn upload<B: Backend>(
    like: &Tensor<B>,
    data: &[f32],
    shape: impl Into<Shape>,
) -> Result<Tensor<B>> {
    Tensor::from_f32(like.backend().clone(), data, shape)
}

fn checked_out(size: usize, kernel: usize, stride: usize, padding: usize) -> Result<usize> {
    let padded = size + 2 * padding;
    if padded < kernel {
        return Err(Error::invalid("kernel larger than padded input"));
    }
    Ok((padded - kernel) / stride + 1)
}

#[derive(Clone, Debug)]
pub struct Conv2dConfig {
    pub stride: usize,
    pub padding: usize,
}

pub struct Conv2dOp {
    stride: usize,
    padding: usize,
}

impl Conv2dOp {
    pub fn new(stride: usize, padding: usize) -> Self {
        Self { stride, padding }
    }
}

impl<B: Backend> Op<B> for Conv2dOp {
    const NAME: &'static str = "Conv2d";

    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        if inputs.len() != 2 && inputs.len() != 3 {
            return Err(Error::invalid("conv2d expects x, weight[, bias]"));
        }
        let x = inputs[0];
        let w = inputs[1];
        let xd = x.shape().dims();
        let wd = w.shape().dims();
        if xd.len() != 4 || wd.len() != 4 {
            return Err(Error::invalid("conv2d expects [N,C,H,W] and [O,C,KH,KW]"));
        }
        let (n, c, h, width) = (xd[0], xd[1], xd[2], xd[3]);
        let (o, wc, kh, kw) = (wd[0], wd[1], wd[2], wd[3]);
        if c != wc {
            return Err(Error::invalid("conv2d channel mismatch"));
        }
        let oh = checked_out(h, kh, self.stride, self.padding)?;
        let ow = checked_out(width, kw, self.stride, self.padding)?;
        let xv = host(x)?;
        let wv = host(w)?;
        let bv = if inputs.len() == 3 {
            Some(host(inputs[2])?)
        } else {
            None
        };
        if let Some(b) = &bv
            && b.len() != o
        {
            return Err(Error::invalid("conv2d bias size mismatch"));
        }
        let mut out = vec![0f32; n * o * oh * ow];
        for ni in 0..n {
            for oi in 0..o {
                for y in 0..oh {
                    for xi in 0..ow {
                        let mut sum = bv.as_ref().map_or(0.0, |b| b[oi]);
                        for ci in 0..c {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let iy = y * self.stride + ky;
                                    let ix = xi * self.stride + kx;
                                    if iy >= self.padding
                                        && ix >= self.padding
                                        && iy - self.padding < h
                                        && ix - self.padding < width
                                    {
                                        let iv = ((ni * c + ci) * h + (iy - self.padding)) * width
                                            + (ix - self.padding);
                                        let ww = ((oi * c + ci) * kh + ky) * kw + kx;
                                        sum += xv[iv] * wv[ww];
                                    }
                                }
                            }
                        }
                        out[((ni * o + oi) * oh + y) * ow + xi] = sum;
                    }
                }
            }
        }
        Ok(vec![upload(x, &out, [n, o, oh, ow])?])
    }

    fn backward(
        &self,
        grad: &Tensor<B>,
        inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let x = inputs[0];
        let w = inputs[1];
        let xd = x.shape().dims();
        let wd = w.shape().dims();
        let (n, c, h, width) = (xd[0], xd[1], xd[2], xd[3]);
        let (o, _, kh, kw) = (wd[0], wd[1], wd[2], wd[3]);
        let gd = grad.shape().dims();
        let (oh, ow) = (gd[2], gd[3]);
        let xv = host(x)?;
        let wv = host(w)?;
        let gv = host(grad)?;
        let mut dx = vec![0f32; xv.len()];
        let mut dw = vec![0f32; wv.len()];
        let mut db = vec![0f32; o];
        for ni in 0..n {
            for oi in 0..o {
                for y in 0..oh {
                    for xi in 0..ow {
                        let g = gv[((ni * o + oi) * oh + y) * ow + xi];
                        db[oi] += g;
                        for ci in 0..c {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let iy = y * self.stride + ky;
                                    let ix = xi * self.stride + kx;
                                    if iy >= self.padding
                                        && ix >= self.padding
                                        && iy - self.padding < h
                                        && ix - self.padding < width
                                    {
                                        let iv = ((ni * c + ci) * h + (iy - self.padding)) * width
                                            + (ix - self.padding);
                                        let wi = ((oi * c + ci) * kh + ky) * kw + kx;
                                        dx[iv] += g * wv[wi];
                                        dw[wi] += g * xv[iv];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let mut grads = vec![
            Some(upload(x, &dx, x.shape().clone())?),
            Some(upload(w, &dw, w.shape().clone())?),
        ];
        if inputs.len() == 3 {
            grads.push(Some(upload(inputs[2], &db, inputs[2].shape().clone())?));
        }
        Ok(grads)
    }
}

pub struct ConvTranspose2dOp {
    stride: usize,
    padding: usize,
    output_padding: usize,
}

impl ConvTranspose2dOp {
    pub fn new(stride: usize, padding: usize, output_padding: usize) -> Self {
        Self {
            stride,
            padding,
            output_padding,
        }
    }
}

impl<B: Backend> Op<B> for ConvTranspose2dOp {
    const NAME: &'static str = "ConvTranspose2d";

    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        if inputs.len() != 2 && inputs.len() != 3 {
            return Err(Error::invalid("conv_transpose2d expects x, weight[, bias]"));
        }
        let x = inputs[0];
        let w = inputs[1];
        let xd = x.shape().dims();
        let wd = w.shape().dims();
        if xd.len() != 4 || wd.len() != 4 {
            return Err(Error::invalid(
                "conv_transpose2d expects [N,C,H,W] and [C,O,KH,KW]",
            ));
        }
        let (n, c, h, width) = (xd[0], xd[1], xd[2], xd[3]);
        let (wc, o, kh, kw) = (wd[0], wd[1], wd[2], wd[3]);
        if c != wc {
            return Err(Error::invalid("conv_transpose2d channel mismatch"));
        }
        let oh = (h - 1) * self.stride + kh + self.output_padding - 2 * self.padding;
        let ow = (width - 1) * self.stride + kw + self.output_padding - 2 * self.padding;
        let xv = host(x)?;
        let wv = host(w)?;
        let bv = if inputs.len() == 3 {
            Some(host(inputs[2])?)
        } else {
            None
        };
        let mut out = vec![0f32; n * o * oh * ow];
        if let Some(b) = &bv {
            for ni in 0..n {
                for oi in 0..o {
                    for y in 0..oh {
                        for xi in 0..ow {
                            out[((ni * o + oi) * oh + y) * ow + xi] = b[oi];
                        }
                    }
                }
            }
        }
        for ni in 0..n {
            for ci in 0..c {
                for y in 0..h {
                    for xi in 0..width {
                        let xvv = xv[((ni * c + ci) * h + y) * width + xi];
                        for oi in 0..o {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let oy = y * self.stride + ky;
                                    let ox = xi * self.stride + kx;
                                    if oy >= self.padding
                                        && ox >= self.padding
                                        && oy - self.padding < oh
                                        && ox - self.padding < ow
                                    {
                                        let wi = ((ci * o + oi) * kh + ky) * kw + kx;
                                        out[((ni * o + oi) * oh + (oy - self.padding)) * ow
                                            + (ox - self.padding)] += xvv * wv[wi];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(vec![upload(x, &out, [n, o, oh, ow])?])
    }

    fn backward(
        &self,
        grad: &Tensor<B>,
        inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let x = inputs[0];
        let w = inputs[1];
        let xd = x.shape().dims();
        let wd = w.shape().dims();
        let (n, c, h, width) = (xd[0], xd[1], xd[2], xd[3]);
        let (_, o, kh, kw) = (wd[0], wd[1], wd[2], wd[3]);
        let gd = grad.shape().dims();
        let (oh, ow) = (gd[2], gd[3]);
        let xv = host(x)?;
        let wv = host(w)?;
        let gv = host(grad)?;
        let mut dx = vec![0f32; xv.len()];
        let mut dw = vec![0f32; wv.len()];
        let mut db = vec![0f32; o];
        for ni in 0..n {
            for oi in 0..o {
                for y in 0..oh {
                    for xi in 0..ow {
                        db[oi] += gv[((ni * o + oi) * oh + y) * ow + xi];
                    }
                }
            }
        }
        for ni in 0..n {
            for ci in 0..c {
                for y in 0..h {
                    for xi in 0..width {
                        let iv = ((ni * c + ci) * h + y) * width + xi;
                        let xvv = xv[iv];
                        let mut dxv = 0f32;
                        for oi in 0..o {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let oy = y * self.stride + ky;
                                    let ox = xi * self.stride + kx;
                                    if oy >= self.padding
                                        && ox >= self.padding
                                        && oy - self.padding < oh
                                        && ox - self.padding < ow
                                    {
                                        let gi = ((ni * o + oi) * oh + (oy - self.padding)) * ow
                                            + (ox - self.padding);
                                        let wi = ((ci * o + oi) * kh + ky) * kw + kx;
                                        let g = gv[gi];
                                        dxv += g * wv[wi];
                                        dw[wi] += g * xvv;
                                    }
                                }
                            }
                        }
                        dx[iv] = dxv;
                    }
                }
            }
        }
        let mut grads = vec![
            Some(upload(x, &dx, x.shape().clone())?),
            Some(upload(w, &dw, w.shape().clone())?),
        ];
        if inputs.len() == 3 {
            grads.push(Some(upload(inputs[2], &db, inputs[2].shape().clone())?));
        }
        Ok(grads)
    }
}

pub struct Pool2dOp {
    kernel: usize,
    stride: usize,
    padding: usize,
    average: bool,
}

impl Pool2dOp {
    pub fn new(kernel: usize, stride: usize, padding: usize, average: bool) -> Self {
        Self {
            kernel,
            stride,
            padding,
            average,
        }
    }
}

impl<B: Backend> Op<B> for Pool2dOp {
    const NAME: &'static str = "Pool2d";

    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        if inputs.len() != 1 {
            return Err(Error::invalid("pool2d expects one input"));
        }
        let x = inputs[0];
        let xd = x.shape().dims();
        if xd.len() != 4 {
            return Err(Error::invalid("pool2d expects [N,C,H,W]"));
        }
        let (n, c, h, w) = (xd[0], xd[1], xd[2], xd[3]);
        let oh = checked_out(h, self.kernel, self.stride, self.padding)?;
        let ow = checked_out(w, self.kernel, self.stride, self.padding)?;
        let xv = host(x)?;
        let mut out = vec![0f32; n * c * oh * ow];
        for ni in 0..n {
            for ci in 0..c {
                for y in 0..oh {
                    for xi in 0..ow {
                        let mut acc = if self.average { 0.0 } else { f32::NEG_INFINITY };
                        let mut count = 0usize;
                        for ky in 0..self.kernel {
                            for kx in 0..self.kernel {
                                let iy = y * self.stride + ky;
                                let ix = xi * self.stride + kx;
                                if iy >= self.padding
                                    && ix >= self.padding
                                    && iy - self.padding < h
                                    && ix - self.padding < w
                                {
                                    let v = xv[((ni * c + ci) * h + (iy - self.padding)) * w
                                        + (ix - self.padding)];
                                    if self.average {
                                        acc += v;
                                    } else {
                                        acc = acc.max(v);
                                    }
                                    count += 1;
                                }
                            }
                        }
                        if self.average && count > 0 {
                            acc /= count as f32;
                        }
                        out[((ni * c + ci) * oh + y) * ow + xi] = acc;
                    }
                }
            }
        }
        Ok(vec![upload(x, &out, [n, c, oh, ow])?])
    }

    fn backward(
        &self,
        grad: &Tensor<B>,
        inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let x = inputs[0];
        let xd = x.shape().dims();
        let (n, c, h, w) = (xd[0], xd[1], xd[2], xd[3]);
        let gd = grad.shape().dims();
        let (oh, ow) = (gd[2], gd[3]);
        let xv = host(x)?;
        let gv = host(grad)?;
        let mut dx = vec![0f32; xv.len()];
        for ni in 0..n {
            for ci in 0..c {
                for y in 0..oh {
                    for xi in 0..ow {
                        let g = gv[((ni * c + ci) * oh + y) * ow + xi];
                        let mut count = 0usize;
                        let mut best = f32::NEG_INFINITY;
                        for ky in 0..self.kernel {
                            for kx in 0..self.kernel {
                                let iy = y * self.stride + ky;
                                let ix = xi * self.stride + kx;
                                if iy >= self.padding
                                    && ix >= self.padding
                                    && iy - self.padding < h
                                    && ix - self.padding < w
                                {
                                    count += 1;
                                    let v = xv[((ni * c + ci) * h + (iy - self.padding)) * w
                                        + (ix - self.padding)];
                                    best = best.max(v);
                                }
                            }
                        }
                        for ky in 0..self.kernel {
                            for kx in 0..self.kernel {
                                let iy = y * self.stride + ky;
                                let ix = xi * self.stride + kx;
                                if iy >= self.padding
                                    && ix >= self.padding
                                    && iy - self.padding < h
                                    && ix - self.padding < w
                                {
                                    let idx = ((ni * c + ci) * h + (iy - self.padding)) * w
                                        + (ix - self.padding);
                                    if self.average {
                                        dx[idx] += g / count.max(1) as f32;
                                    } else if xv[idx] == best {
                                        dx[idx] += g;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(vec![Some(upload(x, &dx, x.shape().clone())?)])
    }
}

pub struct BatchNormOp {
    eps: f64,
}

impl BatchNormOp {
    pub fn new(eps: f64) -> Self {
        Self { eps }
    }
}

impl<B: Backend> Op<B> for BatchNormOp {
    const NAME: &'static str = "BatchNorm";

    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        if inputs.len() != 3 {
            return Err(Error::invalid("batch_norm expects x, gamma, beta"));
        }
        let x = inputs[0];
        let xd = x.shape().dims();
        if xd.len() < 2 {
            return Err(Error::invalid("batch_norm expects rank >= 2"));
        }
        let (n, c) = (xd[0], xd[1]);
        let spatial: usize = xd[2..].iter().product();
        let gamma = host(inputs[1])?;
        let beta = host(inputs[2])?;
        if gamma.len() != c || beta.len() != c {
            return Err(Error::invalid("batch_norm gamma/beta size mismatch"));
        }
        let xv = host(x)?;
        let mut out = vec![0f32; xv.len()];
        for ci in 0..c {
            let mut sum = 0f64;
            let mut sq = 0f64;
            for ni in 0..n {
                for s in 0..spatial {
                    let v = xv[((ni * c + ci) * spatial) + s] as f64;
                    sum += v;
                    sq += v * v;
                }
            }
            let count = (n * spatial) as f64;
            let mean = sum / count;
            let var = (sq / count - mean * mean).max(0.0);
            let inv = 1.0 / (var + self.eps).sqrt();
            for ni in 0..n {
                for s in 0..spatial {
                    let idx = (ni * c + ci) * spatial + s;
                    let norm = (xv[idx] as f64 - mean) * inv;
                    out[idx] = (norm * gamma[ci] as f64 + beta[ci] as f64) as f32;
                }
            }
        }
        Ok(vec![upload(x, &out, x.shape().clone())?])
    }

    fn backward(
        &self,
        grad: &Tensor<B>,
        inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let x = inputs[0];
        let xd = x.shape().dims();
        let (n, c) = (xd[0], xd[1]);
        let spatial: usize = xd[2..].iter().product();
        let gamma = host(inputs[1])?;
        let xv = host(x)?;
        let gv = host(grad)?;
        let mut dx = vec![0f32; xv.len()];
        let mut dgamma = vec![0f32; c];
        let mut dbeta = vec![0f32; c];
        let count = (n * spatial) as f64;
        for ci in 0..c {
            let mut sum = 0f64;
            let mut sq = 0f64;
            for ni in 0..n {
                for s in 0..spatial {
                    let v = xv[(ni * c + ci) * spatial + s] as f64;
                    sum += v;
                    sq += v * v;
                }
            }
            let mean = sum / count;
            let var = (sq / count - mean * mean).max(0.0);
            let inv = 1.0 / (var + self.eps).sqrt();
            let mut sum_dy = 0f64;
            let mut sum_dy_norm = 0f64;
            for ni in 0..n {
                for s in 0..spatial {
                    let idx = (ni * c + ci) * spatial + s;
                    let norm = (xv[idx] as f64 - mean) * inv;
                    let dy = gv[idx] as f64 * gamma[ci] as f64;
                    sum_dy += dy;
                    sum_dy_norm += dy * norm;
                    dgamma[ci] += (gv[idx] as f64 * norm) as f32;
                    dbeta[ci] += gv[idx];
                }
            }
            for ni in 0..n {
                for s in 0..spatial {
                    let idx = (ni * c + ci) * spatial + s;
                    let norm = (xv[idx] as f64 - mean) * inv;
                    let dy = gv[idx] as f64 * gamma[ci] as f64;
                    dx[idx] = ((dy - sum_dy / count - norm * sum_dy_norm / count) * inv) as f32;
                }
            }
        }
        Ok(vec![
            Some(upload(x, &dx, x.shape().clone())?),
            Some(upload(inputs[1], &dgamma, inputs[1].shape().clone())?),
            Some(upload(inputs[2], &dbeta, inputs[2].shape().clone())?),
        ])
    }
}

pub fn conv2d<B: Backend>(
    x: &Variable<B>,
    weight: &Variable<B>,
    bias: Option<&Variable<B>>,
    stride: usize,
    padding: usize,
) -> Result<Variable<B>> {
    let mut inputs = vec![x.clone(), weight.clone()];
    if let Some(bias) = bias {
        inputs.push(bias.clone());
    }
    let mut out = apply(Conv2dOp::new(stride, padding), &inputs)?;
    Ok(out.remove(0))
}

pub fn conv_transpose2d<B: Backend>(
    x: &Variable<B>,
    weight: &Variable<B>,
    bias: Option<&Variable<B>>,
    stride: usize,
    padding: usize,
    output_padding: usize,
) -> Result<Variable<B>> {
    let mut inputs = vec![x.clone(), weight.clone()];
    if let Some(bias) = bias {
        inputs.push(bias.clone());
    }
    let mut out = apply(
        ConvTranspose2dOp::new(stride, padding, output_padding),
        &inputs,
    )?;
    Ok(out.remove(0))
}

pub fn max_pool2d<B: Backend>(
    x: &Variable<B>,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<Variable<B>> {
    apply1(Pool2dOp::new(kernel, stride, padding, false), x)
}

pub fn average_pool2d<B: Backend>(
    x: &Variable<B>,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<Variable<B>> {
    apply1(Pool2dOp::new(kernel, stride, padding, true), x)
}

pub fn batch_norm<B: Backend>(
    x: &Variable<B>,
    gamma: &Variable<B>,
    beta: &Variable<B>,
    eps: f64,
) -> Result<Variable<B>> {
    let mut out = apply(
        BatchNormOp::new(eps),
        &[x.clone(), gamma.clone(), beta.clone()],
    )?;
    Ok(out.remove(0))
}

pub struct ReshapeOp {
    shape: Shape,
}

impl ReshapeOp {
    pub fn new(shape: impl Into<Shape>) -> Self {
        Self {
            shape: shape.into(),
        }
    }
}

impl<B: Backend> Op<B> for ReshapeOp {
    const NAME: &'static str = "Reshape";

    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        if inputs.len() != 1 {
            return Err(Error::invalid("reshape expects one input"));
        }
        if inputs[0].numel() != self.shape.numel() {
            return Err(Error::invalid("reshape element count mismatch"));
        }
        let data = host(inputs[0])?;
        Ok(vec![upload(inputs[0], &data, self.shape.clone())?])
    }

    fn backward(
        &self,
        grad: &Tensor<B>,
        inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let data = host(grad)?;
        Ok(vec![Some(upload(
            inputs[0],
            &data,
            inputs[0].shape().clone(),
        )?)])
    }
}

pub struct PermuteOp {
    axes: Vec<usize>,
}

impl PermuteOp {
    pub fn new(axes: Vec<usize>) -> Self {
        Self { axes }
    }
}

fn permute_data(data: &[f32], dims: &[usize], axes: &[usize], inverse: bool) -> Result<Vec<f32>> {
    if dims.len() != axes.len() {
        return Err(Error::invalid("permute rank mismatch"));
    }
    let mut seen = vec![false; axes.len()];
    for &axis in axes {
        if axis >= axes.len() || seen[axis] {
            return Err(Error::invalid("invalid permute axes"));
        }
        seen[axis] = true;
    }
    let out_dims: Vec<usize> = axes.iter().map(|&axis| dims[axis]).collect();
    let out_strides = Shape::new(out_dims.clone()).contiguous_strides();
    let out_strides = out_strides.as_slice();
    let mut out = vec![0f32; data.len()];
    for (linear, &value) in data.iter().enumerate() {
        let mut rest = linear;
        let mut coords = vec![0usize; dims.len()];
        for d in (0..dims.len()).rev() {
            coords[d] = rest % dims[d];
            rest /= dims[d];
        }
        let mut out_index = 0usize;
        for d in 0..axes.len() {
            let out_coord = coords[axes[d]];
            out_index += out_coord * out_strides[d] as usize;
        }
        out[out_index] = value;
    }
    if inverse {
        return Ok(out);
    }
    Ok(out)
}

impl<B: Backend> Op<B> for PermuteOp {
    const NAME: &'static str = "Permute";

    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        if inputs.len() != 1 {
            return Err(Error::invalid("permute expects one input"));
        }
        let dims = inputs[0].shape().dims();
        let data = host(inputs[0])?;
        let out = permute_data(&data, dims, &self.axes, false)?;
        let out_dims: Vec<usize> = self.axes.iter().map(|&axis| dims[axis]).collect();
        Ok(vec![upload(inputs[0], &out, Shape::new(out_dims))?])
    }

    fn backward(
        &self,
        grad: &Tensor<B>,
        inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let mut inverse = vec![0usize; self.axes.len()];
        for (i, &axis) in self.axes.iter().enumerate() {
            inverse[axis] = i;
        }
        let data = host(grad)?;
        let out = permute_data(&data, grad.shape().dims(), &inverse, false)?;
        Ok(vec![Some(upload(
            inputs[0],
            &out,
            inputs[0].shape().clone(),
        )?)])
    }
}

pub struct MultiHeadAttentionOp {
    heads: usize,
    causal: bool,
}

impl MultiHeadAttentionOp {
    pub fn new(heads: usize, causal: bool) -> Self {
        Self { heads, causal }
    }
}

impl<B: Backend> Op<B> for MultiHeadAttentionOp {
    const NAME: &'static str = "MultiHeadAttention";

    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        if inputs.len() != 3 {
            return Err(Error::invalid("multi_head_attention expects q, k, v"));
        }
        let (n, seq, dim) = {
            let d = inputs[0].shape().dims();
            if d.len() != 3 {
                return Err(Error::invalid("attention expects [N, S, D]"));
            }
            (d[0], d[1], d[2])
        };
        if self.heads == 0 || dim % self.heads != 0 {
            return Err(Error::invalid("attention dimension must divide heads"));
        }
        if inputs[1].shape().dims() != [n, seq, dim] || inputs[2].shape().dims() != [n, seq, dim] {
            return Err(Error::invalid("attention q/k/v shape mismatch"));
        }
        let hd = dim / self.heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let q = host(inputs[0])?;
        let k = host(inputs[1])?;
        let v = host(inputs[2])?;
        let mut out = vec![0f32; n * seq * dim];
        let mut scores = vec![0f32; seq * seq];
        for ni in 0..n {
            for hi in 0..self.heads {
                for qi in 0..seq {
                    let mut maxv = f32::NEG_INFINITY;
                    for ki in 0..seq {
                        if self.causal && ki > qi {
                            scores[qi * seq + ki] = f32::NEG_INFINITY;
                            continue;
                        }
                        let mut dot = 0f32;
                        for d in 0..hd {
                            dot += q[(ni * seq + qi) * dim + hi * hd + d]
                                * k[(ni * seq + ki) * dim + hi * hd + d];
                        }
                        let s = dot * scale;
                        scores[qi * seq + ki] = s;
                        maxv = maxv.max(s);
                    }
                    let mut sum = 0f32;
                    for ki in 0..seq {
                        if scores[qi * seq + ki] > f32::NEG_INFINITY {
                            scores[qi * seq + ki] = (scores[qi * seq + ki] - maxv).exp();
                            sum += scores[qi * seq + ki];
                        }
                    }
                    for ki in 0..seq {
                        if sum > 0.0 {
                            scores[qi * seq + ki] /= sum;
                        }
                    }
                    for d in 0..hd {
                        let mut acc = 0f32;
                        for ki in 0..seq {
                            acc += scores[qi * seq + ki] * v[(ni * seq + ki) * dim + hi * hd + d];
                        }
                        out[(ni * seq + qi) * dim + hi * hd + d] = acc;
                    }
                }
            }
        }
        Ok(vec![upload(inputs[0], &out, inputs[0].shape().clone())?])
    }

    fn backward(
        &self,
        grad: &Tensor<B>,
        inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let (n, seq, dim) = {
            let d = inputs[0].shape().dims();
            (d[0], d[1], d[2])
        };
        let hd = dim / self.heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let q = host(inputs[0])?;
        let k = host(inputs[1])?;
        let v = host(inputs[2])?;
        let gout = host(grad)?;
        let mut dq = vec![0f32; q.len()];
        let mut dk = vec![0f32; k.len()];
        let mut dv = vec![0f32; v.len()];
        let mut probs = vec![0f32; seq * seq];
        let mut dp = vec![0f32; seq * seq];
        let mut ds = vec![0f32; seq * seq];
        for ni in 0..n {
            for hi in 0..self.heads {
                for qi in 0..seq {
                    let mut maxv = f32::NEG_INFINITY;
                    for ki in 0..seq {
                        if self.causal && ki > qi {
                            probs[qi * seq + ki] = 0.0;
                            continue;
                        }
                        let mut dot = 0f32;
                        for d in 0..hd {
                            dot += q[(ni * seq + qi) * dim + hi * hd + d]
                                * k[(ni * seq + ki) * dim + hi * hd + d];
                        }
                        let s = dot * scale;
                        probs[qi * seq + ki] = s;
                        maxv = maxv.max(s);
                    }
                    let mut sum = 0f32;
                    for ki in 0..seq {
                        if probs[qi * seq + ki] > f32::NEG_INFINITY {
                            probs[qi * seq + ki] = (probs[qi * seq + ki] - maxv).exp();
                            sum += probs[qi * seq + ki];
                        } else {
                            probs[qi * seq + ki] = 0.0;
                        }
                    }
                    for ki in 0..seq {
                        if sum > 0.0 {
                            probs[qi * seq + ki] /= sum;
                        }
                    }
                }
                for qi in 0..seq {
                    for ki in 0..seq {
                        let mut dot = 0f32;
                        for d in 0..hd {
                            dot += gout[(ni * seq + qi) * dim + hi * hd + d]
                                * v[(ni * seq + ki) * dim + hi * hd + d];
                        }
                        dp[qi * seq + ki] = dot;
                    }
                    let row: f32 = (0..seq)
                        .map(|ki| probs[qi * seq + ki] * dp[qi * seq + ki])
                        .sum();
                    for ki in 0..seq {
                        ds[qi * seq + ki] = probs[qi * seq + ki] * (dp[qi * seq + ki] - row);
                    }
                }
                for ki in 0..seq {
                    for d in 0..hd {
                        let mut acc = 0f32;
                        for qi in 0..seq {
                            acc += probs[qi * seq + ki] * gout[(ni * seq + qi) * dim + hi * hd + d];
                        }
                        dv[(ni * seq + ki) * dim + hi * hd + d] = acc;
                    }
                }
                for qi in 0..seq {
                    for d in 0..hd {
                        let mut acc = 0f32;
                        for ki in 0..seq {
                            acc += ds[qi * seq + ki] * k[(ni * seq + ki) * dim + hi * hd + d];
                        }
                        dq[(ni * seq + qi) * dim + hi * hd + d] = acc * scale;
                    }
                }
                for ki in 0..seq {
                    for d in 0..hd {
                        let mut acc = 0f32;
                        for qi in 0..seq {
                            acc += ds[qi * seq + ki] * q[(ni * seq + qi) * dim + hi * hd + d];
                        }
                        dk[(ni * seq + ki) * dim + hi * hd + d] = acc * scale;
                    }
                }
            }
        }
        Ok(vec![
            Some(upload(inputs[0], &dq, inputs[0].shape().clone())?),
            Some(upload(inputs[1], &dk, inputs[1].shape().clone())?),
            Some(upload(inputs[2], &dv, inputs[2].shape().clone())?),
        ])
    }
}

pub fn reshape<B: Backend>(x: &Variable<B>, shape: impl Into<Shape>) -> Result<Variable<B>> {
    apply1(ReshapeOp::new(shape), x)
}

pub fn permute<B: Backend>(x: &Variable<B>, axes: Vec<usize>) -> Result<Variable<B>> {
    apply1(PermuteOp::new(axes), x)
}

pub fn multi_head_attention<B: Backend>(
    q: &Variable<B>,
    k: &Variable<B>,
    v: &Variable<B>,
    heads: usize,
    causal: bool,
) -> Result<Variable<B>> {
    let mut out = apply(
        MultiHeadAttentionOp::new(heads, causal),
        &[q.clone(), k.clone(), v.clone()],
    )?;
    Ok(out.remove(0))
}
