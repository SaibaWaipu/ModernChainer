use std::sync::Arc;

use modern_chainer_core::{AdamConfig, Backend, DType, Error, Ops, Result, Shape, Tensor};
use modern_chainer_cpu::CpuBackend;

use crate::backend::VulkanBackend;
use crate::gpu::GpuContext;
use crate::shaders;

#[derive(Clone)]
pub struct VulkanOps {
    pub(crate) ctx: Arc<GpuContext>,
}

impl VulkanOps {
    fn backend(&self) -> VulkanBackend {
        VulkanBackend::from_context(Arc::clone(&self.ctx))
    }

    fn empty(&self, shape: &Shape, dtype: DType) -> Result<Tensor<VulkanBackend>> {
        let backend = self.backend();
        let bytes = shape.numel() * dtype.size_in_bytes();
        let storage = backend.alloc(bytes)?;
        Tensor::from_storage(
            backend,
            storage,
            dtype,
            shape.clone(),
            shape.contiguous_strides(),
            0,
        )
    }

    fn uniform(&self, bytes: &[u8]) -> Arc<wgpu::Buffer> {
        let size = bytes.len().max(16).div_ceil(16) * 16;
        let mut padded = vec![0u8; size];
        padded[..bytes.len()].copy_from_slice(bytes);
        let mut uniforms = self
            .ctx
            .uniforms
            .lock()
            .expect("vulkan uniform cache poisoned");
        let buffer = uniforms.entry(size).or_insert_with(|| {
            Arc::new(self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("modern-chainer params"),
                size: size as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        });
        self.ctx.queue.write_buffer(buffer, 0, &padded);
        Arc::clone(buffer)
    }

    fn pipeline(
        &self,
        label: &str,
        code: &str,
        entries: &[wgpu::BindGroupLayoutEntry],
    ) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        if let Some((pipeline, layout)) = self
            .ctx
            .pipelines
            .lock()
            .expect("vulkan pipeline cache poisoned")
            .get(label)
        {
            return (pipeline.clone(), layout.clone());
        }
        let device = &self.ctx.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(code.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        (pipeline, layout)
    }

    fn bind_group(
        &self,
        layout: &wgpu::BindGroupLayout,
        buffers: &[(u32, &wgpu::Buffer)],
    ) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        self.ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("modern-chainer bind group"),
                layout,
                entries: &entries,
            })
    }

    fn dispatch(&self, pipeline: &wgpu::ComputePipeline, bind_group: &wgpu::BindGroup, n: usize) {
        if n == 0 {
            return;
        }
        let workgroups = (n as u64).div_ceil(64);
        let x = workgroups.min(65_535) as u32;
        let y = workgroups.div_ceil(65_535) as u32;
        self.dispatch_2d(pipeline, bind_group, x, y.max(1));
    }

    fn uniform_unique(&self, bytes: &[u8]) -> Arc<wgpu::Buffer> {
        let size = bytes.len().max(16).div_ceil(16) * 16;
        let mut padded = vec![0u8; size];
        padded[..bytes.len()].copy_from_slice(bytes);
        let buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("modern-chainer batched params"),
            size: size as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.ctx.queue.write_buffer(&buffer, 0, &padded);
        Arc::new(buffer)
    }

    fn submit_batch(&self, dispatches: &[PreparedDispatch]) -> Result<()> {
        if dispatches.is_empty() {
            return Ok(());
        }
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("modern-chainer batch"),
            });
        for dispatch in dispatches {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("modern-chainer batch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&dispatch.pipeline);
            pass.set_bind_group(0, &dispatch.group, &[]);
            pass.dispatch_workgroups(dispatch.x, dispatch.y, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    fn dispatch_vec4(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        n: usize,
    ) {
        if n == 0 {
            return;
        }
        let workgroups = (n as u64).div_ceil(256 * 4);
        let x = workgroups.min(65_535) as u32;
        let y = workgroups.div_ceil(65_535) as u32;
        self.dispatch_2d(pipeline, bind_group, x, y.max(1));
    }

    fn dispatch_2d(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        x: u32,
        y: u32,
    ) {
        if x == 0 || y == 0 {
            return;
        }
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("modern-chainer kernel"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("modern-chainer kernel"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    fn binary_vec4(
        &self,
        label: &str,
        code: &str,
        a: &Tensor<VulkanBackend>,
        b: &Tensor<VulkanBackend>,
    ) -> Result<Tensor<VulkanBackend>> {
        if a.shape() != b.shape() {
            return Err(Error::invalid(format!(
                "{label} requires equal shapes, got {} and {}",
                a.shape(),
                b.shape()
            )));
        }
        let out = self.empty(a.shape(), DType::F32)?;
        let (pipeline, layout) = self.pipeline(
            label,
            code,
            &[storage(0, true), storage(1, true), storage(2, false)],
        );
        let group = self.bind_group(
            &layout,
            &[
                (0, a.storage().buffer.as_ref()),
                (1, b.storage().buffer.as_ref()),
                (2, out.storage().buffer.as_ref()),
            ],
        );
        self.dispatch_vec4(&pipeline, &group, out.numel());
        Ok(out)
    }

    fn unary(
        &self,
        label: &str,
        code: &str,
        a: &Tensor<VulkanBackend>,
    ) -> Result<Tensor<VulkanBackend>> {
        let out = self.empty(a.shape(), DType::F32)?;
        let (pipeline, layout) = self.pipeline(label, code, &[storage(0, true), storage(1, false)]);
        let group = self.bind_group(
            &layout,
            &[
                (0, a.storage().buffer.as_ref()),
                (1, out.storage().buffer.as_ref()),
            ],
        );
        self.dispatch(&pipeline, &group, out.numel());
        Ok(out)
    }

    fn unary_vec4(
        &self,
        label: &str,
        code: &str,
        a: &Tensor<VulkanBackend>,
    ) -> Result<Tensor<VulkanBackend>> {
        let out = self.empty(a.shape(), DType::F32)?;
        let (pipeline, layout) = self.pipeline(label, code, &[storage(0, true), storage(1, false)]);
        let group = self.bind_group(
            &layout,
            &[
                (0, a.storage().buffer.as_ref()),
                (1, out.storage().buffer.as_ref()),
            ],
        );
        self.dispatch_vec4(&pipeline, &group, out.numel());
        Ok(out)
    }

    fn to_cpu(&self, tensor: &Tensor<VulkanBackend>) -> Result<Tensor<CpuBackend>> {
        let bytes = tensor.to_bytes()?;
        Tensor::from_bytes(
            CpuBackend::new(),
            &bytes,
            tensor.shape().clone(),
            tensor.dtype(),
        )
    }

    fn upload(&self, tensor: &Tensor<CpuBackend>) -> Result<Tensor<VulkanBackend>> {
        let bytes = tensor.to_bytes()?;
        Tensor::from_bytes(
            self.backend(),
            &bytes,
            tensor.shape().clone(),
            tensor.dtype(),
        )
    }
}

fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

struct PreparedDispatch {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    x: u32,
    y: u32,
}

impl Ops<VulkanBackend> for VulkanOps {
    fn add(
        &self,
        a: &Tensor<VulkanBackend>,
        b: &Tensor<VulkanBackend>,
    ) -> Result<Tensor<VulkanBackend>> {
        if a.shape() == b.shape() {
            return self.binary_vec4("add4", shaders::ADD_VEC, a, b);
        }
        if a.rank() == 2 && b.rank() == 1 && b.shape().dims()[0] == a.shape().dims()[1] {
            let out = self.empty(a.shape(), DType::F32)?;
            let params = self.uniform(&(b.shape().dims()[0] as u32).to_ne_bytes());
            let (pipeline, layout) = self.pipeline(
                "bias_add",
                shaders::BIAS_ADD,
                &[
                    storage(0, true),
                    storage(1, true),
                    storage(2, false),
                    uniform(3),
                ],
            );
            let group = self.bind_group(
                &layout,
                &[
                    (0, a.storage().buffer.as_ref()),
                    (1, b.storage().buffer.as_ref()),
                    (2, out.storage().buffer.as_ref()),
                    (3, &params),
                ],
            );
            self.dispatch(&pipeline, &group, out.numel());
            return Ok(out);
        }
        Err(Error::unsupported("vulkan add with this broadcast pattern"))
    }

    fn sub(
        &self,
        a: &Tensor<VulkanBackend>,
        b: &Tensor<VulkanBackend>,
    ) -> Result<Tensor<VulkanBackend>> {
        self.binary_vec4("sub4", shaders::SUB_VEC, a, b)
    }

    fn mul(
        &self,
        a: &Tensor<VulkanBackend>,
        b: &Tensor<VulkanBackend>,
    ) -> Result<Tensor<VulkanBackend>> {
        self.binary_vec4("mul4", shaders::MUL_VEC, a, b)
    }

    fn div(
        &self,
        a: &Tensor<VulkanBackend>,
        b: &Tensor<VulkanBackend>,
    ) -> Result<Tensor<VulkanBackend>> {
        self.binary_vec4("div4", shaders::DIV_VEC, a, b)
    }

    fn sqrt(&self, a: &Tensor<VulkanBackend>) -> Result<Tensor<VulkanBackend>> {
        self.unary_vec4("sqrt4", shaders::SQRT4, a)
    }

    fn relu(&self, a: &Tensor<VulkanBackend>) -> Result<Tensor<VulkanBackend>> {
        self.unary_vec4("relu4", shaders::RELU4, a)
    }

    fn relu_backward(
        &self,
        x: &Tensor<VulkanBackend>,
        grad: &Tensor<VulkanBackend>,
    ) -> Result<Tensor<VulkanBackend>> {
        if x.shape() != grad.shape() {
            return Err(Error::shape_mismatch(
                "vulkan relu backward",
                x.shape().clone(),
                grad.shape().clone(),
            ));
        }
        let out = self.empty(x.shape(), DType::F32)?;
        let (pipeline, layout) = self.pipeline(
            "relu_backward",
            shaders::RELU_BACKWARD,
            &[storage(0, true), storage(1, true), storage(2, false)],
        );
        let group = self.bind_group(
            &layout,
            &[
                (0, x.storage().buffer.as_ref()),
                (1, grad.storage().buffer.as_ref()),
                (2, out.storage().buffer.as_ref()),
            ],
        );
        self.dispatch(&pipeline, &group, out.numel());
        Ok(out)
    }

    fn matmul(
        &self,
        a: &Tensor<VulkanBackend>,
        b: &Tensor<VulkanBackend>,
    ) -> Result<Tensor<VulkanBackend>> {
        if a.rank() != 2 || b.rank() != 2 {
            return Err(Error::invalid("vulkan matmul expects 2-D tensors"));
        }
        let (m, k) = (a.shape().dims()[0], a.shape().dims()[1]);
        let (k2, n) = (b.shape().dims()[0], b.shape().dims()[1]);
        if k != k2 {
            return Err(Error::shape_mismatch(
                "vulkan matmul inner dimension",
                a.shape().clone(),
                b.shape().clone(),
            ));
        }
        self.matmul_with(a, b, m, k, n, false)
    }

    fn linear(
        &self,
        x: &Tensor<VulkanBackend>,
        weight: &Tensor<VulkanBackend>,
        bias: Option<&Tensor<VulkanBackend>>,
    ) -> Result<Tensor<VulkanBackend>> {
        if x.rank() != 2 || weight.rank() != 2 {
            return Err(Error::invalid("vulkan linear expects 2-D tensors"));
        }
        let (batch, in_features) = (x.shape().dims()[0], x.shape().dims()[1]);
        let (out_features, k2) = (weight.shape().dims()[0], weight.shape().dims()[1]);
        if in_features != k2 {
            return Err(Error::shape_mismatch(
                "vulkan linear inner dimension",
                x.shape().clone(),
                weight.shape().clone(),
            ));
        }
        if batch <= 128 {
            let (matmul, y) =
                self.prepare_matmul_with(x, weight, batch, in_features, out_features, true, true)?;
            return match bias {
                Some(bias) => {
                    let (bias_dispatch, out) = self.prepare_bias_add(&y, bias)?;
                    self.submit_batch(&[matmul, bias_dispatch])?;
                    Ok(out)
                }
                None => {
                    self.submit_batch(&[matmul])?;
                    Ok(y)
                }
            };
        }
        match bias {
            Some(bias) => self.matmul_nt_bias(x, weight, bias, batch, in_features, out_features),
            None => self.matmul_with(x, weight, batch, in_features, out_features, true),
        }
    }

    fn linear_backward(
        &self,
        grad: &Tensor<VulkanBackend>,
        x: &Tensor<VulkanBackend>,
        weight: &Tensor<VulkanBackend>,
    ) -> Result<(Tensor<VulkanBackend>, Tensor<VulkanBackend>)> {
        let (batch, in_features) = (x.shape().dims()[0], x.shape().dims()[1]);
        let out_features = weight.shape().dims()[0];
        let (grad_x_dispatch, grad_x) =
            self.prepare_matmul_with(grad, weight, batch, out_features, in_features, false, true)?;
        let (grad_w_dispatch, grad_w) =
            self.prepare_matmul_tn(grad, x, out_features, batch, in_features, true)?;
        self.submit_batch(&[grad_x_dispatch, grad_w_dispatch])?;
        Ok((grad_x, grad_w))
    }

    fn transpose2d(&self, a: &Tensor<VulkanBackend>) -> Result<Tensor<VulkanBackend>> {
        if a.rank() != 2 {
            return Err(Error::invalid("vulkan transpose2d expects a 2-D tensor"));
        }
        let (rows, cols) = (a.shape().dims()[0], a.shape().dims()[1]);
        let out = self.empty(&Shape::new([cols, rows]), DType::F32)?;
        let mut params = Vec::with_capacity(16);
        for value in [rows as u32, cols as u32, 0, 0] {
            params.extend_from_slice(&value.to_ne_bytes());
        }
        let dims = self.uniform(&params);
        let (pipeline, layout) = self.pipeline(
            "transpose",
            shaders::TRANSPOSE,
            &[storage(0, true), storage(1, false), uniform(2)],
        );
        let group = self.bind_group(
            &layout,
            &[
                (0, a.storage().buffer.as_ref()),
                (1, out.storage().buffer.as_ref()),
                (2, &dims),
            ],
        );
        self.dispatch_2d(
            &pipeline,
            &group,
            (cols as u64).div_ceil(16) as u32,
            (rows as u64).div_ceil(16) as u32,
        );
        Ok(out)
    }

    fn sum_all(&self, a: &Tensor<VulkanBackend>) -> Result<Tensor<VulkanBackend>> {
        self.sum_all_gpu(a)
    }

    fn sum_to(&self, a: &Tensor<VulkanBackend>, shape: &Shape) -> Result<Tensor<VulkanBackend>> {
        if shape.rank() == 0 {
            return self.sum_all_gpu(a);
        }
        if a.shape() == shape {
            return Ok(a.clone());
        }
        if a.rank() == 2 && shape.dims() == [a.shape().dims()[1]] {
            return self.sum_to_bias(a);
        }
        if let Ok(result) = self.sum_to_generic(a, shape) {
            return Ok(result);
        }
        let cpu = self.to_cpu(a)?;
        let backend = CpuBackend::new();
        let result = backend.ops().sum_to(&cpu, shape)?;
        self.upload(&result)
    }

    fn silu(&self, a: &Tensor<VulkanBackend>) -> Result<Tensor<VulkanBackend>> {
        self.unary("silu", shaders::SILU, a)
    }

    fn gelu(&self, a: &Tensor<VulkanBackend>) -> Result<Tensor<VulkanBackend>> {
        self.unary("gelu", shaders::GELU, a)
    }

    fn scale(&self, a: &Tensor<VulkanBackend>, factor: f64) -> Result<Tensor<VulkanBackend>> {
        let out = self.empty(a.shape(), DType::F32)?;
        let mut raw = Vec::with_capacity(16);
        raw.extend_from_slice(&(factor as f32).to_ne_bytes());
        raw.extend_from_slice(&[0u8; 12]);
        let params = self.uniform(&raw);
        let (pipeline, layout) = self.pipeline(
            "scale4",
            shaders::SCALE4,
            &[storage(0, true), storage(1, false), uniform(2)],
        );
        let group = self.bind_group(
            &layout,
            &[
                (0, a.storage().buffer.as_ref()),
                (1, out.storage().buffer.as_ref()),
                (2, &params),
            ],
        );
        self.dispatch_vec4(&pipeline, &group, out.numel());
        Ok(out)
    }

    fn ones_like(&self, a: &Tensor<VulkanBackend>) -> Result<Tensor<VulkanBackend>> {
        self.fill(a, 1.0)
    }

    fn zeros_like(&self, a: &Tensor<VulkanBackend>) -> Result<Tensor<VulkanBackend>> {
        self.fill(a, 0.0)
    }

    fn to_scalar(&self, a: &Tensor<VulkanBackend>) -> Result<f64> {
        if !a.is_scalar() {
            return Err(Error::invalid(format!(
                "expected scalar tensor, got {}",
                a.shape()
            )));
        }
        let bytes = self
            .backend()
            .read_buffer(a.storage().buffer.as_ref(), a.dtype().size_in_bytes())?;
        match a.dtype() {
            DType::F32 => Ok(f32::from_ne_bytes(bytes[..4].try_into().unwrap()) as f64),
            DType::F64 => Ok(f64::from_ne_bytes(bytes[..8].try_into().unwrap())),
            DType::I64 => Ok(i64::from_ne_bytes(bytes[..8].try_into().unwrap()) as f64),
            DType::I32 => Ok(i32::from_ne_bytes(bytes[..4].try_into().unwrap()) as f64),
            other => Err(Error::UnsupportedDType(other)),
        }
    }

    fn adam_update(
        &self,
        param: &Tensor<VulkanBackend>,
        grad: &Tensor<VulkanBackend>,
        m: &Tensor<VulkanBackend>,
        v: &Tensor<VulkanBackend>,
        cfg: &AdamConfig,
    ) -> Result<(
        Tensor<VulkanBackend>,
        Tensor<VulkanBackend>,
        Tensor<VulkanBackend>,
    )> {
        let n = param.numel();
        if grad.numel() != n || m.numel() != n || v.numel() != n {
            return Err(Error::invalid("vulkan adam shape mismatch"));
        }
        let new_p = self.empty(param.shape(), DType::F32)?;
        let new_m = self.empty(m.shape(), DType::F32)?;
        let new_v = self.empty(v.shape(), DType::F32)?;
        let bc1 = 1.0 - cfg.beta1.powi(cfg.step as i32);
        let bc2 = 1.0 - cfg.beta2.powi(cfg.step as i32);
        let mut raw = Vec::with_capacity(32);
        for value in [
            cfg.lr as f32,
            cfg.beta1 as f32,
            cfg.beta2 as f32,
            cfg.eps as f32,
            bc1 as f32,
            bc2 as f32,
        ] {
            raw.extend_from_slice(&value.to_ne_bytes());
        }
        raw.extend_from_slice(&(n as u32).to_ne_bytes());
        raw.extend_from_slice(&0u32.to_ne_bytes());
        let params = self.uniform(&raw);
        let (pipeline, layout) = self.pipeline(
            "adam",
            shaders::ADAM,
            &[
                storage(0, true),
                storage(1, true),
                storage(2, true),
                storage(3, true),
                storage(4, false),
                storage(5, false),
                storage(6, false),
                uniform(7),
            ],
        );
        let group = self.bind_group(
            &layout,
            &[
                (0, param.storage().buffer.as_ref()),
                (1, grad.storage().buffer.as_ref()),
                (2, m.storage().buffer.as_ref()),
                (3, v.storage().buffer.as_ref()),
                (4, new_p.storage().buffer.as_ref()),
                (5, new_m.storage().buffer.as_ref()),
                (6, new_v.storage().buffer.as_ref()),
                (7, &params),
            ],
        );
        self.dispatch(&pipeline, &group, n);
        Ok((new_p, new_m, new_v))
    }

    fn softmax_cross_entropy(
        &self,
        logits: &Tensor<VulkanBackend>,
        target: &Tensor<VulkanBackend>,
    ) -> Result<Tensor<VulkanBackend>> {
        if logits.rank() != 2 {
            return Err(Error::invalid("vulkan softmax CE expects 2-D logits"));
        }
        let (rows, classes) = (logits.shape().dims()[0], logits.shape().dims()[1]);
        if target.numel() != rows {
            return Err(Error::invalid("vulkan softmax CE target size mismatch"));
        }
        let losses = self.empty(&Shape::new([rows]), DType::F32)?;
        let mut raw = Vec::with_capacity(16);
        raw.extend_from_slice(&(rows as u32).to_ne_bytes());
        raw.extend_from_slice(&(classes as u32).to_ne_bytes());
        raw.extend_from_slice(&0u32.to_ne_bytes());
        raw.extend_from_slice(&0u32.to_ne_bytes());
        let params = self.uniform(&raw);
        let (pipeline, layout) = self.pipeline(
            "softmax_ce_forward",
            shaders::SOFTMAX_CE_FORWARD,
            &[
                storage(0, true),
                storage(1, true),
                storage(2, false),
                uniform(3),
            ],
        );
        let group = self.bind_group(
            &layout,
            &[
                (0, logits.storage().buffer.as_ref()),
                (1, target.storage().buffer.as_ref()),
                (2, losses.storage().buffer.as_ref()),
                (3, &params),
            ],
        );
        self.dispatch_2d(&pipeline, &group, rows as u32, 1);
        self.sum_all_gpu(&losses)
    }

    fn softmax_cross_entropy_backward(
        &self,
        logits: &Tensor<VulkanBackend>,
        target: &Tensor<VulkanBackend>,
    ) -> Result<Tensor<VulkanBackend>> {
        if logits.rank() != 2 {
            return Err(Error::invalid("vulkan softmax CE expects 2-D logits"));
        }
        let (rows, classes) = (logits.shape().dims()[0], logits.shape().dims()[1]);
        if target.numel() != rows {
            return Err(Error::invalid("vulkan softmax CE target size mismatch"));
        }
        let grad = self.empty(logits.shape(), DType::F32)?;
        let mut raw = Vec::with_capacity(16);
        raw.extend_from_slice(&(rows as u32).to_ne_bytes());
        raw.extend_from_slice(&(classes as u32).to_ne_bytes());
        raw.extend_from_slice(&0u32.to_ne_bytes());
        raw.extend_from_slice(&0u32.to_ne_bytes());
        let params = self.uniform(&raw);
        let (pipeline, layout) = self.pipeline(
            "softmax_ce_backward",
            shaders::SOFTMAX_CE_BACKWARD,
            &[
                storage(0, true),
                storage(1, true),
                storage(2, false),
                uniform(3),
            ],
        );
        let group = self.bind_group(
            &layout,
            &[
                (0, logits.storage().buffer.as_ref()),
                (1, target.storage().buffer.as_ref()),
                (2, grad.storage().buffer.as_ref()),
                (3, &params),
            ],
        );
        self.dispatch_2d(&pipeline, &group, rows as u32, 1);
        Ok(grad)
    }
}

impl VulkanOps {
    fn prepare_bias_add(
        &self,
        a: &Tensor<VulkanBackend>,
        bias: &Tensor<VulkanBackend>,
    ) -> Result<(PreparedDispatch, Tensor<VulkanBackend>)> {
        let out = self.empty(a.shape(), DType::F32)?;
        let cols = a.shape().dims()[a.rank() - 1];
        let params = self.uniform_unique(&(cols as u32).to_ne_bytes());
        let (pipeline, layout) = self.pipeline(
            "bias_add",
            shaders::BIAS_ADD,
            &[
                storage(0, true),
                storage(1, true),
                storage(2, false),
                uniform(3),
            ],
        );
        let group = self.bind_group(
            &layout,
            &[
                (0, a.storage().buffer.as_ref()),
                (1, bias.storage().buffer.as_ref()),
                (2, out.storage().buffer.as_ref()),
                (3, &params),
            ],
        );
        let total = out.numel() as u64;
        let x = total.div_ceil(64).min(65_535) as u32;
        let y = total.div_ceil(64).div_ceil(65_535) as u32;
        Ok((
            PreparedDispatch {
                pipeline,
                group,
                x: x.max(1),
                y: y.max(1),
            },
            out,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_matmul_with(
        &self,
        a: &Tensor<VulkanBackend>,
        b: &Tensor<VulkanBackend>,
        m: usize,
        k: usize,
        n: usize,
        transposed_b: bool,
        unique_uniform: bool,
    ) -> Result<(PreparedDispatch, Tensor<VulkanBackend>)> {
        let out = self.empty(&Shape::new([m, n]), DType::F32)?;
        let mut params = Vec::with_capacity(16);
        for value in [m as u32, k as u32, n as u32, 0] {
            params.extend_from_slice(&value.to_ne_bytes());
        }
        let dims = if unique_uniform {
            self.uniform_unique(&params)
        } else {
            self.uniform(&params)
        };
        let small = m <= 128 || n <= 128;
        let (label, code) = match (transposed_b, small) {
            (true, true) => ("matmul_nt_small", shaders::MATMUL_NT_SMALL),
            (false, true) => ("matmul_small", shaders::MATMUL_SMALL),
            (true, false) => ("matmul_nt", shaders::MATMUL_NT),
            (false, false) => ("matmul", shaders::MATMUL),
        };
        let (pipeline, layout) = self.pipeline(
            label,
            code,
            &[
                storage(0, true),
                storage(1, true),
                storage(2, false),
                uniform(3),
            ],
        );
        let group = self.bind_group(
            &layout,
            &[
                (0, a.storage().buffer.as_ref()),
                (1, b.storage().buffer.as_ref()),
                (2, out.storage().buffer.as_ref()),
                (3, &dims),
            ],
        );
        let tile = if small { 16u64 } else { 32u64 };
        let x = (n as u64).div_ceil(tile).min(65_535) as u32;
        let y = (m as u64).div_ceil(tile).min(65_535) as u32;
        Ok((
            PreparedDispatch {
                pipeline,
                group,
                x: x.max(1),
                y: y.max(1),
            },
            out,
        ))
    }

    fn matmul_with(
        &self,
        a: &Tensor<VulkanBackend>,
        b: &Tensor<VulkanBackend>,
        m: usize,
        k: usize,
        n: usize,
        transposed_b: bool,
    ) -> Result<Tensor<VulkanBackend>> {
        let (dispatch, out) = self.prepare_matmul_with(a, b, m, k, n, transposed_b, false)?;
        self.submit_batch(&[dispatch])?;
        Ok(out)
    }

    fn matmul_nt_bias(
        &self,
        a: &Tensor<VulkanBackend>,
        b: &Tensor<VulkanBackend>,
        bias: &Tensor<VulkanBackend>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Tensor<VulkanBackend>> {
        let out = self.empty(&Shape::new([m, n]), DType::F32)?;
        let mut params = Vec::with_capacity(16);
        for value in [m as u32, k as u32, n as u32, 0] {
            params.extend_from_slice(&value.to_ne_bytes());
        }
        let dims = self.uniform(&params);
        let (pipeline, layout) = self.pipeline(
            "matmul_nt_bias",
            shaders::MATMUL_NT_BIAS,
            &[
                storage(0, true),
                storage(1, true),
                storage(2, false),
                storage(3, true),
                uniform(4),
            ],
        );
        let group = self.bind_group(
            &layout,
            &[
                (0, a.storage().buffer.as_ref()),
                (1, b.storage().buffer.as_ref()),
                (2, out.storage().buffer.as_ref()),
                (3, bias.storage().buffer.as_ref()),
                (4, &dims),
            ],
        );
        let x = (n as u64).div_ceil(32).min(65_535) as u32;
        let y = (m as u64).div_ceil(32).min(65_535) as u32;
        self.dispatch_2d(&pipeline, &group, x.max(1), y.max(1));
        Ok(out)
    }

    fn prepare_matmul_tn(
        &self,
        a: &Tensor<VulkanBackend>,
        b: &Tensor<VulkanBackend>,
        m: usize,
        k: usize,
        n: usize,
        unique_uniform: bool,
    ) -> Result<(PreparedDispatch, Tensor<VulkanBackend>)> {
        let out = self.empty(&Shape::new([m, n]), DType::F32)?;
        let mut params = Vec::with_capacity(16);
        for value in [m as u32, k as u32, n as u32, 0] {
            params.extend_from_slice(&value.to_ne_bytes());
        }
        let dims = if unique_uniform {
            self.uniform_unique(&params)
        } else {
            self.uniform(&params)
        };
        let (pipeline, layout) = self.pipeline(
            "matmul_tn",
            shaders::MATMUL_TN,
            &[
                storage(0, true),
                storage(1, true),
                storage(2, false),
                uniform(3),
            ],
        );
        let group = self.bind_group(
            &layout,
            &[
                (0, a.storage().buffer.as_ref()),
                (1, b.storage().buffer.as_ref()),
                (2, out.storage().buffer.as_ref()),
                (3, &dims),
            ],
        );
        let x = (n as u64).div_ceil(32).min(65_535) as u32;
        let y = (m as u64).div_ceil(32).min(65_535) as u32;
        Ok((
            PreparedDispatch {
                pipeline,
                group,
                x: x.max(1),
                y: y.max(1),
            },
            out,
        ))
    }

    fn prepare_sum_partial(
        &self,
        input: &Tensor<VulkanBackend>,
        len: usize,
    ) -> Result<(PreparedDispatch, Tensor<VulkanBackend>)> {
        let groups = len.div_ceil(256);
        let output = self.empty(&Shape::new([groups]), DType::F32)?;
        let params = self.uniform_unique(&(len as u32).to_ne_bytes());
        let (pipeline, layout) = self.pipeline(
            "sum_partial",
            shaders::SUM_PARTIAL,
            &[storage(0, true), storage(1, false), uniform(2)],
        );
        let group = self.bind_group(
            &layout,
            &[
                (0, input.storage().buffer.as_ref()),
                (1, output.storage().buffer.as_ref()),
                (2, &params),
            ],
        );
        Ok((
            PreparedDispatch {
                pipeline,
                group,
                x: groups as u32,
                y: 1,
            },
            output,
        ))
    }

    fn prepare_sum_final(
        &self,
        input: &Tensor<VulkanBackend>,
        len: usize,
    ) -> Result<(PreparedDispatch, Tensor<VulkanBackend>)> {
        let output = self.empty(&Shape::scalar(), DType::F32)?;
        let params = self.uniform_unique(&(len as u32).to_ne_bytes());
        let (pipeline, layout) = self.pipeline(
            "sum_final",
            shaders::SUM_FINAL,
            &[storage(0, true), storage(1, false), uniform(2)],
        );
        let group = self.bind_group(
            &layout,
            &[
                (0, input.storage().buffer.as_ref()),
                (1, output.storage().buffer.as_ref()),
                (2, &params),
            ],
        );
        Ok((
            PreparedDispatch {
                pipeline,
                group,
                x: 1,
                y: 1,
            },
            output,
        ))
    }

    fn sum_all_gpu(&self, a: &Tensor<VulkanBackend>) -> Result<Tensor<VulkanBackend>> {
        let mut current = a.clone();
        let mut len = a.numel();
        let mut dispatches = Vec::new();
        let mut keep = Vec::new();
        while len > 256 {
            let (dispatch, next) = self.prepare_sum_partial(&current, len)?;
            dispatches.push(dispatch);
            keep.push(current);
            current = next;
            len = len.div_ceil(256);
        }
        let (dispatch, output) = self.prepare_sum_final(&current, len)?;
        dispatches.push(dispatch);
        keep.push(current);
        self.submit_batch(&dispatches)?;
        Ok(output)
    }

    fn sum_to_bias(&self, a: &Tensor<VulkanBackend>) -> Result<Tensor<VulkanBackend>> {
        let (rows, cols) = (a.shape().dims()[0], a.shape().dims()[1]);
        let output = self.empty(&Shape::new([cols]), DType::F32)?;
        let mut raw = Vec::with_capacity(16);
        raw.extend_from_slice(&(rows as u32).to_ne_bytes());
        raw.extend_from_slice(&(cols as u32).to_ne_bytes());
        raw.extend_from_slice(&0u32.to_ne_bytes());
        raw.extend_from_slice(&0u32.to_ne_bytes());
        let params = self.uniform(&raw);
        let (pipeline, layout) = self.pipeline(
            "sum_to_bias",
            shaders::SUM_TO_BIAS,
            &[storage(0, true), storage(1, false), uniform(2)],
        );
        let group = self.bind_group(
            &layout,
            &[
                (0, a.storage().buffer.as_ref()),
                (1, output.storage().buffer.as_ref()),
                (2, &params),
            ],
        );
        self.dispatch_2d(&pipeline, &group, cols.div_ceil(256) as u32, 1);
        Ok(output)
    }

    fn sum_to_generic(
        &self,
        a: &Tensor<VulkanBackend>,
        shape: &Shape,
    ) -> Result<Tensor<VulkanBackend>> {
        let in_dims = a.shape().dims();
        let out_dims = shape.dims();
        if out_dims.len() > in_dims.len() || out_dims.len() > 8 || in_dims.len() > 8 {
            return Err(Error::invalid("vulkan sum_to rank is not supported"));
        }
        let offset = in_dims.len() - out_dims.len();
        let output_len: usize = out_dims.iter().product();
        if output_len == 0 {
            return Err(Error::invalid("vulkan sum_to empty output"));
        }
        for (d, &dim) in out_dims.iter().enumerate() {
            let source = in_dims[offset + d];
            if dim != 1 && dim != source {
                return Err(Error::invalid("vulkan sum_to incompatible shape"));
            }
        }
        let in_strides = a.strides().as_slice();
        let mut red_dims = Vec::new();
        for d in 0..offset {
            red_dims.push(d);
        }
        for d in 0..out_dims.len() {
            if out_dims[d] == 1 && in_dims[offset + d] > 1 {
                red_dims.push(offset + d);
            }
        }
        if red_dims.len() > 8 {
            return Err(Error::invalid(
                "vulkan sum_to reduction rank is not supported",
            ));
        }
        let red_count: usize = red_dims.iter().map(|&d| in_dims[d]).product();
        let mut meta = [0u32; 40];
        meta[0] = out_dims.len() as u32;
        meta[1] = red_dims.len() as u32;
        meta[2] = offset as u32;
        meta[3] = red_count as u32;
        meta[4] = output_len as u32;
        for d in 0..in_dims.len() {
            meta[8 + d] = in_dims[d] as u32;
            meta[16 + d] = in_strides[d] as u32;
        }
        for d in 0..out_dims.len() {
            meta[24 + d] = out_dims[d] as u32;
        }
        for (i, &d) in red_dims.iter().enumerate() {
            meta[32 + i] = d as u32;
        }
        let bytes: Vec<u8> = meta.iter().flat_map(|value| value.to_ne_bytes()).collect();
        let meta =
            Tensor::from_bytes(self.backend(), &bytes, Shape::new([bytes.len()]), DType::U8)?;
        let output = self.empty(shape, DType::F32)?;
        let (pipeline, layout) = self.pipeline(
            "sum_to_generic",
            shaders::SUM_TO_GENERIC,
            &[storage(0, true), storage(1, false), storage(2, true)],
        );
        let group = self.bind_group(
            &layout,
            &[
                (0, a.storage().buffer.as_ref()),
                (1, output.storage().buffer.as_ref()),
                (2, meta.storage().buffer.as_ref()),
            ],
        );
        self.dispatch(&pipeline, &group, output_len);
        Ok(output)
    }

    fn fill(&self, a: &Tensor<VulkanBackend>, value: f32) -> Result<Tensor<VulkanBackend>> {
        let out = self.empty(a.shape(), DType::F32)?;
        let mut raw = Vec::with_capacity(16);
        raw.extend_from_slice(&value.to_ne_bytes());
        raw.extend_from_slice(&[0u8; 12]);
        let params = self.uniform(&raw);
        let (pipeline, layout) =
            self.pipeline("fill4", shaders::FILL4, &[storage(0, false), uniform(1)]);
        let group = self.bind_group(&layout, &[(0, out.storage().buffer.as_ref()), (1, &params)]);
        self.dispatch_vec4(&pipeline, &group, out.numel());
        Ok(out)
    }
}
