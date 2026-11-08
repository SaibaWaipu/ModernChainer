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
        let vec4_count = (n as u64).div_ceil(4);
        let workgroups = vec4_count.div_ceil(256);
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

    fn check_f32(&self, tensor: &Tensor<VulkanBackend>, op: &str) -> Result<()> {
        if tensor.dtype() != DType::F32 {
            return Err(Error::UnsupportedDType(tensor.dtype()));
        }
        if !tensor.is_contiguous() {
            return Err(Error::invalid(format!("{op} requires contiguous tensor")));
        }
        Ok(())
    }

    fn cpu_binary(
        &self,
        op: &str,
        a: &Tensor<VulkanBackend>,
        b: &Tensor<VulkanBackend>,
        f: impl FnOnce(&Tensor<CpuBackend>, &Tensor<CpuBackend>) -> Result<Tensor<CpuBackend>>,
    ) -> Result<Tensor<VulkanBackend>> {
        let ac = self.to_cpu(a)?;
        let bc = self.to_cpu(b)?;
        let out = f(&ac, &bc)?;
        let _ = op;
        self.upload(&out)
    }

    fn binary_broadcast(
        &self,
        label: &str,
        a: &Tensor<VulkanBackend>,
        b: &Tensor<VulkanBackend>,
        cpu_f: impl FnOnce(&Tensor<CpuBackend>, &Tensor<CpuBackend>) -> Result<Tensor<CpuBackend>>,
    ) -> Result<Tensor<VulkanBackend>> {
        self.check_f32(a, label)?;
        self.check_f32(b, label)?;
        if a.shape() == b.shape() {
            return match label {
                "add" => self.binary_vec4("add4", shaders::ADD_VEC, a, b),
                "sub" => self.binary_vec4("sub4", shaders::SUB_VEC, a, b),
                "mul" => self.binary_vec4("mul4", shaders::MUL_VEC, a, b),
                _ => self.binary_vec4("div4", shaders::DIV_VEC, a, b),
            };
        }
        if a.rank() == 2 && b.rank() == 1 && b.shape().dims()[0] == a.shape().dims()[1] {
            return self.bias_broadcast(label, a, b);
        }
        if b.rank() == 2 && a.rank() == 1 && a.shape().dims()[0] == b.shape().dims()[1] {
            return self.cpu_binary(label, a, b, cpu_f);
        }
        if a.numel() == 1 || b.numel() == 1 {
            return self.cpu_binary(label, a, b, cpu_f);
        }
        self.cpu_binary(label, a, b, cpu_f)
    }

    fn bias_broadcast(
        &self,
        label: &str,
        a: &Tensor<VulkanBackend>,
        b: &Tensor<VulkanBackend>,
    ) -> Result<Tensor<VulkanBackend>> {
        let out = self.empty(a.shape(), DType::F32)?;
        let params = self.uniform(&(b.shape().dims()[0] as u32).to_ne_bytes());
        let (code, lbl) = match label {
            "add" => (shaders::BIAS_ADD, "bias_add"),
            "sub" => {
                return self.cpu_binary(label, a, b, |x, y| CpuBackend::new().ops().sub(x, y));
            }
            "mul" => {
                return self.cpu_binary(label, a, b, |x, y| CpuBackend::new().ops().mul(x, y));
            }
            _ => {
                return self.cpu_binary(label, a, b, |x, y| CpuBackend::new().ops().div(x, y));
            }
        };
        let (pipeline, layout) = self.pipeline(
            lbl,
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
                (3, &params),
            ],
        );
        self.dispatch(&pipeline, &group, out.numel());
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
        self.binary_broadcast("add", a, b, |x, y| CpuBackend::new().ops().add(x, y))
    }

    fn sub(
        &self,
        a: &Tensor<VulkanBackend>,
        b: &Tensor<VulkanBackend>,
    ) -> Result<Tensor<VulkanBackend>> {
        self.binary_broadcast("sub", a, b, |x, y| CpuBackend::new().ops().sub(x, y))
    }

    fn mul(
        &self,
        a: &Tensor<VulkanBackend>,
        b: &Tensor<VulkanBackend>,
    ) -> Result<Tensor<VulkanBackend>> {
        self.binary_broadcast("mul", a, b, |x, y| CpuBackend::new().ops().mul(x, y))
    }

    fn div(
        &self,
        a: &Tensor<VulkanBackend>,
        b: &Tensor<VulkanBackend>,
    ) -> Result<Tensor<VulkanBackend>> {
        self.binary_broadcast("div", a, b, |x, y| CpuBackend::new().ops().div(x, y))
    }

    fn exp(&self, a: &Tensor<VulkanBackend>) -> Result<Tensor<VulkanBackend>> {
        self.check_f32(a, "exp")?;
        self.unary_vec4("exp4", shaders::EXP4, a)
    }

    fn log(&self, a: &Tensor<VulkanBackend>) -> Result<Tensor<VulkanBackend>> {
        self.check_f32(a, "log")?;
        self.unary_vec4("log4", shaders::LOG4, a)
    }

    fn sigmoid(&self, a: &Tensor<VulkanBackend>) -> Result<Tensor<VulkanBackend>> {
        self.check_f32(a, "sigmoid")?;
        self.unary_vec4("sigmoid4", shaders::SIGMOID4, a)
    }

    fn sqrt(&self, a: &Tensor<VulkanBackend>) -> Result<Tensor<VulkanBackend>> {
        self.check_f32(a, "sqrt")?;
        self.unary_vec4("sqrt4", shaders::SQRT4, a)
    }

    fn relu(&self, a: &Tensor<VulkanBackend>) -> Result<Tensor<VulkanBackend>> {
        self.check_f32(a, "relu")?;
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
        self.check_f32(x, "relu_backward")?;
        self.check_f32(grad, "relu_backward")?;
        let out = self.empty(x.shape(), DType::F32)?;
        let (pipeline, layout) = self.pipeline(
            "relu_backward4",
            shaders::RELU_BACKWARD4,
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
        self.dispatch_vec4(&pipeline, &group, out.numel());
        Ok(out)
    }

    fn matmul(
        &self,
        a: &Tensor<VulkanBackend>,
        b: &Tensor<VulkanBackend>,
    ) -> Result<Tensor<VulkanBackend>> {
        self.check_f32(a, "matmul")?;
        self.check_f32(b, "matmul")?;
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
        if m == 0 || n == 0 || k == 0 {
            return self.empty(&Shape::new([m, n]), DType::F32);
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
            if let Some(bias) = bias {
                let (dispatch, out) = self.prepare_matmul_nt_small_bias(
                    x,
                    weight,
                    bias,
                    batch,
                    in_features,
                    out_features,
                )?;
                self.submit_batch(&[dispatch])?;
                return Ok(out);
            }
            let (dispatch, y) =
                self.prepare_matmul_with(x, weight, batch, in_features, out_features, true, true)?;
            self.submit_batch(&[dispatch])?;
            return Ok(y);
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

    fn linear_backward_with_bias(
        &self,
        grad: &Tensor<VulkanBackend>,
        x: &Tensor<VulkanBackend>,
        weight: &Tensor<VulkanBackend>,
        bias_shape: &Shape,
    ) -> Result<(
        Tensor<VulkanBackend>,
        Tensor<VulkanBackend>,
        Tensor<VulkanBackend>,
    )> {
        let (batch, in_features) = (x.shape().dims()[0], x.shape().dims()[1]);
        let out_features = weight.shape().dims()[0];
        let (grad_x_dispatch, grad_x) =
            self.prepare_matmul_with(grad, weight, batch, out_features, in_features, false, true)?;
        let (grad_w_dispatch, grad_w) =
            self.prepare_matmul_tn(grad, x, out_features, batch, in_features, true)?;
        if bias_shape.dims() != [out_features] {
            self.submit_batch(&[grad_x_dispatch, grad_w_dispatch])?;
            let grad_b = self.sum_to(grad, bias_shape)?;
            return Ok((grad_x, grad_w, grad_b));
        }
        let (grad_b_dispatch, grad_b) = self.prepare_sum_to_bias(grad)?;
        self.submit_batch(&[grad_x_dispatch, grad_w_dispatch, grad_b_dispatch])?;
        Ok((grad_x, grad_w, grad_b))
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
        self.check_f32(a, "silu")?;
        self.unary_vec4("silu4", shaders::SILU4, a)
    }

    fn gelu(&self, a: &Tensor<VulkanBackend>) -> Result<Tensor<VulkanBackend>> {
        self.check_f32(a, "gelu")?;
        self.unary_vec4("gelu4", shaders::GELU4, a)
    }

    fn linear_activation(
        &self,
        x: &Tensor<VulkanBackend>,
        weight: &Tensor<VulkanBackend>,
        bias: Option<&Tensor<VulkanBackend>>,
        activation: modern_chainer_core::Activation,
    ) -> Result<Tensor<VulkanBackend>> {
        self.check_f32(x, "linear_activation")?;
        self.check_f32(weight, "linear_activation")?;
        if let Some(b) = bias {
            self.check_f32(b, "linear_activation")?;
        }
        if x.rank() != 2 || weight.rank() != 2 {
            return Err(Error::invalid("linear_activation expects 2-D tensors"));
        }
        let (batch, in_features) = (x.shape().dims()[0], x.shape().dims()[1]);
        let (out_features, k2) = (weight.shape().dims()[0], weight.shape().dims()[1]);
        if in_features != k2 {
            return Err(Error::shape_mismatch(
                "linear_activation inner",
                x.shape().clone(),
                weight.shape().clone(),
            ));
        }
        if batch == 0 || out_features == 0 || in_features == 0 {
            return self.empty(&Shape::new([batch, out_features]), DType::F32);
        }
        if let Some(b) = bias {
            let small = batch <= 128;
            let (label, code) = match (small, activation) {
                (true, modern_chainer_core::Activation::Silu) => (
                    "matmul_nt_small_bias_silu",
                    shaders::MATMUL_NT_SMALL_BIAS_SILU,
                ),
                (true, modern_chainer_core::Activation::Gelu) => (
                    "matmul_nt_small_bias_gelu",
                    shaders::MATMUL_NT_SMALL_BIAS_GELU,
                ),
                (false, modern_chainer_core::Activation::Silu) => {
                    ("matmul_nt_bias_silu", shaders::MATMUL_NT_BIAS_SILU)
                }
                (false, modern_chainer_core::Activation::Gelu) => {
                    ("matmul_nt_bias_gelu", shaders::MATMUL_NT_BIAS_GELU)
                }
                (false, modern_chainer_core::Activation::Relu) => {
                    ("matmul_nt_bias_relu", shaders::MATMUL_NT_BIAS_RELU)
                }
                (true, modern_chainer_core::Activation::Relu) => {
                    let y = self.linear(x, weight, Some(b))?;
                    return self.relu(&y);
                }
            };
            let out = self.empty(&Shape::new([batch, out_features]), DType::F32)?;
            let mut params = Vec::with_capacity(16);
            for v in [batch as u32, in_features as u32, out_features as u32, 0] {
                params.extend_from_slice(&v.to_ne_bytes());
            }
            let dims = self.uniform_unique(&params);
            let (pipeline, layout) = self.pipeline(
                label,
                code,
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
                    (0, x.storage().buffer.as_ref()),
                    (1, weight.storage().buffer.as_ref()),
                    (2, out.storage().buffer.as_ref()),
                    (3, b.storage().buffer.as_ref()),
                    (4, &dims),
                ],
            );
            let tile = if small { 16u64 } else { 32u64 };
            let px = (out_features as u64).div_ceil(tile).min(65_535) as u32;
            let py = (batch as u64).div_ceil(tile).min(65_535) as u32;
            self.dispatch_2d(&pipeline, &group, px.max(1), py.max(1));
            return Ok(out);
        }
        let y = self.linear(x, weight, None)?;
        match activation {
            modern_chainer_core::Activation::Relu => self.relu(&y),
            modern_chainer_core::Activation::Gelu => self.gelu(&y),
            modern_chainer_core::Activation::Silu => self.silu(&y),
        }
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

    fn full_like(&self, a: &Tensor<VulkanBackend>, value: f64) -> Result<Tensor<VulkanBackend>> {
        self.fill(a, value as f32)
    }

    fn full_like_scalar(
        &self,
        target: &Tensor<VulkanBackend>,
        scalar: &Tensor<VulkanBackend>,
        factor: f64,
    ) -> Result<Tensor<VulkanBackend>> {
        let out = self.empty(target.shape(), DType::F32)?;
        let mut raw = Vec::with_capacity(16);
        raw.extend_from_slice(&(factor as f32).to_ne_bytes());
        raw.extend_from_slice(&[0u8; 12]);
        let params = self.uniform(&raw);
        let (pipeline, layout) = self.pipeline(
            "fill_from_scalar",
            shaders::FILL_FROM_SCALAR,
            &[storage(0, true), storage(1, false), uniform(2)],
        );
        let group = self.bind_group(
            &layout,
            &[
                (0, scalar.storage().buffer.as_ref()),
                (1, out.storage().buffer.as_ref()),
                (2, &params),
            ],
        );
        self.dispatch_vec4(&pipeline, &group, out.numel());
        Ok(out)
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

    fn embedding(
        &self,
        ids: &Tensor<VulkanBackend>,
        weight: &Tensor<VulkanBackend>,
    ) -> Result<Tensor<VulkanBackend>> {
        if weight.rank() != 2 {
            return Err(Error::invalid("embedding weight must be [vocab, dim]"));
        }
        let n = ids.numel();
        let (vocab, dim) = (weight.shape().dims()[0], weight.shape().dims()[1]);
        let id_stride = match ids.dtype() {
            DType::I64 => 2u32,
            DType::I32 | DType::U8 => 1,
            other => return Err(Error::UnsupportedDType(other)),
        };
        let mut shape = ids.shape().dims().to_vec();
        shape.push(dim);
        let out = self.empty(&Shape::new(shape), DType::F32)?;
        let mut raw = Vec::with_capacity(16);
        raw.extend_from_slice(&(n as u32).to_ne_bytes());
        raw.extend_from_slice(&(dim as u32).to_ne_bytes());
        raw.extend_from_slice(&id_stride.to_ne_bytes());
        raw.extend_from_slice(&0u32.to_ne_bytes());
        let params = self.uniform(&raw);
        let (pipeline, layout) = self.pipeline(
            "embedding_forward",
            shaders::EMBEDDING_FORWARD,
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
                (0, ids.storage().buffer.as_ref()),
                (1, weight.storage().buffer.as_ref()),
                (2, out.storage().buffer.as_ref()),
                (3, &params),
            ],
        );
        self.dispatch(&pipeline, &group, n * dim);
        let _ = vocab;
        Ok(out)
    }

    fn embedding_backward(
        &self,
        ids: &Tensor<VulkanBackend>,
        weight: &Tensor<VulkanBackend>,
        grad: &Tensor<VulkanBackend>,
    ) -> Result<Tensor<VulkanBackend>> {
        let n = ids.numel();
        let (vocab, dim) = (weight.shape().dims()[0], weight.shape().dims()[1]);
        let id_stride = match ids.dtype() {
            DType::I64 => 2u32,
            DType::I32 | DType::U8 => 1,
            other => return Err(Error::UnsupportedDType(other)),
        };
        let out = self.empty(weight.shape(), DType::F32)?;
        let mut raw = Vec::with_capacity(16);
        raw.extend_from_slice(&(n as u32).to_ne_bytes());
        raw.extend_from_slice(&(dim as u32).to_ne_bytes());
        raw.extend_from_slice(&id_stride.to_ne_bytes());
        raw.extend_from_slice(&0u32.to_ne_bytes());
        let params = self.uniform(&raw);
        let (pipeline, layout) = self.pipeline(
            "embedding_backward",
            shaders::EMBEDDING_BACKWARD,
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
                (0, ids.storage().buffer.as_ref()),
                (1, grad.storage().buffer.as_ref()),
                (2, out.storage().buffer.as_ref()),
                (3, &params),
            ],
        );
        self.dispatch_2d(&pipeline, &group, vocab as u32, 1);
        Ok(out)
    }

    fn layer_norm(
        &self,
        x: &Tensor<VulkanBackend>,
        gamma: &Tensor<VulkanBackend>,
        beta: &Tensor<VulkanBackend>,
        eps: f64,
    ) -> Result<Tensor<VulkanBackend>> {
        if x.rank() < 2 {
            return Err(Error::invalid("layer_norm expects rank >= 2"));
        }
        let dim = *x.shape().dims().last().unwrap();
        let rows = x.numel() / dim.max(1);
        let out = self.empty(x.shape(), DType::F32)?;
        let mut raw = Vec::with_capacity(16);
        raw.extend_from_slice(&(rows as u32).to_ne_bytes());
        raw.extend_from_slice(&(dim as u32).to_ne_bytes());
        raw.extend_from_slice(&(eps as f32).to_ne_bytes());
        raw.extend_from_slice(&0u32.to_ne_bytes());
        let params = self.uniform(&raw);
        let (pipeline, layout) = self.pipeline(
            "layer_norm_forward",
            shaders::LAYER_NORM_FORWARD,
            &[
                storage(0, true),
                storage(1, true),
                storage(2, true),
                storage(3, false),
                uniform(4),
            ],
        );
        let group = self.bind_group(
            &layout,
            &[
                (0, x.storage().buffer.as_ref()),
                (1, gamma.storage().buffer.as_ref()),
                (2, beta.storage().buffer.as_ref()),
                (3, out.storage().buffer.as_ref()),
                (4, &params),
            ],
        );
        self.dispatch_2d(&pipeline, &group, rows as u32, 1);
        Ok(out)
    }

    fn layer_norm_backward(
        &self,
        x: &Tensor<VulkanBackend>,
        gamma: &Tensor<VulkanBackend>,
        beta: &Tensor<VulkanBackend>,
        grad: &Tensor<VulkanBackend>,
        eps: f64,
    ) -> Result<(
        Tensor<VulkanBackend>,
        Tensor<VulkanBackend>,
        Tensor<VulkanBackend>,
    )> {
        let dim = *x.shape().dims().last().unwrap();
        let rows = x.numel() / dim.max(1);
        let dx = self.empty(x.shape(), DType::F32)?;
        let dgamma = self.empty(gamma.shape(), DType::F32)?;
        let dbeta = self.empty(beta.shape(), DType::F32)?;
        let mut raw = Vec::with_capacity(16);
        raw.extend_from_slice(&(rows as u32).to_ne_bytes());
        raw.extend_from_slice(&(dim as u32).to_ne_bytes());
        raw.extend_from_slice(&(eps as f32).to_ne_bytes());
        raw.extend_from_slice(&0u32.to_ne_bytes());
        let params = self.uniform(&raw);
        let (pipeline, layout) = self.pipeline(
            "layer_norm_backward_dx",
            shaders::LAYER_NORM_BACKWARD_DX,
            &[
                storage(0, true),
                storage(1, true),
                storage(2, true),
                storage(3, false),
                uniform(4),
            ],
        );
        let group = self.bind_group(
            &layout,
            &[
                (0, x.storage().buffer.as_ref()),
                (1, gamma.storage().buffer.as_ref()),
                (2, grad.storage().buffer.as_ref()),
                (3, dx.storage().buffer.as_ref()),
                (4, &params),
            ],
        );
        self.dispatch_2d(&pipeline, &group, rows as u32, 1);
        let (pipeline, layout) = self.pipeline(
            "layer_norm_backward_params",
            shaders::LAYER_NORM_BACKWARD_PARAMS,
            &[
                storage(0, true),
                storage(1, true),
                storage(2, false),
                storage(3, false),
                uniform(4),
            ],
        );
        let group = self.bind_group(
            &layout,
            &[
                (0, x.storage().buffer.as_ref()),
                (1, grad.storage().buffer.as_ref()),
                (2, dgamma.storage().buffer.as_ref()),
                (3, dbeta.storage().buffer.as_ref()),
                (4, &params),
            ],
        );
        self.dispatch_2d(&pipeline, &group, dim.div_ceil(256) as u32, 1);
        Ok((dx, dgamma, dbeta))
    }

    fn softmax_last_dim(&self, x: &Tensor<VulkanBackend>) -> Result<Tensor<VulkanBackend>> {
        if x.rank() != 2 {
            return Err(Error::invalid("softmax expects a 2-D tensor"));
        }
        let (rows, cols) = (x.shape().dims()[0], x.shape().dims()[1]);
        let out = self.empty(x.shape(), DType::F32)?;
        let mut raw = Vec::with_capacity(16);
        raw.extend_from_slice(&(rows as u32).to_ne_bytes());
        raw.extend_from_slice(&(cols as u32).to_ne_bytes());
        raw.extend_from_slice(&0u32.to_ne_bytes());
        raw.extend_from_slice(&0u32.to_ne_bytes());
        let params = self.uniform(&raw);
        let (pipeline, layout) = self.pipeline(
            "softmax_forward",
            shaders::SOFTMAX_FORWARD,
            &[storage(0, true), storage(1, false), uniform(2)],
        );
        let group = self.bind_group(
            &layout,
            &[
                (0, x.storage().buffer.as_ref()),
                (1, out.storage().buffer.as_ref()),
                (2, &params),
            ],
        );
        self.dispatch_2d(&pipeline, &group, rows as u32, 1);
        Ok(out)
    }

    fn softmax_last_dim_backward(
        &self,
        x: &Tensor<VulkanBackend>,
        grad: &Tensor<VulkanBackend>,
    ) -> Result<Tensor<VulkanBackend>> {
        if x.rank() != 2 || x.shape() != grad.shape() {
            return Err(Error::invalid("softmax backward shape mismatch"));
        }
        let (rows, cols) = (x.shape().dims()[0], x.shape().dims()[1]);
        let out = self.empty(x.shape(), DType::F32)?;
        let mut raw = Vec::with_capacity(16);
        raw.extend_from_slice(&(rows as u32).to_ne_bytes());
        raw.extend_from_slice(&(cols as u32).to_ne_bytes());
        raw.extend_from_slice(&0u32.to_ne_bytes());
        raw.extend_from_slice(&0u32.to_ne_bytes());
        let params = self.uniform(&raw);
        let (pipeline, layout) = self.pipeline(
            "softmax_backward",
            shaders::SOFTMAX_BACKWARD,
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
                (0, x.storage().buffer.as_ref()),
                (1, grad.storage().buffer.as_ref()),
                (2, out.storage().buffer.as_ref()),
                (3, &params),
            ],
        );
        self.dispatch_2d(&pipeline, &group, rows as u32, 1);
        Ok(out)
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

    fn split_heads(
        &self,
        x: &Tensor<VulkanBackend>,
        heads: usize,
        head_dim: usize,
    ) -> Result<Tensor<VulkanBackend>> {
        let xc = self.to_cpu(x)?;
        self.upload(&CpuBackend::new().ops().split_heads(&xc, heads, head_dim)?)
    }

    fn merge_heads(
        &self,
        x: &Tensor<VulkanBackend>,
        heads: usize,
    ) -> Result<Tensor<VulkanBackend>> {
        let xc = self.to_cpu(x)?;
        self.upload(&CpuBackend::new().ops().merge_heads(&xc, heads)?)
    }

    fn attention(
        &self,
        q: &Tensor<VulkanBackend>,
        k: &Tensor<VulkanBackend>,
        v: &Tensor<VulkanBackend>,
        scale: f64,
        causal: bool,
    ) -> Result<Tensor<VulkanBackend>> {
        let qc = self.to_cpu(q)?;
        let kc = self.to_cpu(k)?;
        let vc = self.to_cpu(v)?;
        self.upload(
            &CpuBackend::new()
                .ops()
                .attention(&qc, &kc, &vc, scale, causal)?,
        )
    }

    fn concat_seq(
        &self,
        a: &Tensor<VulkanBackend>,
        b: &Tensor<VulkanBackend>,
    ) -> Result<Tensor<VulkanBackend>> {
        let ac = self.to_cpu(a)?;
        let bc = self.to_cpu(b)?;
        self.upload(&CpuBackend::new().ops().concat_seq(&ac, &bc)?)
    }

    fn last_row(&self, x: &Tensor<VulkanBackend>) -> Result<Tensor<VulkanBackend>> {
        let xc = self.to_cpu(x)?;
        self.upload(&CpuBackend::new().ops().last_row(&xc)?)
    }

    #[allow(clippy::too_many_arguments)]
    fn paged_attention(
        &self,
        q: &Tensor<VulkanBackend>,
        k_blocks: &Tensor<VulkanBackend>,
        v_blocks: &Tensor<VulkanBackend>,
        block_table: &Tensor<VulkanBackend>,
        seq_len: usize,
        scale: f64,
        causal: bool,
    ) -> Result<Tensor<VulkanBackend>> {
        let qc = self.to_cpu(q)?;
        let kc = self.to_cpu(k_blocks)?;
        let vc = self.to_cpu(v_blocks)?;
        let tc = self.to_cpu(block_table)?;
        self.upload(
            &CpuBackend::new()
                .ops()
                .paged_attention(&qc, &kc, &vc, &tc, seq_len, scale, causal)?,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn paged_append(
        &self,
        k_blocks: &Tensor<VulkanBackend>,
        v_blocks: &Tensor<VulkanBackend>,
        block_table: &Tensor<VulkanBackend>,
        len: usize,
        k_new: &Tensor<VulkanBackend>,
        v_new: &Tensor<VulkanBackend>,
    ) -> Result<(Tensor<VulkanBackend>, Tensor<VulkanBackend>)> {
        let kc = self.to_cpu(k_blocks)?;
        let vc = self.to_cpu(v_blocks)?;
        let tc = self.to_cpu(block_table)?;
        let knc = self.to_cpu(k_new)?;
        let vnc = self.to_cpu(v_new)?;
        let (nk, nv) = CpuBackend::new()
            .ops()
            .paged_append(&kc, &vc, &tc, len, &knc, &vnc)?;
        Ok((self.upload(&nk)?, self.upload(&nv)?))
    }

    #[allow(clippy::too_many_arguments)]
    fn paged_attention_batch(
        &self,
        q: &Tensor<VulkanBackend>,
        k_blocks: &Tensor<VulkanBackend>,
        v_blocks: &Tensor<VulkanBackend>,
        block_tables: &Tensor<VulkanBackend>,
        seq_lens: &Tensor<VulkanBackend>,
        scale: f64,
        causal: bool,
    ) -> Result<Tensor<VulkanBackend>> {
        let qc = self.to_cpu(q)?;
        let kc = self.to_cpu(k_blocks)?;
        let vc = self.to_cpu(v_blocks)?;
        let tc = self.to_cpu(block_tables)?;
        let sc = self.to_cpu(seq_lens)?;
        self.upload(
            &CpuBackend::new()
                .ops()
                .paged_attention_batch(&qc, &kc, &vc, &tc, &sc, scale, causal)?,
        )
    }

    fn rms_norm(
        &self,
        x: &Tensor<VulkanBackend>,
        weight: &Tensor<VulkanBackend>,
        eps: f64,
    ) -> Result<Tensor<VulkanBackend>> {
        let xc = self.to_cpu(x)?;
        let wc = self.to_cpu(weight)?;
        self.upload(&CpuBackend::new().ops().rms_norm(&xc, &wc, eps)?)
    }

    fn rms_norm_backward(
        &self,
        x: &Tensor<VulkanBackend>,
        weight: &Tensor<VulkanBackend>,
        grad: &Tensor<VulkanBackend>,
        eps: f64,
    ) -> Result<(Tensor<VulkanBackend>, Tensor<VulkanBackend>)> {
        let xc = self.to_cpu(x)?;
        let wc = self.to_cpu(weight)?;
        let gc = self.to_cpu(grad)?;
        let (dx, dw) = CpuBackend::new()
            .ops()
            .rms_norm_backward(&xc, &wc, &gc, eps)?;
        Ok((self.upload(&dx)?, self.upload(&dw)?))
    }

    fn clamp(
        &self,
        x: &Tensor<VulkanBackend>,
        min: f64,
        max: f64,
    ) -> Result<Tensor<VulkanBackend>> {
        self.check_f32(x, "clamp")?;
        let out = self.empty(x.shape(), DType::F32)?;
        let mut raw = Vec::with_capacity(16);
        raw.extend_from_slice(&(min as f32).to_ne_bytes());
        raw.extend_from_slice(&(max as f32).to_ne_bytes());
        raw.extend_from_slice(&[0u8; 8]);
        let params = self.uniform(&raw);
        let (pipeline, layout) = self.pipeline(
            "clamp4",
            shaders::CLAMP4,
            &[storage(0, true), storage(1, false), uniform(2)],
        );
        let group = self.bind_group(
            &layout,
            &[
                (0, x.storage().buffer.as_ref()),
                (1, out.storage().buffer.as_ref()),
                (2, &params),
            ],
        );
        self.dispatch_vec4(&pipeline, &group, out.numel());
        Ok(out)
    }

    fn abs(&self, x: &Tensor<VulkanBackend>) -> Result<Tensor<VulkanBackend>> {
        self.check_f32(x, "abs")?;
        self.unary_vec4("abs4", shaders::ABS4, x)
    }

    fn apply_rope(
        &self,
        q: &Tensor<VulkanBackend>,
        k: &Tensor<VulkanBackend>,
        positions: &Tensor<VulkanBackend>,
        theta: f64,
        interleaved: bool,
    ) -> Result<(Tensor<VulkanBackend>, Tensor<VulkanBackend>)> {
        let qc = self.to_cpu(q)?;
        let kc = self.to_cpu(k)?;
        let pc = self.to_cpu(positions)?;
        let (rq, rk) = CpuBackend::new()
            .ops()
            .apply_rope(&qc, &kc, &pc, theta, interleaved)?;
        Ok((self.upload(&rq)?, self.upload(&rk)?))
    }

    fn cast(&self, x: &Tensor<VulkanBackend>, dtype: DType) -> Result<Tensor<VulkanBackend>> {
        let xc = self.to_cpu(x)?;
        self.upload(&CpuBackend::new().ops().cast(&xc, dtype)?)
    }

    fn dropout(
        &self,
        x: &Tensor<VulkanBackend>,
        mask: &Tensor<VulkanBackend>,
        p: f64,
    ) -> Result<Tensor<VulkanBackend>> {
        let xc = self.to_cpu(x)?;
        let mc = self.to_cpu(mask)?;
        self.upload(&CpuBackend::new().ops().dropout(&xc, &mc, p)?)
    }
}

impl VulkanOps {
    fn prepare_matmul_nt_small_bias(
        &self,
        a: &Tensor<VulkanBackend>,
        b: &Tensor<VulkanBackend>,
        bias: &Tensor<VulkanBackend>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(PreparedDispatch, Tensor<VulkanBackend>)> {
        let out = self.empty(&Shape::new([m, n]), DType::F32)?;
        let mut params = Vec::with_capacity(16);
        for value in [m as u32, k as u32, n as u32, 0] {
            params.extend_from_slice(&value.to_ne_bytes());
        }
        let dims = self.uniform_unique(&params);
        let (pipeline, layout) = self.pipeline(
            "matmul_nt_small_bias",
            shaders::MATMUL_NT_SMALL_BIAS,
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
        let x = (n as u64).div_ceil(16).min(65_535) as u32;
        let y = (m as u64).div_ceil(16).min(65_535) as u32;
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
        let aligned = !transposed_b
            && !small
            && m.is_multiple_of(32)
            && n.is_multiple_of(32)
            && k.is_multiple_of(32);
        let (label, code) = if aligned {
            ("matmul_aligned", shaders::MATMUL_ALIGNED)
        } else {
            match (transposed_b, small) {
                (true, true) => ("matmul_nt_small", shaders::MATMUL_NT_SMALL),
                (false, true) => ("matmul_small", shaders::MATMUL_SMALL),
                (true, false) => ("matmul_nt", shaders::MATMUL_NT),
                (false, false) => ("matmul", shaders::MATMUL),
            }
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
        let small = k <= 128;
        let (label, code) = if small {
            ("matmul_tn_small", shaders::MATMUL_TN_SMALL)
        } else {
            ("matmul_tn", shaders::MATMUL_TN)
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
        while len > 65_536 {
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

    fn prepare_sum_to_bias(
        &self,
        a: &Tensor<VulkanBackend>,
    ) -> Result<(PreparedDispatch, Tensor<VulkanBackend>)> {
        let (rows, cols) = (a.shape().dims()[0], a.shape().dims()[1]);
        let output = self.empty(&Shape::new([cols]), DType::F32)?;
        let mut raw = Vec::with_capacity(16);
        raw.extend_from_slice(&(rows as u32).to_ne_bytes());
        raw.extend_from_slice(&(cols as u32).to_ne_bytes());
        raw.extend_from_slice(&0u32.to_ne_bytes());
        raw.extend_from_slice(&0u32.to_ne_bytes());
        let params = self.uniform_unique(&raw);
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
        Ok((
            PreparedDispatch {
                pipeline,
                group,
                x: cols.div_ceil(256) as u32,
                y: 1,
            },
            output,
        ))
    }

    fn sum_to_bias(&self, a: &Tensor<VulkanBackend>) -> Result<Tensor<VulkanBackend>> {
        let (dispatch, output) = self.prepare_sum_to_bias(a)?;
        self.submit_batch(&[dispatch])?;
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
