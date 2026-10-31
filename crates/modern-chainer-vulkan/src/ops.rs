use std::sync::Arc;

use modern_chainer_core::{Backend, DType, Error, Ops, Result, Shape, Tensor};
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

    fn uniform(&self, bytes: &[u8]) -> wgpu::Buffer {
        let mut padded = [0u8; 16];
        padded[..bytes.len()].copy_from_slice(bytes);
        let buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("modern-chainer params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.ctx.queue.write_buffer(&buffer, 0, &padded);
        buffer
    }

    fn pipeline(
        &self,
        label: &str,
        code: &str,
        entries: &[wgpu::BindGroupLayoutEntry],
    ) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
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

    fn binary_same(
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
                (0, &a.storage().buffer),
                (1, &b.storage().buffer),
                (2, &out.storage().buffer),
            ],
        );
        self.dispatch(&pipeline, &group, out.numel());
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
            &[(0, &a.storage().buffer), (1, &out.storage().buffer)],
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

impl Ops<VulkanBackend> for VulkanOps {
    fn add(
        &self,
        a: &Tensor<VulkanBackend>,
        b: &Tensor<VulkanBackend>,
    ) -> Result<Tensor<VulkanBackend>> {
        if a.shape() == b.shape() {
            return self.binary_same("add", shaders::ADD, a, b);
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
                    (0, &a.storage().buffer),
                    (1, &b.storage().buffer),
                    (2, &out.storage().buffer),
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
        self.binary_same("sub", shaders::SUB, a, b)
    }

    fn mul(
        &self,
        a: &Tensor<VulkanBackend>,
        b: &Tensor<VulkanBackend>,
    ) -> Result<Tensor<VulkanBackend>> {
        self.binary_same("mul", shaders::MUL, a, b)
    }

    fn div(
        &self,
        a: &Tensor<VulkanBackend>,
        b: &Tensor<VulkanBackend>,
    ) -> Result<Tensor<VulkanBackend>> {
        self.binary_same("div", shaders::DIV, a, b)
    }

    fn sqrt(&self, a: &Tensor<VulkanBackend>) -> Result<Tensor<VulkanBackend>> {
        self.unary("sqrt", shaders::SQRT, a)
    }

    fn relu(&self, a: &Tensor<VulkanBackend>) -> Result<Tensor<VulkanBackend>> {
        self.unary("relu", shaders::RELU, a)
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
                (0, &x.storage().buffer),
                (1, &grad.storage().buffer),
                (2, &out.storage().buffer),
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
        let out = self.empty(&Shape::new([m, n]), DType::F32)?;
        let mut params = Vec::with_capacity(16);
        for value in [m as u32, k as u32, n as u32, 0] {
            params.extend_from_slice(&value.to_ne_bytes());
        }
        let dims = self.uniform(&params);
        let (pipeline, layout) = self.pipeline(
            "matmul",
            shaders::MATMUL,
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
                (0, &a.storage().buffer),
                (1, &b.storage().buffer),
                (2, &out.storage().buffer),
                (3, &dims),
            ],
        );
        let x = (n as u64).div_ceil(16).min(65_535) as u32;
        let y = (m as u64).div_ceil(16).min(65_535) as u32;
        self.dispatch_2d(&pipeline, &group, x.max(1), y.max(1));
        Ok(out)
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
                (0, &a.storage().buffer),
                (1, &out.storage().buffer),
                (2, &dims),
            ],
        );
        self.dispatch(&pipeline, &group, out.numel());
        Ok(out)
    }

    fn sum_all(&self, a: &Tensor<VulkanBackend>) -> Result<Tensor<VulkanBackend>> {
        let cpu = self.to_cpu(a)?;
        let backend = CpuBackend::new();
        let result = backend.ops().sum_all(&cpu)?;
        self.upload(&result)
    }

    fn sum_to(&self, a: &Tensor<VulkanBackend>, shape: &Shape) -> Result<Tensor<VulkanBackend>> {
        let cpu = self.to_cpu(a)?;
        let backend = CpuBackend::new();
        let result = backend.ops().sum_to(&cpu, shape)?;
        self.upload(&result)
    }

    fn scale(&self, a: &Tensor<VulkanBackend>, factor: f64) -> Result<Tensor<VulkanBackend>> {
        let out = self.empty(a.shape(), DType::F32)?;
        let params = self.uniform(&(factor as f32).to_ne_bytes());
        let (pipeline, layout) = self.pipeline(
            "scale",
            shaders::SCALE,
            &[storage(0, true), storage(1, false), uniform(2)],
        );
        let group = self.bind_group(
            &layout,
            &[
                (0, &a.storage().buffer),
                (1, &out.storage().buffer),
                (2, &params),
            ],
        );
        self.dispatch(&pipeline, &group, out.numel());
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
            .read_buffer(&a.storage().buffer, a.dtype().size_in_bytes())?;
        match a.dtype() {
            DType::F32 => Ok(f32::from_ne_bytes(bytes[..4].try_into().unwrap()) as f64),
            DType::F64 => Ok(f64::from_ne_bytes(bytes[..8].try_into().unwrap())),
            DType::I64 => Ok(i64::from_ne_bytes(bytes[..8].try_into().unwrap()) as f64),
            DType::I32 => Ok(i32::from_ne_bytes(bytes[..4].try_into().unwrap()) as f64),
            other => Err(Error::UnsupportedDType(other)),
        }
    }

    fn softmax_cross_entropy(
        &self,
        logits: &Tensor<VulkanBackend>,
        target: &Tensor<VulkanBackend>,
    ) -> Result<Tensor<VulkanBackend>> {
        let logits = self.to_cpu(logits)?;
        let target = self.to_cpu(target)?;
        let backend = CpuBackend::new();
        let result = backend.ops().softmax_cross_entropy(&logits, &target)?;
        self.upload(&result)
    }

    fn softmax_cross_entropy_backward(
        &self,
        logits: &Tensor<VulkanBackend>,
        target: &Tensor<VulkanBackend>,
    ) -> Result<Tensor<VulkanBackend>> {
        let logits = self.to_cpu(logits)?;
        let target = self.to_cpu(target)?;
        let backend = CpuBackend::new();
        let result = backend
            .ops()
            .softmax_cross_entropy_backward(&logits, &target)?;
        self.upload(&result)
    }
}

impl VulkanOps {
    fn fill(&self, a: &Tensor<VulkanBackend>, value: f32) -> Result<Tensor<VulkanBackend>> {
        let out = self.empty(a.shape(), DType::F32)?;
        let params = self.uniform(&value.to_ne_bytes());
        let (pipeline, layout) =
            self.pipeline("fill", shaders::FILL, &[storage(0, false), uniform(1)]);
        let group = self.bind_group(&layout, &[(0, &out.storage().buffer), (1, &params)]);
        self.dispatch(&pipeline, &group, out.numel());
        Ok(out)
    }
}
