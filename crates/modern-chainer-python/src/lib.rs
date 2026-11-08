use std::collections::HashMap;
use std::str::FromStr;
use std::sync::{Mutex, OnceLock};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};

use modern_chainer::optimizers::{LinearWarmupCosineLR, Optimizer};
use modern_chainer::prelude::*;
use modern_chainer::vulkan::VulkanBackend;

fn to_py(error: modern_chainer::Error) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

fn parse_device(device: &str) -> PyResult<Device> {
    Device::from_str(device).map_err(to_py)
}

fn backend_error(device: Device) -> PyErr {
    PyRuntimeError::new_err(format!("{device} is not available"))
}

fn cuda_backend(index: u32) -> PyResult<CudaBackend> {
    static CACHE: OnceLock<Mutex<HashMap<u32, CudaBackend>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut backends = cache.lock().expect("cuda backend cache poisoned");
    if let Some(backend) = backends.get(&index) {
        return Ok(backend.clone());
    }
    let backend = CudaBackend::new(index).map_err(|_| backend_error(Device::Cuda(index)))?;
    backends.insert(index, backend.clone());
    Ok(backend)
}

fn vulkan_backend(index: usize) -> PyResult<VulkanBackend> {
    static CACHE: OnceLock<Mutex<HashMap<usize, VulkanBackend>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut backends = cache.lock().expect("vulkan backend cache poisoned");
    if let Some(backend) = backends.get(&index) {
        return Ok(backend.clone());
    }
    let backend = VulkanBackend::with_adapter(index).map_err(|error| {
        PyRuntimeError::new_err(format!("vulkan:{index} is not available: {error}"))
    })?;
    backends.insert(index, backend.clone());
    Ok(backend)
}

#[derive(Clone)]
enum DynTensor {
    Cpu(Tensor<CpuBackend>),
    Cuda(Tensor<CudaBackend>),
    Vulkan(Tensor<VulkanBackend>),
}

impl DynTensor {
    fn device(&self) -> Device {
        match self {
            Self::Cpu(_) => Device::Cpu,
            Self::Cuda(t) => Device::Cuda(t.backend().ordinal()),
            Self::Vulkan(t) => Device::Vulkan(t.backend().adapter_index() as u32),
        }
    }

    fn shape(&self) -> Vec<usize> {
        match self {
            Self::Cpu(t) => t.shape().dims().to_vec(),
            Self::Cuda(t) => t.shape().dims().to_vec(),
            Self::Vulkan(t) => t.shape().dims().to_vec(),
        }
    }

    fn dtype(&self) -> DType {
        match self {
            Self::Cpu(t) => t.dtype(),
            Self::Cuda(t) => t.dtype(),
            Self::Vulkan(t) => t.dtype(),
        }
    }

    fn numel(&self) -> usize {
        match self {
            Self::Cpu(t) => t.numel(),
            Self::Cuda(t) => t.numel(),
            Self::Vulkan(t) => t.numel(),
        }
    }

    fn to_bytes(&self) -> PyResult<Vec<u8>> {
        match self {
            Self::Cpu(t) => t.to_bytes().map_err(to_py),
            Self::Cuda(t) => t.to_bytes().map_err(to_py),
            Self::Vulkan(t) => t.to_bytes().map_err(to_py),
        }
    }

    fn from_bytes_on(
        device: Device,
        bytes: &[u8],
        shape: Vec<usize>,
        dtype: DType,
    ) -> PyResult<Self> {
        match device {
            Device::Cpu => Tensor::from_bytes(CpuBackend::new(), bytes, shape, dtype)
                .map(Self::Cpu)
                .map_err(to_py),
            Device::Cuda(index) => {
                let backend = cuda_backend(index)?;
                Tensor::from_bytes(backend, bytes, shape, dtype)
                    .map(Self::Cuda)
                    .map_err(to_py)
            }
            Device::Vulkan(index) => {
                let backend = vulkan_backend(index as usize)?;
                Tensor::from_bytes(backend, bytes, shape, dtype)
                    .map(Self::Vulkan)
                    .map_err(to_py)
            }
        }
    }

    fn zeros_on(device: Device, shape: Vec<usize>, dtype: DType) -> PyResult<Self> {
        match device {
            Device::Cpu => Tensor::zeros(CpuBackend::new(), shape, dtype)
                .map(Self::Cpu)
                .map_err(to_py),
            Device::Cuda(index) => {
                let backend = cuda_backend(index)?;
                Tensor::zeros(backend, shape, dtype)
                    .map(Self::Cuda)
                    .map_err(to_py)
            }
            Device::Vulkan(index) => {
                let backend = vulkan_backend(index as usize)?;
                Tensor::zeros(backend, shape, dtype)
                    .map(Self::Vulkan)
                    .map_err(to_py)
            }
        }
    }

    fn from_f32_on(device: Device, values: &[f32], shape: Vec<usize>) -> PyResult<Self> {
        match device {
            Device::Cpu => Tensor::from_f32(CpuBackend::new(), values, shape)
                .map(Self::Cpu)
                .map_err(to_py),
            Device::Cuda(index) => {
                let backend = cuda_backend(index)?;
                Tensor::from_f32(backend, values, shape)
                    .map(Self::Cuda)
                    .map_err(to_py)
            }
            Device::Vulkan(index) => {
                let backend = vulkan_backend(index as usize)?;
                Tensor::from_f32(backend, values, shape)
                    .map(Self::Vulkan)
                    .map_err(to_py)
            }
        }
    }

    fn to_vec_f32(&self) -> PyResult<Vec<f32>> {
        let bytes = self.to_bytes()?;
        let (chunks, rest) = bytes.as_chunks::<4>();
        if !rest.is_empty() {
            return Err(PyRuntimeError::new_err(
                "tensor size is not a multiple of 4",
            ));
        }
        Ok(chunks
            .iter()
            .map(|chunk| f32::from_ne_bytes(*chunk))
            .collect())
    }

    fn to_device(&self, device: &str) -> PyResult<Self> {
        let target = parse_device(device)?;
        if self.device() == target {
            return Ok(self.clone());
        }
        let bytes = self.to_bytes()?;
        Self::from_bytes_on(target, &bytes, self.shape(), self.dtype())
    }
}

fn numpy_array<'py>(py: Python<'py>, tensor: &DynTensor) -> PyResult<Bound<'py, PyAny>> {
    let numpy = py.import("numpy")?;
    let bytes = tensor.to_bytes()?;
    let buffer = PyBytes::new(py, &bytes);
    let array = numpy.call_method1("frombuffer", (buffer, "float32"))?;
    let shape = PyTuple::new(py, tensor.shape())?;
    array.call_method1("reshape", (shape,))
}

#[pyclass(name = "Tensor", module = "chainer")]
#[derive(Clone)]
pub struct PyTensor {
    inner: DynTensor,
}

#[pymethods]
impl PyTensor {
    #[staticmethod]
    #[pyo3(signature = (shape, device="cpu"))]
    fn zeros(shape: Vec<usize>, device: &str) -> PyResult<Self> {
        Ok(Self {
            inner: DynTensor::zeros_on(parse_device(device)?, shape, DType::F32)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (data, shape, device="cpu"))]
    fn from_list(data: Vec<f64>, shape: Vec<usize>, device: &str) -> PyResult<Self> {
        let values: Vec<f32> = data.into_iter().map(|x| x as f32).collect();
        Ok(Self {
            inner: DynTensor::from_f32_on(parse_device(device)?, &values, shape)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (object, device="cpu"))]
    fn from_numpy(object: &Bound<'_, PyAny>, device: &str) -> PyResult<Self> {
        let target = parse_device(device)?;
        let shape: Vec<usize> = object.getattr("shape")?.extract()?;
        let dtype: String = object.getattr("dtype")?.getattr("name")?.extract()?;
        let (array, kind) = if target == Device::Cpu {
            match dtype.as_str() {
                "int64" => (object.clone(), DType::I64),
                "int32" => (object.clone(), DType::I32),
                "uint8" => (object.clone(), DType::U8),
                _ => {
                    let numpy = object.py().import("numpy")?;
                    (
                        object.call_method1("astype", (numpy.getattr("float32")?,))?,
                        DType::F32,
                    )
                }
            }
        } else {
            let numpy = object.py().import("numpy")?;
            (
                object.call_method1("astype", (numpy.getattr("float32")?,))?,
                DType::F32,
            )
        };
        let bytes: Vec<u8> = array.call_method0("tobytes")?.extract()?;
        Ok(Self {
            inner: DynTensor::from_bytes_on(target, &bytes, shape, kind)?,
        })
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape()
    }

    #[getter]
    fn dtype(&self) -> String {
        self.inner.dtype().to_string()
    }

    #[getter]
    fn device(&self) -> String {
        self.inner.device().to_string()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.shape().len()
    }

    fn numel(&self) -> usize {
        self.inner.numel()
    }

    fn to(&self, device: &str) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.to_device(device)?,
        })
    }

    fn astype(&self, dtype: &str) -> PyResult<Self> {
        let target = DType::from_name(dtype)
            .ok_or_else(|| PyRuntimeError::new_err(format!("unknown dtype {dtype}")))?;
        let bytes = self.inner.to_bytes()?;
        let (chunks, _) = bytes.as_chunks::<4>();
        let values: Vec<f32> = chunks.iter().map(|c| f32::from_ne_bytes(*c)).collect();
        let cpu =
            Tensor::from_f32(CpuBackend::new(), &values, self.inner.shape()).map_err(to_py)?;
        let casted = cpu.backend().ops().cast(&cpu, target).map_err(to_py)?;
        let bytes = casted.to_bytes().map_err(to_py)?;
        Ok(Self {
            inner: DynTensor::from_bytes_on(
                self.inner.device(),
                &bytes,
                self.inner.shape(),
                target,
            )?,
        })
    }

    fn contiguous(&self) -> PyResult<Self> {
        match &self.inner {
            DynTensor::Cpu(t) => Ok(Self {
                inner: DynTensor::Cpu(t.contiguous().map_err(to_py)?),
            }),
            DynTensor::Cuda(t) => Ok(Self {
                inner: DynTensor::Cuda(t.contiguous().map_err(to_py)?),
            }),
            DynTensor::Vulkan(t) => Ok(Self {
                inner: DynTensor::Vulkan(t.contiguous().map_err(to_py)?),
            }),
        }
    }

    fn tolist(&self) -> PyResult<Vec<f32>> {
        self.inner.to_vec_f32()
    }

    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        numpy_array(py, &self.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, dtype={}, device={})",
            self.inner.shape(),
            self.inner.dtype(),
            self.inner.device()
        )
    }
}

#[derive(Clone)]
enum DynVariable {
    Cpu(Variable<CpuBackend>),
    Cuda(Variable<CudaBackend>),
    Vulkan(Variable<VulkanBackend>),
}

impl DynVariable {
    fn from_tensor(tensor: DynTensor) -> Self {
        match tensor {
            DynTensor::Cpu(t) => Self::Cpu(Variable::new(t)),
            DynTensor::Cuda(t) => Self::Cuda(Variable::new(t)),
            DynTensor::Vulkan(t) => Self::Vulkan(Variable::new(t)),
        }
    }

    fn constant(tensor: DynTensor) -> Self {
        match tensor {
            DynTensor::Cpu(t) => Self::Cpu(Variable::constant(t)),
            DynTensor::Cuda(t) => Self::Cuda(Variable::constant(t)),
            DynTensor::Vulkan(t) => Self::Vulkan(Variable::constant(t)),
        }
    }

    fn data(&self) -> DynTensor {
        match self {
            Self::Cpu(v) => DynTensor::Cpu(v.data()),
            Self::Cuda(v) => DynTensor::Cuda(v.data()),
            Self::Vulkan(v) => DynTensor::Vulkan(v.data()),
        }
    }

    fn shape(&self) -> Vec<usize> {
        match self {
            Self::Cpu(v) => v.data().shape().dims().to_vec(),
            Self::Cuda(v) => v.data().shape().dims().to_vec(),
            Self::Vulkan(v) => v.data().shape().dims().to_vec(),
        }
    }

    fn device(&self) -> Device {
        self.data().device()
    }
}

#[pyclass(name = "Variable", module = "chainer")]
#[derive(Clone)]
pub struct PyVariable {
    inner: DynVariable,
}

impl PyVariable {
    fn wrap(inner: DynVariable) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyVariable {
    #[new]
    fn new(tensor: PyRef<'_, PyTensor>) -> Self {
        Self::wrap(DynVariable::from_tensor(tensor.inner.clone()))
    }

    #[staticmethod]
    fn constant(tensor: PyRef<'_, PyTensor>) -> Self {
        Self::wrap(DynVariable::constant(tensor.inner.clone()))
    }

    #[getter]
    fn data(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.data(),
        }
    }

    #[getter]
    fn array<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        numpy_array(py, &self.inner.data())
    }

    #[getter]
    fn device(&self) -> String {
        self.inner.device().to_string()
    }

    #[getter]
    fn grad(&self) -> Option<PyTensor> {
        match &self.inner {
            DynVariable::Cpu(v) => v.grad().map(|inner| PyTensor {
                inner: DynTensor::Cpu(inner),
            }),
            DynVariable::Cuda(v) => v.grad().map(|inner| PyTensor {
                inner: DynTensor::Cuda(inner),
            }),
            DynVariable::Vulkan(v) => v.grad().map(|inner| PyTensor {
                inner: DynTensor::Vulkan(inner),
            }),
        }
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape()
    }

    #[getter]
    fn grad_fn(&self) -> Option<String> {
        match &self.inner {
            DynVariable::Cpu(v) => v.grad_fn_name().map(str::to_owned),
            DynVariable::Cuda(v) => v.grad_fn_name().map(str::to_owned),
            DynVariable::Vulkan(v) => v.grad_fn_name().map(str::to_owned),
        }
    }

    #[getter]
    fn requires_grad(&self) -> bool {
        match &self.inner {
            DynVariable::Cpu(v) => v.requires_grad(),
            DynVariable::Cuda(v) => v.requires_grad(),
            DynVariable::Vulkan(v) => v.requires_grad(),
        }
    }

    #[setter]
    fn set_requires_grad(&self, value: bool) {
        match &self.inner {
            DynVariable::Cpu(v) => v.set_requires_grad(value),
            DynVariable::Cuda(v) => v.set_requires_grad(value),
            DynVariable::Vulkan(v) => v.set_requires_grad(value),
        }
    }

    fn to(&self, device: &str) -> PyResult<Self> {
        Ok(Self::wrap(DynVariable::from_tensor(
            self.inner.data().to_device(device)?,
        )))
    }

    fn backward(&self) -> PyResult<()> {
        match &self.inner {
            DynVariable::Cpu(v) => v.backward().map_err(to_py),
            DynVariable::Cuda(v) => v.backward().map_err(to_py),
            DynVariable::Vulkan(v) => v.backward().map_err(to_py),
        }
    }

    fn cleargrad(&self) {
        match &self.inner {
            DynVariable::Cpu(v) => v.cleargrad(),
            DynVariable::Cuda(v) => v.cleargrad(),
            DynVariable::Vulkan(v) => v.cleargrad(),
        }
    }

    fn unchain_backward(&self) {
        match &self.inner {
            DynVariable::Cpu(v) => v.unchain_backward(),
            DynVariable::Cuda(v) => v.unchain_backward(),
            DynVariable::Vulkan(v) => v.unchain_backward(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Variable(shape={:?}, requires_grad={}, device={})",
            self.inner.shape(),
            self.requires_grad(),
            self.inner.device()
        )
    }
}

enum DynLinear {
    Cpu(Linear<CpuBackend>),
    Cuda(Linear<CudaBackend>),
    Vulkan(Linear<VulkanBackend>),
}

#[pyclass(name = "Linear", module = "chainer.links")]
pub struct PyLinear {
    inner: DynLinear,
}

#[pymethods]
impl PyLinear {
    #[new]
    #[pyo3(signature = (in_features, out_features, seed=0, device="cpu"))]
    fn new(in_features: usize, out_features: usize, seed: u64, device: &str) -> PyResult<Self> {
        let target = parse_device(device)?;
        let inner = match target {
            Device::Cpu => {
                DynLinear::Cpu(Linear::new(in_features, out_features, seed).map_err(to_py)?)
            }
            Device::Cuda(index) => {
                let backend = cuda_backend(index)?;
                DynLinear::Cuda(
                    Linear::with_backend(backend, in_features, out_features, seed)
                        .map_err(to_py)?,
                )
            }
            Device::Vulkan(index) => {
                let backend = vulkan_backend(index as usize)?;
                DynLinear::Vulkan(
                    Linear::with_backend(backend, in_features, out_features, seed)
                        .map_err(to_py)?,
                )
            }
        };
        Ok(Self { inner })
    }

    fn forward(&self, x: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
        match (&self.inner, &x.inner) {
            (DynLinear::Cpu(l), DynVariable::Cpu(x)) => Ok(PyVariable::wrap(DynVariable::Cpu(
                l.forward(x).map_err(to_py)?,
            ))),
            (DynLinear::Cuda(l), DynVariable::Cuda(x)) => Ok(PyVariable::wrap(DynVariable::Cuda(
                l.forward(x).map_err(to_py)?,
            ))),
            (DynLinear::Vulkan(l), DynVariable::Vulkan(x)) => Ok(PyVariable::wrap(
                DynVariable::Vulkan(l.forward(x).map_err(to_py)?),
            )),
            _ => Err(PyRuntimeError::new_err("Linear and input device mismatch")),
        }
    }

    fn parameters(&self) -> Vec<PyVariable> {
        match &self.inner {
            DynLinear::Cpu(l) => l
                .parameters()
                .into_iter()
                .map(|p| PyVariable::wrap(DynVariable::Cpu(p.variable)))
                .collect(),
            DynLinear::Cuda(l) => l
                .parameters()
                .into_iter()
                .map(|p| PyVariable::wrap(DynVariable::Cuda(p.variable)))
                .collect(),
            DynLinear::Vulkan(l) => l
                .parameters()
                .into_iter()
                .map(|p| PyVariable::wrap(DynVariable::Vulkan(p.variable)))
                .collect(),
        }
    }

    fn cleargrads(&self) {
        for parameter in self.parameters() {
            parameter.cleargrad();
        }
    }

    #[getter]
    #[allow(non_snake_case)]
    fn W(&self) -> PyVariable {
        match &self.inner {
            DynLinear::Cpu(l) => PyVariable::wrap(DynVariable::Cpu(l.weight().clone())),
            DynLinear::Cuda(l) => PyVariable::wrap(DynVariable::Cuda(l.weight().clone())),
            DynLinear::Vulkan(l) => PyVariable::wrap(DynVariable::Vulkan(l.weight().clone())),
        }
    }

    #[getter]
    fn b(&self) -> Option<PyVariable> {
        match &self.inner {
            DynLinear::Cpu(l) => l
                .bias()
                .cloned()
                .map(|v| PyVariable::wrap(DynVariable::Cpu(v))),
            DynLinear::Cuda(l) => l
                .bias()
                .cloned()
                .map(|v| PyVariable::wrap(DynVariable::Cuda(v))),
            DynLinear::Vulkan(l) => l
                .bias()
                .cloned()
                .map(|v| PyVariable::wrap(DynVariable::Vulkan(v))),
        }
    }

    fn __call__(&self, x: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
        self.forward(x)
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            DynLinear::Cpu(l) => format!(
                "Linear(in_features={}, out_features={}, device=cpu)",
                l.in_features(),
                l.out_features()
            ),
            DynLinear::Cuda(l) => format!(
                "Linear(in_features={}, out_features={}, device=cuda)",
                l.in_features(),
                l.out_features()
            ),
            DynLinear::Vulkan(l) => format!(
                "Linear(in_features={}, out_features={}, device=vulkan)",
                l.in_features(),
                l.out_features()
            ),
        }
    }
}

macro_rules! binary_op {
    ($name:ident, $op:path) => {
        #[pyfunction]
        fn $name(a: PyRef<'_, PyVariable>, b: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
            match (&a.inner, &b.inner) {
                (DynVariable::Cpu(a), DynVariable::Cpu(b)) => Ok(PyVariable::wrap(
                    DynVariable::Cpu($op(a, b).map_err(to_py)?),
                )),
                (DynVariable::Cuda(a), DynVariable::Cuda(b)) => Ok(PyVariable::wrap(
                    DynVariable::Cuda($op(a, b).map_err(to_py)?),
                )),
                (DynVariable::Vulkan(a), DynVariable::Vulkan(b)) => Ok(PyVariable::wrap(
                    DynVariable::Vulkan($op(a, b).map_err(to_py)?),
                )),
                _ => Err(PyRuntimeError::new_err("operand device mismatch")),
            }
        }
    };
}

macro_rules! unary_op {
    ($name:ident, $op:path) => {
        #[pyfunction]
        fn $name(x: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
            match &x.inner {
                DynVariable::Cpu(x) => {
                    Ok(PyVariable::wrap(DynVariable::Cpu($op(x).map_err(to_py)?)))
                }
                DynVariable::Cuda(x) => {
                    Ok(PyVariable::wrap(DynVariable::Cuda($op(x).map_err(to_py)?)))
                }
                DynVariable::Vulkan(x) => Ok(PyVariable::wrap(DynVariable::Vulkan(
                    $op(x).map_err(to_py)?,
                ))),
            }
        }
    };
}

binary_op!(add, F::add);
binary_op!(sub, F::sub);
binary_op!(mul, F::mul);
binary_op!(div, F::div);
binary_op!(matmul, F::matmul);
unary_op!(relu, F::relu);
unary_op!(sqrt, F::sqrt);
unary_op!(transpose, F::transpose);
unary_op!(neg, F::neg);
unary_op!(sum, F::sum);
unary_op!(mean, F::mean);
unary_op!(exp, F::exp);
unary_op!(log, F::log);
unary_op!(sigmoid, F::sigmoid);
unary_op!(softmax, F::softmax);
unary_op!(silu, F::silu);
unary_op!(gelu, F::gelu);
unary_op!(abs, F::abs);

#[pyfunction]
#[pyo3(signature = (x, factor))]
fn scale(x: PyRef<'_, PyVariable>, factor: f64) -> PyResult<PyVariable> {
    match &x.inner {
        DynVariable::Cpu(x) => Ok(PyVariable::wrap(DynVariable::Cpu(
            F::scale(x, factor).map_err(to_py)?,
        ))),
        DynVariable::Cuda(x) => Ok(PyVariable::wrap(DynVariable::Cuda(
            F::scale(x, factor).map_err(to_py)?,
        ))),
        DynVariable::Vulkan(x) => Ok(PyVariable::wrap(DynVariable::Vulkan(
            F::scale(x, factor).map_err(to_py)?,
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (x, min, max))]
fn clamp(x: PyRef<'_, PyVariable>, min: f64, max: f64) -> PyResult<PyVariable> {
    match &x.inner {
        DynVariable::Cpu(x) => Ok(PyVariable::wrap(DynVariable::Cpu(
            F::clamp(x, min, max).map_err(to_py)?,
        ))),
        DynVariable::Cuda(x) => Ok(PyVariable::wrap(DynVariable::Cuda(
            F::clamp(x, min, max).map_err(to_py)?,
        ))),
        DynVariable::Vulkan(x) => Ok(PyVariable::wrap(DynVariable::Vulkan(
            F::clamp(x, min, max).map_err(to_py)?,
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (x, weight, eps=1e-6))]
fn rms_norm(
    x: PyRef<'_, PyVariable>,
    weight: PyRef<'_, PyVariable>,
    eps: f64,
) -> PyResult<PyVariable> {
    match (&x.inner, &weight.inner) {
        (DynVariable::Cpu(x), DynVariable::Cpu(w)) => Ok(PyVariable::wrap(DynVariable::Cpu(
            F::rms_norm(x, w, eps).map_err(to_py)?,
        ))),
        (DynVariable::Cuda(x), DynVariable::Cuda(w)) => Ok(PyVariable::wrap(DynVariable::Cuda(
            F::rms_norm(x, w, eps).map_err(to_py)?,
        ))),
        (DynVariable::Vulkan(x), DynVariable::Vulkan(w)) => Ok(PyVariable::wrap(
            DynVariable::Vulkan(F::rms_norm(x, w, eps).map_err(to_py)?),
        )),
        _ => Err(PyRuntimeError::new_err("rms_norm device mismatch")),
    }
}

#[pyfunction]
#[pyo3(signature = (x, weight, bias=None))]
fn linear(
    x: PyRef<'_, PyVariable>,
    weight: PyRef<'_, PyVariable>,
    bias: Option<PyRef<'_, PyVariable>>,
) -> PyResult<PyVariable> {
    let bias = bias.as_ref().map(|b| &b.inner);
    match (&x.inner, &weight.inner, bias) {
        (DynVariable::Cpu(x), DynVariable::Cpu(w), None) => Ok(PyVariable::wrap(DynVariable::Cpu(
            F::linear(x, w, None).map_err(to_py)?,
        ))),
        (DynVariable::Cuda(x), DynVariable::Cuda(w), None) => Ok(PyVariable::wrap(
            DynVariable::Cuda(F::linear(x, w, None).map_err(to_py)?),
        )),
        (DynVariable::Vulkan(x), DynVariable::Vulkan(w), None) => Ok(PyVariable::wrap(
            DynVariable::Vulkan(F::linear(x, w, None).map_err(to_py)?),
        )),
        (DynVariable::Cpu(x), DynVariable::Cpu(w), Some(DynVariable::Cpu(b))) => Ok(
            PyVariable::wrap(DynVariable::Cpu(F::linear(x, w, Some(b)).map_err(to_py)?)),
        ),
        (DynVariable::Cuda(x), DynVariable::Cuda(w), Some(DynVariable::Cuda(b))) => Ok(
            PyVariable::wrap(DynVariable::Cuda(F::linear(x, w, Some(b)).map_err(to_py)?)),
        ),
        (DynVariable::Vulkan(x), DynVariable::Vulkan(w), Some(DynVariable::Vulkan(b))) => {
            Ok(PyVariable::wrap(DynVariable::Vulkan(
                F::linear(x, w, Some(b)).map_err(to_py)?,
            )))
        }
        _ => Err(PyRuntimeError::new_err("linear device mismatch")),
    }
}

#[pyfunction]
#[pyo3(signature = (x, mask, p))]
fn dropout(x: PyRef<'_, PyVariable>, mask: PyRef<'_, PyVariable>, p: f64) -> PyResult<PyVariable> {
    match (&x.inner, &mask.inner) {
        (DynVariable::Cpu(x), DynVariable::Cpu(m)) => Ok(PyVariable::wrap(DynVariable::Cpu(
            F::dropout(x, m, p).map_err(to_py)?,
        ))),
        (DynVariable::Cuda(x), DynVariable::Cuda(m)) => Ok(PyVariable::wrap(DynVariable::Cuda(
            F::dropout(x, m, p).map_err(to_py)?,
        ))),
        (DynVariable::Vulkan(x), DynVariable::Vulkan(m)) => Ok(PyVariable::wrap(
            DynVariable::Vulkan(F::dropout(x, m, p).map_err(to_py)?),
        )),
        _ => Err(PyRuntimeError::new_err("dropout device mismatch")),
    }
}

#[pyfunction]
fn reshape(x: PyRef<'_, PyVariable>, shape: Vec<usize>) -> PyResult<PyVariable> {
    match &x.inner {
        DynVariable::Cpu(x) => Ok(PyVariable::wrap(DynVariable::Cpu(
            F::reshape(x, Shape::new(shape)).map_err(to_py)?,
        ))),
        DynVariable::Cuda(x) => Ok(PyVariable::wrap(DynVariable::Cuda(
            F::reshape(x, Shape::new(shape)).map_err(to_py)?,
        ))),
        DynVariable::Vulkan(x) => Ok(PyVariable::wrap(DynVariable::Vulkan(
            F::reshape(x, Shape::new(shape)).map_err(to_py)?,
        ))),
    }
}

#[pyfunction]
fn permute(x: PyRef<'_, PyVariable>, axes: Vec<usize>) -> PyResult<PyVariable> {
    match &x.inner {
        DynVariable::Cpu(x) => Ok(PyVariable::wrap(DynVariable::Cpu(
            F::permute(x, axes).map_err(to_py)?,
        ))),
        DynVariable::Cuda(x) => Ok(PyVariable::wrap(DynVariable::Cuda(
            F::permute(x, axes).map_err(to_py)?,
        ))),
        DynVariable::Vulkan(x) => Ok(PyVariable::wrap(DynVariable::Vulkan(
            F::permute(x, axes).map_err(to_py)?,
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (x, gamma, beta, eps=1e-5))]
fn layer_norm(
    x: PyRef<'_, PyVariable>,
    gamma: PyRef<'_, PyVariable>,
    beta: PyRef<'_, PyVariable>,
    eps: f64,
) -> PyResult<PyVariable> {
    match (&x.inner, &gamma.inner, &beta.inner) {
        (DynVariable::Cpu(x), DynVariable::Cpu(g), DynVariable::Cpu(b)) => Ok(PyVariable::wrap(
            DynVariable::Cpu(F::layer_norm(x, g, b, eps).map_err(to_py)?),
        )),
        (DynVariable::Cuda(x), DynVariable::Cuda(g), DynVariable::Cuda(b)) => Ok(PyVariable::wrap(
            DynVariable::Cuda(F::layer_norm(x, g, b, eps).map_err(to_py)?),
        )),
        (DynVariable::Vulkan(x), DynVariable::Vulkan(g), DynVariable::Vulkan(b)) => {
            Ok(PyVariable::wrap(DynVariable::Vulkan(
                F::layer_norm(x, g, b, eps).map_err(to_py)?,
            )))
        }
        _ => Err(PyRuntimeError::new_err("layer_norm device mismatch")),
    }
}

#[pyfunction]
#[pyo3(signature = (x, gamma, beta, eps=1e-5))]
fn batch_norm(
    x: PyRef<'_, PyVariable>,
    gamma: PyRef<'_, PyVariable>,
    beta: PyRef<'_, PyVariable>,
    eps: f64,
) -> PyResult<PyVariable> {
    match (&x.inner, &gamma.inner, &beta.inner) {
        (DynVariable::Cpu(x), DynVariable::Cpu(g), DynVariable::Cpu(b)) => Ok(PyVariable::wrap(
            DynVariable::Cpu(F::batch_norm(x, g, b, eps).map_err(to_py)?),
        )),
        (DynVariable::Cuda(x), DynVariable::Cuda(g), DynVariable::Cuda(b)) => Ok(PyVariable::wrap(
            DynVariable::Cuda(F::batch_norm(x, g, b, eps).map_err(to_py)?),
        )),
        (DynVariable::Vulkan(x), DynVariable::Vulkan(g), DynVariable::Vulkan(b)) => {
            Ok(PyVariable::wrap(DynVariable::Vulkan(
                F::batch_norm(x, g, b, eps).map_err(to_py)?,
            )))
        }
        _ => Err(PyRuntimeError::new_err("batch_norm device mismatch")),
    }
}

#[pyfunction]
fn embedding(ids: PyRef<'_, PyVariable>, weight: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
    match (&ids.inner, &weight.inner) {
        (DynVariable::Cpu(i), DynVariable::Cpu(w)) => Ok(PyVariable::wrap(DynVariable::Cpu(
            F::embedding(i, w).map_err(to_py)?,
        ))),
        (DynVariable::Cuda(i), DynVariable::Cuda(w)) => Ok(PyVariable::wrap(DynVariable::Cuda(
            F::embedding(i, w).map_err(to_py)?,
        ))),
        (DynVariable::Vulkan(i), DynVariable::Vulkan(w)) => Ok(PyVariable::wrap(
            DynVariable::Vulkan(F::embedding(i, w).map_err(to_py)?),
        )),
        _ => Err(PyRuntimeError::new_err("embedding device mismatch")),
    }
}

#[pyfunction]
#[pyo3(signature = (x, weight, bias, stride=1, padding=0))]
fn conv2d(
    x: PyRef<'_, PyVariable>,
    weight: PyRef<'_, PyVariable>,
    bias: Option<PyRef<'_, PyVariable>>,
    stride: usize,
    padding: usize,
) -> PyResult<PyVariable> {
    let bias = bias.as_ref().map(|b| &b.inner);
    match (&x.inner, &weight.inner, bias) {
        (DynVariable::Cpu(x), DynVariable::Cpu(w), None) => Ok(PyVariable::wrap(DynVariable::Cpu(
            F::conv2d(x, w, None, stride, padding).map_err(to_py)?,
        ))),
        (DynVariable::Cuda(x), DynVariable::Cuda(w), None) => Ok(PyVariable::wrap(
            DynVariable::Cuda(F::conv2d(x, w, None, stride, padding).map_err(to_py)?),
        )),
        (DynVariable::Vulkan(x), DynVariable::Vulkan(w), None) => Ok(PyVariable::wrap(
            DynVariable::Vulkan(F::conv2d(x, w, None, stride, padding).map_err(to_py)?),
        )),
        (DynVariable::Cpu(x), DynVariable::Cpu(w), Some(DynVariable::Cpu(b))) => {
            Ok(PyVariable::wrap(DynVariable::Cpu(
                F::conv2d(x, w, Some(b), stride, padding).map_err(to_py)?,
            )))
        }
        (DynVariable::Cuda(x), DynVariable::Cuda(w), Some(DynVariable::Cuda(b))) => {
            Ok(PyVariable::wrap(DynVariable::Cuda(
                F::conv2d(x, w, Some(b), stride, padding).map_err(to_py)?,
            )))
        }
        (DynVariable::Vulkan(x), DynVariable::Vulkan(w), Some(DynVariable::Vulkan(b))) => {
            Ok(PyVariable::wrap(DynVariable::Vulkan(
                F::conv2d(x, w, Some(b), stride, padding).map_err(to_py)?,
            )))
        }
        _ => Err(PyRuntimeError::new_err("conv2d device mismatch")),
    }
}

#[pyfunction]
#[pyo3(signature = (x, weight, bias, stride=1, padding=0, output_padding=0))]
fn conv_transpose2d(
    x: PyRef<'_, PyVariable>,
    weight: PyRef<'_, PyVariable>,
    bias: Option<PyRef<'_, PyVariable>>,
    stride: usize,
    padding: usize,
    output_padding: usize,
) -> PyResult<PyVariable> {
    let bias = bias.as_ref().map(|b| &b.inner);
    match (&x.inner, &weight.inner, bias) {
        (DynVariable::Cpu(x), DynVariable::Cpu(w), None) => Ok(PyVariable::wrap(DynVariable::Cpu(
            F::conv_transpose2d(x, w, None, stride, padding, output_padding).map_err(to_py)?,
        ))),
        (DynVariable::Cuda(x), DynVariable::Cuda(w), None) => {
            Ok(PyVariable::wrap(DynVariable::Cuda(
                F::conv_transpose2d(x, w, None, stride, padding, output_padding).map_err(to_py)?,
            )))
        }
        (DynVariable::Vulkan(x), DynVariable::Vulkan(w), None) => {
            Ok(PyVariable::wrap(DynVariable::Vulkan(
                F::conv_transpose2d(x, w, None, stride, padding, output_padding).map_err(to_py)?,
            )))
        }
        (DynVariable::Cpu(x), DynVariable::Cpu(w), Some(DynVariable::Cpu(b))) => {
            Ok(PyVariable::wrap(DynVariable::Cpu(
                F::conv_transpose2d(x, w, Some(b), stride, padding, output_padding)
                    .map_err(to_py)?,
            )))
        }
        (DynVariable::Cuda(x), DynVariable::Cuda(w), Some(DynVariable::Cuda(b))) => {
            Ok(PyVariable::wrap(DynVariable::Cuda(
                F::conv_transpose2d(x, w, Some(b), stride, padding, output_padding)
                    .map_err(to_py)?,
            )))
        }
        (DynVariable::Vulkan(x), DynVariable::Vulkan(w), Some(DynVariable::Vulkan(b))) => {
            Ok(PyVariable::wrap(DynVariable::Vulkan(
                F::conv_transpose2d(x, w, Some(b), stride, padding, output_padding)
                    .map_err(to_py)?,
            )))
        }
        _ => Err(PyRuntimeError::new_err("conv_transpose2d device mismatch")),
    }
}

#[pyfunction]
#[pyo3(signature = (x, kernel, stride=1, padding=0))]
fn max_pool2d(
    x: PyRef<'_, PyVariable>,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> PyResult<PyVariable> {
    match &x.inner {
        DynVariable::Cpu(x) => Ok(PyVariable::wrap(DynVariable::Cpu(
            F::max_pool2d(x, kernel, stride, padding).map_err(to_py)?,
        ))),
        DynVariable::Cuda(x) => Ok(PyVariable::wrap(DynVariable::Cuda(
            F::max_pool2d(x, kernel, stride, padding).map_err(to_py)?,
        ))),
        DynVariable::Vulkan(x) => Ok(PyVariable::wrap(DynVariable::Vulkan(
            F::max_pool2d(x, kernel, stride, padding).map_err(to_py)?,
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (x, kernel, stride=1, padding=0))]
fn average_pool2d(
    x: PyRef<'_, PyVariable>,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> PyResult<PyVariable> {
    match &x.inner {
        DynVariable::Cpu(x) => Ok(PyVariable::wrap(DynVariable::Cpu(
            F::average_pool2d(x, kernel, stride, padding).map_err(to_py)?,
        ))),
        DynVariable::Cuda(x) => Ok(PyVariable::wrap(DynVariable::Cuda(
            F::average_pool2d(x, kernel, stride, padding).map_err(to_py)?,
        ))),
        DynVariable::Vulkan(x) => Ok(PyVariable::wrap(DynVariable::Vulkan(
            F::average_pool2d(x, kernel, stride, padding).map_err(to_py)?,
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (q, k, v, heads, causal=false))]
fn multi_head_attention(
    q: PyRef<'_, PyVariable>,
    k: PyRef<'_, PyVariable>,
    v: PyRef<'_, PyVariable>,
    heads: usize,
    causal: bool,
) -> PyResult<PyVariable> {
    match (&q.inner, &k.inner, &v.inner) {
        (DynVariable::Cpu(q), DynVariable::Cpu(k), DynVariable::Cpu(v)) => Ok(PyVariable::wrap(
            DynVariable::Cpu(F::multi_head_attention(q, k, v, heads, causal).map_err(to_py)?),
        )),
        (DynVariable::Cuda(q), DynVariable::Cuda(k), DynVariable::Cuda(v)) => Ok(PyVariable::wrap(
            DynVariable::Cuda(F::multi_head_attention(q, k, v, heads, causal).map_err(to_py)?),
        )),
        (DynVariable::Vulkan(q), DynVariable::Vulkan(k), DynVariable::Vulkan(v)) => {
            Ok(PyVariable::wrap(DynVariable::Vulkan(
                F::multi_head_attention(q, k, v, heads, causal).map_err(to_py)?,
            )))
        }
        _ => Err(PyRuntimeError::new_err("attention device mismatch")),
    }
}

#[pyfunction]
#[pyo3(signature = (logits, target))]
fn softmax_cross_entropy(
    logits: PyRef<'_, PyVariable>,
    target: PyRef<'_, PyVariable>,
) -> PyResult<PyVariable> {
    match (&logits.inner, &target.inner) {
        (DynVariable::Cpu(a), DynVariable::Cpu(b)) => Ok(PyVariable::wrap(DynVariable::Cpu(
            F::softmax_cross_entropy(a, b).map_err(to_py)?,
        ))),
        (DynVariable::Cuda(a), DynVariable::Cuda(b)) => Ok(PyVariable::wrap(DynVariable::Cuda(
            F::softmax_cross_entropy(a, b).map_err(to_py)?,
        ))),
        (DynVariable::Vulkan(a), DynVariable::Vulkan(b)) => Ok(PyVariable::wrap(
            DynVariable::Vulkan(F::softmax_cross_entropy(a, b).map_err(to_py)?),
        )),
        _ => Err(PyRuntimeError::new_err("operand device mismatch")),
    }
}

enum DynConv2d {
    Cpu(Conv2d<CpuBackend>),
    Cuda(Conv2d<CudaBackend>),
    Vulkan(Conv2d<VulkanBackend>),
}

#[pyclass(name = "Conv2d", module = "chainer.links")]
pub struct PyConv2d {
    inner: DynConv2d,
}

#[pymethods]
impl PyConv2d {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel, stride=1, padding=0, seed=0, device="cpu"))]
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        seed: u64,
        device: &str,
    ) -> PyResult<Self> {
        let target = parse_device(device)?;
        let inner = match target {
            Device::Cpu => DynConv2d::Cpu(
                Conv2d::new(in_channels, out_channels, kernel, stride, padding, seed)
                    .map_err(to_py)?,
            ),
            Device::Cuda(index) => {
                let backend = cuda_backend(index)?;
                DynConv2d::Cuda(
                    Conv2d::with_backend(
                        backend,
                        in_channels,
                        out_channels,
                        kernel,
                        stride,
                        padding,
                        seed,
                    )
                    .map_err(to_py)?,
                )
            }
            Device::Vulkan(index) => {
                let backend = vulkan_backend(index as usize)?;
                DynConv2d::Vulkan(
                    Conv2d::with_backend(
                        backend,
                        in_channels,
                        out_channels,
                        kernel,
                        stride,
                        padding,
                        seed,
                    )
                    .map_err(to_py)?,
                )
            }
        };
        Ok(Self { inner })
    }

    fn forward(&self, x: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
        match (&self.inner, &x.inner) {
            (DynConv2d::Cpu(l), DynVariable::Cpu(x)) => Ok(PyVariable::wrap(DynVariable::Cpu(
                l.forward(x).map_err(to_py)?,
            ))),
            (DynConv2d::Cuda(l), DynVariable::Cuda(x)) => Ok(PyVariable::wrap(DynVariable::Cuda(
                l.forward(x).map_err(to_py)?,
            ))),
            (DynConv2d::Vulkan(l), DynVariable::Vulkan(x)) => Ok(PyVariable::wrap(
                DynVariable::Vulkan(l.forward(x).map_err(to_py)?),
            )),
            _ => Err(PyRuntimeError::new_err("Conv2d device mismatch")),
        }
    }

    fn parameters(&self) -> Vec<PyVariable> {
        match &self.inner {
            DynConv2d::Cpu(l) => l
                .parameters()
                .into_iter()
                .map(|p| PyVariable::wrap(DynVariable::Cpu(p.variable)))
                .collect(),
            DynConv2d::Cuda(l) => l
                .parameters()
                .into_iter()
                .map(|p| PyVariable::wrap(DynVariable::Cuda(p.variable)))
                .collect(),
            DynConv2d::Vulkan(l) => l
                .parameters()
                .into_iter()
                .map(|p| PyVariable::wrap(DynVariable::Vulkan(p.variable)))
                .collect(),
        }
    }

    fn __call__(&self, x: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
        self.forward(x)
    }

    #[getter]
    #[allow(non_snake_case)]
    fn W(&self) -> PyVariable {
        match &self.inner {
            DynConv2d::Cpu(l) => PyVariable::wrap(DynVariable::Cpu(l.weight().clone())),
            DynConv2d::Cuda(l) => PyVariable::wrap(DynVariable::Cuda(l.weight().clone())),
            DynConv2d::Vulkan(l) => PyVariable::wrap(DynVariable::Vulkan(l.weight().clone())),
        }
    }
}

enum DynConvTranspose2d {
    Cpu(ConvTranspose2d<CpuBackend>),
    Cuda(ConvTranspose2d<CudaBackend>),
    Vulkan(ConvTranspose2d<VulkanBackend>),
}

#[pyclass(name = "ConvTranspose2d", module = "chainer.links")]
pub struct PyConvTranspose2d {
    inner: DynConvTranspose2d,
}

#[pymethods]
impl PyConvTranspose2d {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (in_channels, out_channels, kernel, stride=1, padding=0, output_padding=0, seed=0, device="cpu"))]
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        seed: u64,
        device: &str,
    ) -> PyResult<Self> {
        let target = parse_device(device)?;
        let inner = match target {
            Device::Cpu => DynConvTranspose2d::Cpu(
                ConvTranspose2d::new(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding,
                    output_padding,
                    seed,
                )
                .map_err(to_py)?,
            ),
            Device::Cuda(index) => {
                let backend = cuda_backend(index)?;
                DynConvTranspose2d::Cuda(
                    ConvTranspose2d::with_backend(
                        backend,
                        in_channels,
                        out_channels,
                        kernel,
                        stride,
                        padding,
                        output_padding,
                        seed,
                    )
                    .map_err(to_py)?,
                )
            }
            Device::Vulkan(index) => {
                let backend = vulkan_backend(index as usize)?;
                DynConvTranspose2d::Vulkan(
                    ConvTranspose2d::with_backend(
                        backend,
                        in_channels,
                        out_channels,
                        kernel,
                        stride,
                        padding,
                        output_padding,
                        seed,
                    )
                    .map_err(to_py)?,
                )
            }
        };
        Ok(Self { inner })
    }

    fn forward(&self, x: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
        match (&self.inner, &x.inner) {
            (DynConvTranspose2d::Cpu(l), DynVariable::Cpu(x)) => Ok(PyVariable::wrap(
                DynVariable::Cpu(l.forward(x).map_err(to_py)?),
            )),
            (DynConvTranspose2d::Cuda(l), DynVariable::Cuda(x)) => Ok(PyVariable::wrap(
                DynVariable::Cuda(l.forward(x).map_err(to_py)?),
            )),
            (DynConvTranspose2d::Vulkan(l), DynVariable::Vulkan(x)) => Ok(PyVariable::wrap(
                DynVariable::Vulkan(l.forward(x).map_err(to_py)?),
            )),
            _ => Err(PyRuntimeError::new_err("ConvTranspose2d device mismatch")),
        }
    }

    fn parameters(&self) -> Vec<PyVariable> {
        match &self.inner {
            DynConvTranspose2d::Cpu(l) => l
                .parameters()
                .into_iter()
                .map(|p| PyVariable::wrap(DynVariable::Cpu(p.variable)))
                .collect(),
            DynConvTranspose2d::Cuda(l) => l
                .parameters()
                .into_iter()
                .map(|p| PyVariable::wrap(DynVariable::Cuda(p.variable)))
                .collect(),
            DynConvTranspose2d::Vulkan(l) => l
                .parameters()
                .into_iter()
                .map(|p| PyVariable::wrap(DynVariable::Vulkan(p.variable)))
                .collect(),
        }
    }

    fn __call__(&self, x: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
        self.forward(x)
    }
}

enum DynBatchNorm {
    Cpu(BatchNorm<CpuBackend>),
    Cuda(BatchNorm<CudaBackend>),
    Vulkan(BatchNorm<VulkanBackend>),
}

#[pyclass(name = "BatchNorm", module = "chainer.links")]
pub struct PyBatchNorm {
    inner: DynBatchNorm,
}

#[pymethods]
impl PyBatchNorm {
    #[new]
    #[pyo3(signature = (channels, eps=1e-5, device="cpu"))]
    fn new(channels: usize, eps: f64, device: &str) -> PyResult<Self> {
        let target = parse_device(device)?;
        let inner = match target {
            Device::Cpu => DynBatchNorm::Cpu(BatchNorm::new(channels, eps).map_err(to_py)?),
            Device::Cuda(index) => DynBatchNorm::Cuda(
                BatchNorm::with_backend(cuda_backend(index)?, channels, eps).map_err(to_py)?,
            ),
            Device::Vulkan(index) => DynBatchNorm::Vulkan(
                BatchNorm::with_backend(vulkan_backend(index as usize)?, channels, eps)
                    .map_err(to_py)?,
            ),
        };
        Ok(Self { inner })
    }

    fn forward(&self, x: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
        match (&self.inner, &x.inner) {
            (DynBatchNorm::Cpu(l), DynVariable::Cpu(x)) => Ok(PyVariable::wrap(DynVariable::Cpu(
                l.forward(x).map_err(to_py)?,
            ))),
            (DynBatchNorm::Cuda(l), DynVariable::Cuda(x)) => Ok(PyVariable::wrap(
                DynVariable::Cuda(l.forward(x).map_err(to_py)?),
            )),
            (DynBatchNorm::Vulkan(l), DynVariable::Vulkan(x)) => Ok(PyVariable::wrap(
                DynVariable::Vulkan(l.forward(x).map_err(to_py)?),
            )),
            _ => Err(PyRuntimeError::new_err("BatchNorm device mismatch")),
        }
    }

    fn parameters(&self) -> Vec<PyVariable> {
        match &self.inner {
            DynBatchNorm::Cpu(l) => l
                .parameters()
                .into_iter()
                .map(|p| PyVariable::wrap(DynVariable::Cpu(p.variable)))
                .collect(),
            DynBatchNorm::Cuda(l) => l
                .parameters()
                .into_iter()
                .map(|p| PyVariable::wrap(DynVariable::Cuda(p.variable)))
                .collect(),
            DynBatchNorm::Vulkan(l) => l
                .parameters()
                .into_iter()
                .map(|p| PyVariable::wrap(DynVariable::Vulkan(p.variable)))
                .collect(),
        }
    }

    fn __call__(&self, x: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
        self.forward(x)
    }
}

enum DynLayerNorm {
    Cpu(LayerNorm<CpuBackend>),
    Cuda(LayerNorm<CudaBackend>),
    Vulkan(LayerNorm<VulkanBackend>),
}

#[pyclass(name = "LayerNorm", module = "chainer.links")]
pub struct PyLayerNorm {
    inner: DynLayerNorm,
}

#[pymethods]
impl PyLayerNorm {
    #[new]
    #[pyo3(signature = (dim, eps=1e-5, device="cpu"))]
    fn new(dim: usize, eps: f64, device: &str) -> PyResult<Self> {
        let target = parse_device(device)?;
        let inner = match target {
            Device::Cpu => DynLayerNorm::Cpu(LayerNorm::new(dim, eps).map_err(to_py)?),
            Device::Cuda(index) => DynLayerNorm::Cuda(
                LayerNorm::with_backend(cuda_backend(index)?, dim, eps).map_err(to_py)?,
            ),
            Device::Vulkan(index) => DynLayerNorm::Vulkan(
                LayerNorm::with_backend(vulkan_backend(index as usize)?, dim, eps)
                    .map_err(to_py)?,
            ),
        };
        Ok(Self { inner })
    }

    fn forward(&self, x: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
        match (&self.inner, &x.inner) {
            (DynLayerNorm::Cpu(l), DynVariable::Cpu(x)) => Ok(PyVariable::wrap(DynVariable::Cpu(
                l.forward(x).map_err(to_py)?,
            ))),
            (DynLayerNorm::Cuda(l), DynVariable::Cuda(x)) => Ok(PyVariable::wrap(
                DynVariable::Cuda(l.forward(x).map_err(to_py)?),
            )),
            (DynLayerNorm::Vulkan(l), DynVariable::Vulkan(x)) => Ok(PyVariable::wrap(
                DynVariable::Vulkan(l.forward(x).map_err(to_py)?),
            )),
            _ => Err(PyRuntimeError::new_err("LayerNorm device mismatch")),
        }
    }

    fn parameters(&self) -> Vec<PyVariable> {
        match &self.inner {
            DynLayerNorm::Cpu(l) => l
                .parameters()
                .into_iter()
                .map(|p| PyVariable::wrap(DynVariable::Cpu(p.variable)))
                .collect(),
            DynLayerNorm::Cuda(l) => l
                .parameters()
                .into_iter()
                .map(|p| PyVariable::wrap(DynVariable::Cuda(p.variable)))
                .collect(),
            DynLayerNorm::Vulkan(l) => l
                .parameters()
                .into_iter()
                .map(|p| PyVariable::wrap(DynVariable::Vulkan(p.variable)))
                .collect(),
        }
    }

    fn __call__(&self, x: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
        self.forward(x)
    }
}

enum DynRmsNorm {
    Cpu(RMSNorm<CpuBackend>),
    Cuda(RMSNorm<CudaBackend>),
    Vulkan(RMSNorm<VulkanBackend>),
}

#[pyclass(name = "RMSNorm", module = "chainer.links")]
pub struct PyRmsNorm {
    inner: DynRmsNorm,
}

#[pymethods]
impl PyRmsNorm {
    #[new]
    #[pyo3(signature = (dim, eps=1e-6, device="cpu"))]
    fn new(dim: usize, eps: f64, device: &str) -> PyResult<Self> {
        let target = parse_device(device)?;
        let inner = match target {
            Device::Cpu => DynRmsNorm::Cpu(RMSNorm::new(dim, eps).map_err(to_py)?),
            Device::Cuda(index) => DynRmsNorm::Cuda(
                RMSNorm::with_backend(cuda_backend(index)?, dim, eps).map_err(to_py)?,
            ),
            Device::Vulkan(index) => DynRmsNorm::Vulkan(
                RMSNorm::with_backend(vulkan_backend(index as usize)?, dim, eps).map_err(to_py)?,
            ),
        };
        Ok(Self { inner })
    }

    fn forward(&self, x: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
        match (&self.inner, &x.inner) {
            (DynRmsNorm::Cpu(l), DynVariable::Cpu(x)) => Ok(PyVariable::wrap(DynVariable::Cpu(
                l.forward(x).map_err(to_py)?,
            ))),
            (DynRmsNorm::Cuda(l), DynVariable::Cuda(x)) => Ok(PyVariable::wrap(DynVariable::Cuda(
                l.forward(x).map_err(to_py)?,
            ))),
            (DynRmsNorm::Vulkan(l), DynVariable::Vulkan(x)) => Ok(PyVariable::wrap(
                DynVariable::Vulkan(l.forward(x).map_err(to_py)?),
            )),
            _ => Err(PyRuntimeError::new_err("RMSNorm device mismatch")),
        }
    }

    fn parameters(&self) -> Vec<PyVariable> {
        match &self.inner {
            DynRmsNorm::Cpu(l) => l
                .parameters()
                .into_iter()
                .map(|p| PyVariable::wrap(DynVariable::Cpu(p.variable)))
                .collect(),
            DynRmsNorm::Cuda(l) => l
                .parameters()
                .into_iter()
                .map(|p| PyVariable::wrap(DynVariable::Cuda(p.variable)))
                .collect(),
            DynRmsNorm::Vulkan(l) => l
                .parameters()
                .into_iter()
                .map(|p| PyVariable::wrap(DynVariable::Vulkan(p.variable)))
                .collect(),
        }
    }

    fn __call__(&self, x: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
        self.forward(x)
    }
}

enum DynEmbedding {
    Cpu(Embedding<CpuBackend>),
    Cuda(Embedding<CudaBackend>),
    Vulkan(Embedding<VulkanBackend>),
}

#[pyclass(name = "Embedding", module = "chainer.links")]
pub struct PyEmbedding {
    inner: DynEmbedding,
}

#[pymethods]
impl PyEmbedding {
    #[new]
    #[pyo3(signature = (vocab, dim, seed=0, device="cpu"))]
    fn new(vocab: usize, dim: usize, seed: u64, device: &str) -> PyResult<Self> {
        let target = parse_device(device)?;
        let inner = match target {
            Device::Cpu => DynEmbedding::Cpu(Embedding::new(vocab, dim, seed).map_err(to_py)?),
            Device::Cuda(index) => DynEmbedding::Cuda(
                Embedding::with_backend(cuda_backend(index)?, vocab, dim, seed).map_err(to_py)?,
            ),
            Device::Vulkan(index) => DynEmbedding::Vulkan(
                Embedding::with_backend(vulkan_backend(index as usize)?, vocab, dim, seed)
                    .map_err(to_py)?,
            ),
        };
        Ok(Self { inner })
    }

    fn forward(&self, ids: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
        match (&self.inner, &ids.inner) {
            (DynEmbedding::Cpu(l), DynVariable::Cpu(x)) => Ok(PyVariable::wrap(DynVariable::Cpu(
                l.forward(x).map_err(to_py)?,
            ))),
            (DynEmbedding::Cuda(l), DynVariable::Cuda(x)) => Ok(PyVariable::wrap(
                DynVariable::Cuda(l.forward(x).map_err(to_py)?),
            )),
            (DynEmbedding::Vulkan(l), DynVariable::Vulkan(x)) => Ok(PyVariable::wrap(
                DynVariable::Vulkan(l.forward(x).map_err(to_py)?),
            )),
            _ => Err(PyRuntimeError::new_err("Embedding device mismatch")),
        }
    }

    fn parameters(&self) -> Vec<PyVariable> {
        match &self.inner {
            DynEmbedding::Cpu(l) => l
                .parameters()
                .into_iter()
                .map(|p| PyVariable::wrap(DynVariable::Cpu(p.variable)))
                .collect(),
            DynEmbedding::Cuda(l) => l
                .parameters()
                .into_iter()
                .map(|p| PyVariable::wrap(DynVariable::Cuda(p.variable)))
                .collect(),
            DynEmbedding::Vulkan(l) => l
                .parameters()
                .into_iter()
                .map(|p| PyVariable::wrap(DynVariable::Vulkan(p.variable)))
                .collect(),
        }
    }

    fn __call__(&self, ids: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
        self.forward(ids)
    }
}

enum DynMultiHeadAttention {
    Cpu(MultiHeadAttention<CpuBackend>),
    Cuda(MultiHeadAttention<CudaBackend>),
    Vulkan(MultiHeadAttention<VulkanBackend>),
}

#[pyclass(name = "MultiHeadAttention", module = "chainer.links")]
pub struct PyMultiHeadAttention {
    inner: DynMultiHeadAttention,
}

#[pymethods]
impl PyMultiHeadAttention {
    #[new]
    #[pyo3(signature = (dim, heads, causal=false, device="cpu"))]
    fn new(dim: usize, heads: usize, causal: bool, device: &str) -> PyResult<Self> {
        let target = parse_device(device)?;
        let inner = match target {
            Device::Cpu => DynMultiHeadAttention::Cpu(
                MultiHeadAttention::new(dim, heads, causal).map_err(to_py)?,
            ),
            Device::Cuda(index) => DynMultiHeadAttention::Cuda(
                MultiHeadAttention::with_backend(cuda_backend(index)?, dim, heads, causal)
                    .map_err(to_py)?,
            ),
            Device::Vulkan(index) => DynMultiHeadAttention::Vulkan(
                MultiHeadAttention::with_backend(
                    vulkan_backend(index as usize)?,
                    dim,
                    heads,
                    causal,
                )
                .map_err(to_py)?,
            ),
        };
        Ok(Self { inner })
    }

    fn forward(&self, x: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
        match (&self.inner, &x.inner) {
            (DynMultiHeadAttention::Cpu(l), DynVariable::Cpu(x)) => Ok(PyVariable::wrap(
                DynVariable::Cpu(l.forward(x).map_err(to_py)?),
            )),
            (DynMultiHeadAttention::Cuda(l), DynVariable::Cuda(x)) => Ok(PyVariable::wrap(
                DynVariable::Cuda(l.forward(x).map_err(to_py)?),
            )),
            (DynMultiHeadAttention::Vulkan(l), DynVariable::Vulkan(x)) => Ok(PyVariable::wrap(
                DynVariable::Vulkan(l.forward(x).map_err(to_py)?),
            )),
            _ => Err(PyRuntimeError::new_err(
                "MultiHeadAttention device mismatch",
            )),
        }
    }

    fn parameters(&self) -> Vec<PyVariable> {
        match &self.inner {
            DynMultiHeadAttention::Cpu(l) => l
                .parameters()
                .into_iter()
                .map(|p| PyVariable::wrap(DynVariable::Cpu(p.variable)))
                .collect(),
            DynMultiHeadAttention::Cuda(l) => l
                .parameters()
                .into_iter()
                .map(|p| PyVariable::wrap(DynVariable::Cuda(p.variable)))
                .collect(),
            DynMultiHeadAttention::Vulkan(l) => l
                .parameters()
                .into_iter()
                .map(|p| PyVariable::wrap(DynVariable::Vulkan(p.variable)))
                .collect(),
        }
    }

    fn __call__(&self, x: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
        self.forward(x)
    }
}

#[pyclass(name = "SGD", module = "chainer.optimizers")]
pub struct PySGD {
    inner: SGD,
    cpu: Vec<Parameter<CpuBackend>>,
    cuda: Vec<Parameter<CudaBackend>>,
    vulkan: Vec<Parameter<VulkanBackend>>,
}

#[pymethods]
impl PySGD {
    #[new]
    fn new(lr: f64) -> Self {
        Self {
            inner: SGD::new(lr),
            cpu: Vec::new(),
            cuda: Vec::new(),
            vulkan: Vec::new(),
        }
    }

    fn setup(&mut self, params: Vec<PyRef<'_, PyVariable>>) {
        self.cpu.clear();
        self.cuda.clear();
        self.vulkan.clear();
        for (i, variable) in params.into_iter().enumerate() {
            let name = format!("param{i}");
            match variable.inner.clone() {
                DynVariable::Cpu(v) => self.cpu.push(Parameter::new(name, v)),
                DynVariable::Cuda(v) => self.cuda.push(Parameter::new(name, v)),
                DynVariable::Vulkan(v) => self.vulkan.push(Parameter::new(name, v)),
            }
        }
    }

    fn update(&mut self) -> PyResult<()> {
        if !self.cpu.is_empty() {
            self.inner.update(&self.cpu).map_err(to_py)?;
        }
        if !self.cuda.is_empty() {
            self.inner.update(&self.cuda).map_err(to_py)?;
        }
        if !self.vulkan.is_empty() {
            self.inner.update(&self.vulkan).map_err(to_py)?;
        }
        Ok(())
    }

    fn zero_grad(&self) {
        modern_chainer::zero_grad(&self.cpu);
        modern_chainer::zero_grad(&self.cuda);
        modern_chainer::zero_grad(&self.vulkan);
    }
}

#[pyclass(name = "Adam", module = "chainer.optimizers")]
pub struct PyAdam {
    cpu: Adam<CpuBackend>,
    cuda: Adam<CudaBackend>,
    vulkan: Adam<VulkanBackend>,
    cpu_params: Vec<Parameter<CpuBackend>>,
    cuda_params: Vec<Parameter<CudaBackend>>,
    vulkan_params: Vec<Parameter<VulkanBackend>>,
}

#[pymethods]
impl PyAdam {
    #[new]
    #[pyo3(signature = (lr=0.001))]
    fn new(lr: f64) -> Self {
        Self {
            cpu: Adam::new(lr),
            cuda: Adam::new(lr),
            vulkan: Adam::new(lr),
            cpu_params: Vec::new(),
            cuda_params: Vec::new(),
            vulkan_params: Vec::new(),
        }
    }

    fn setup(&mut self, params: Vec<PyRef<'_, PyVariable>>) {
        self.cpu_params.clear();
        self.cuda_params.clear();
        self.vulkan_params.clear();
        for (i, variable) in params.into_iter().enumerate() {
            let name = format!("param{i}");
            match variable.inner.clone() {
                DynVariable::Cpu(v) => self.cpu_params.push(Parameter::new(name, v)),
                DynVariable::Cuda(v) => self.cuda_params.push(Parameter::new(name, v)),
                DynVariable::Vulkan(v) => self.vulkan_params.push(Parameter::new(name, v)),
            }
        }
    }

    fn update(&mut self) -> PyResult<()> {
        if !self.cpu_params.is_empty() {
            self.cpu.update(&self.cpu_params).map_err(to_py)?;
        }
        if !self.cuda_params.is_empty() {
            self.cuda.update(&self.cuda_params).map_err(to_py)?;
        }
        if !self.vulkan_params.is_empty() {
            self.vulkan.update(&self.vulkan_params).map_err(to_py)?;
        }
        Ok(())
    }

    fn zero_grad(&self) {
        modern_chainer::zero_grad(&self.cpu_params);
        modern_chainer::zero_grad(&self.cuda_params);
        modern_chainer::zero_grad(&self.vulkan_params);
    }
}

#[pyclass(name = "MomentumSGD", module = "chainer.optimizers")]
pub struct PyMomentumSGD {
    cpu: MomentumSGD<CpuBackend>,
    cuda: MomentumSGD<CudaBackend>,
    vulkan: MomentumSGD<VulkanBackend>,
    cpu_params: Vec<Parameter<CpuBackend>>,
    cuda_params: Vec<Parameter<CudaBackend>>,
    vulkan_params: Vec<Parameter<VulkanBackend>>,
}

#[pymethods]
impl PyMomentumSGD {
    #[new]
    #[pyo3(signature = (lr=0.01, momentum=0.9))]
    fn new(lr: f64, momentum: f64) -> Self {
        Self {
            cpu: MomentumSGD::new(lr, momentum),
            cuda: MomentumSGD::new(lr, momentum),
            vulkan: MomentumSGD::new(lr, momentum),
            cpu_params: Vec::new(),
            cuda_params: Vec::new(),
            vulkan_params: Vec::new(),
        }
    }
    fn setup(&mut self, params: Vec<PyRef<'_, PyVariable>>) {
        self.cpu_params.clear();
        self.cuda_params.clear();
        self.vulkan_params.clear();
        for (i, variable) in params.into_iter().enumerate() {
            let name = format!("param{i}");
            match variable.inner.clone() {
                DynVariable::Cpu(v) => self.cpu_params.push(Parameter::new(name, v)),
                DynVariable::Cuda(v) => self.cuda_params.push(Parameter::new(name, v)),
                DynVariable::Vulkan(v) => self.vulkan_params.push(Parameter::new(name, v)),
            }
        }
    }
    fn update(&mut self) -> PyResult<()> {
        if !self.cpu_params.is_empty() {
            self.cpu.update(&self.cpu_params).map_err(to_py)?;
        }
        if !self.cuda_params.is_empty() {
            self.cuda.update(&self.cuda_params).map_err(to_py)?;
        }
        if !self.vulkan_params.is_empty() {
            self.vulkan.update(&self.vulkan_params).map_err(to_py)?;
        }
        Ok(())
    }
    fn zero_grad(&self) {
        modern_chainer::zero_grad(&self.cpu_params);
        modern_chainer::zero_grad(&self.cuda_params);
        modern_chainer::zero_grad(&self.vulkan_params);
    }
}

#[pyclass(name = "AdamW", module = "chainer.optimizers")]
pub struct PyAdamW {
    cpu: AdamW<CpuBackend>,
    cuda: AdamW<CudaBackend>,
    vulkan: AdamW<VulkanBackend>,
    cpu_params: Vec<Parameter<CpuBackend>>,
    cuda_params: Vec<Parameter<CudaBackend>>,
    vulkan_params: Vec<Parameter<VulkanBackend>>,
}

#[pymethods]
impl PyAdamW {
    #[new]
    #[pyo3(signature = (lr=0.001, weight_decay=0.01))]
    fn new(lr: f64, weight_decay: f64) -> Self {
        Self {
            cpu: AdamW::new(lr, weight_decay),
            cuda: AdamW::new(lr, weight_decay),
            vulkan: AdamW::new(lr, weight_decay),
            cpu_params: Vec::new(),
            cuda_params: Vec::new(),
            vulkan_params: Vec::new(),
        }
    }
    fn setup(&mut self, params: Vec<PyRef<'_, PyVariable>>) {
        self.cpu_params.clear();
        self.cuda_params.clear();
        self.vulkan_params.clear();
        for (i, variable) in params.into_iter().enumerate() {
            let name = format!("param{i}");
            match variable.inner.clone() {
                DynVariable::Cpu(v) => self.cpu_params.push(Parameter::new(name, v)),
                DynVariable::Cuda(v) => self.cuda_params.push(Parameter::new(name, v)),
                DynVariable::Vulkan(v) => self.vulkan_params.push(Parameter::new(name, v)),
            }
        }
    }
    fn update(&mut self) -> PyResult<()> {
        if !self.cpu_params.is_empty() {
            self.cpu.update(&self.cpu_params).map_err(to_py)?;
        }
        if !self.cuda_params.is_empty() {
            self.cuda.update(&self.cuda_params).map_err(to_py)?;
        }
        if !self.vulkan_params.is_empty() {
            self.vulkan.update(&self.vulkan_params).map_err(to_py)?;
        }
        Ok(())
    }
    fn zero_grad(&self) {
        modern_chainer::zero_grad(&self.cpu_params);
        modern_chainer::zero_grad(&self.cuda_params);
        modern_chainer::zero_grad(&self.vulkan_params);
    }
}

#[pyclass(name = "LinearWarmupCosineLR", module = "chainer.optimizers")]
pub struct PyLrScheduler {
    inner: LinearWarmupCosineLR,
}

#[pymethods]
impl PyLrScheduler {
    #[new]
    #[pyo3(signature = (warmup_steps=100, total_steps=1000, base_lr=0.001, min_lr=0.0))]
    fn new(warmup_steps: u64, total_steps: u64, base_lr: f64, min_lr: f64) -> Self {
        Self {
            inner: LinearWarmupCosineLR {
                warmup_steps,
                total_steps,
                base_lr,
                min_lr,
            },
        }
    }
    fn get_lr(&self, step: u64) -> f64 {
        self.inner.get_lr(step)
    }
}

#[pyfunction]
fn cuda_available() -> bool {
    modern_chainer::cuda::is_available()
}

#[pyfunction]
fn vulkan_available() -> bool {
    modern_chainer::vulkan::is_available()
}

#[pyfunction]
#[pyo3(signature = (device="cpu"))]
fn synchronize(device: &str) -> PyResult<()> {
    match parse_device(device)? {
        Device::Cpu => Ok(()),
        Device::Cuda(index) => cuda_backend(index)?.synchronize().map_err(to_py),
        Device::Vulkan(index) => vulkan_backend(index as usize)?.synchronize().map_err(to_py),
    }
}

#[pyfunction]
fn vulkan_devices() -> Vec<String> {
    modern_chainer::vulkan::probe()
        .adapters
        .into_iter()
        .map(|adapter| format!("{} ({})", adapter.name, adapter.device_type))
        .collect()
}

#[pyfunction]
fn version() -> &'static str {
    modern_chainer::VERSION
}

#[pymodule]
fn _chainer(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", modern_chainer::VERSION)?;
    module.add_class::<PyTensor>()?;
    module.add_class::<PyVariable>()?;
    module.add_class::<PyLinear>()?;
    module.add_class::<PyConv2d>()?;
    module.add_class::<PyConvTranspose2d>()?;
    module.add_class::<PyBatchNorm>()?;
    module.add_class::<PyLayerNorm>()?;
    module.add_class::<PyEmbedding>()?;
    module.add_class::<PyMultiHeadAttention>()?;
    module.add_class::<PyRmsNorm>()?;
    module.add_class::<PySGD>()?;
    module.add_class::<PyAdam>()?;
    module.add_class::<PyMomentumSGD>()?;
    module.add_class::<PyAdamW>()?;
    module.add_class::<PyLrScheduler>()?;
    module.add_function(wrap_pyfunction!(relu, module)?)?;
    module.add_function(wrap_pyfunction!(add, module)?)?;
    module.add_function(wrap_pyfunction!(sub, module)?)?;
    module.add_function(wrap_pyfunction!(mul, module)?)?;
    module.add_function(wrap_pyfunction!(matmul, module)?)?;
    module.add_function(wrap_pyfunction!(div, module)?)?;
    module.add_function(wrap_pyfunction!(sqrt, module)?)?;
    module.add_function(wrap_pyfunction!(transpose, module)?)?;
    module.add_function(wrap_pyfunction!(neg, module)?)?;
    module.add_function(wrap_pyfunction!(sum, module)?)?;
    module.add_function(wrap_pyfunction!(mean, module)?)?;
    module.add_function(wrap_pyfunction!(exp, module)?)?;
    module.add_function(wrap_pyfunction!(log, module)?)?;
    module.add_function(wrap_pyfunction!(sigmoid, module)?)?;
    module.add_function(wrap_pyfunction!(softmax, module)?)?;
    module.add_function(wrap_pyfunction!(reshape, module)?)?;
    module.add_function(wrap_pyfunction!(permute, module)?)?;
    module.add_function(wrap_pyfunction!(layer_norm, module)?)?;
    module.add_function(wrap_pyfunction!(batch_norm, module)?)?;
    module.add_function(wrap_pyfunction!(embedding, module)?)?;
    module.add_function(wrap_pyfunction!(conv2d, module)?)?;
    module.add_function(wrap_pyfunction!(conv_transpose2d, module)?)?;
    module.add_function(wrap_pyfunction!(max_pool2d, module)?)?;
    module.add_function(wrap_pyfunction!(average_pool2d, module)?)?;
    module.add_function(wrap_pyfunction!(multi_head_attention, module)?)?;
    module.add_function(wrap_pyfunction!(softmax_cross_entropy, module)?)?;
    module.add_function(wrap_pyfunction!(silu, module)?)?;
    module.add_function(wrap_pyfunction!(gelu, module)?)?;
    module.add_function(wrap_pyfunction!(abs, module)?)?;
    module.add_function(wrap_pyfunction!(scale, module)?)?;
    module.add_function(wrap_pyfunction!(clamp, module)?)?;
    module.add_function(wrap_pyfunction!(rms_norm, module)?)?;
    module.add_function(wrap_pyfunction!(linear, module)?)?;
    module.add_function(wrap_pyfunction!(dropout, module)?)?;
    module.add_function(wrap_pyfunction!(cuda_available, module)?)?;
    module.add_function(wrap_pyfunction!(vulkan_available, module)?)?;
    module.add_function(wrap_pyfunction!(vulkan_devices, module)?)?;
    module.add_function(wrap_pyfunction!(synchronize, module)?)?;
    module.add_function(wrap_pyfunction!(version, module)?)?;
    Ok(())
}
