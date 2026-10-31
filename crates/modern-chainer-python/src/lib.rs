use std::collections::HashMap;
use std::str::FromStr;
use std::sync::{Mutex, OnceLock};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};

use modern_chainer::optimizers::Optimizer;
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
    module.add_class::<PySGD>()?;
    module.add_class::<PyAdam>()?;
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
    module.add_function(wrap_pyfunction!(softmax_cross_entropy, module)?)?;
    module.add_function(wrap_pyfunction!(cuda_available, module)?)?;
    module.add_function(wrap_pyfunction!(vulkan_available, module)?)?;
    module.add_function(wrap_pyfunction!(vulkan_devices, module)?)?;
    module.add_function(wrap_pyfunction!(synchronize, module)?)?;
    module.add_function(wrap_pyfunction!(version, module)?)?;
    Ok(())
}
