use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};

use modern_chainer::optimizers::Optimizer;
use modern_chainer::prelude::*;
use modern_chainer_cpu::CpuTensorExt;

fn to_py(error: modern_chainer::Error) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

fn numpy_array<'py>(py: Python<'py>, tensor: &Tensor<CpuBackend>) -> PyResult<Bound<'py, PyAny>> {
    let numpy = py.import("numpy")?;
    let bytes = tensor.to_bytes().map_err(to_py)?;
    let buffer = PyBytes::new(py, &bytes);
    let array = numpy.call_method1("frombuffer", (buffer, "float32"))?;
    let shape = PyTuple::new(py, tensor.shape().dims())?;
    array.call_method1("reshape", (shape,))
}

#[pyclass(name = "Tensor", module = "chainer")]
#[derive(Clone)]
pub struct PyTensor {
    inner: Tensor<CpuBackend>,
}

#[pymethods]
impl PyTensor {
    #[staticmethod]
    fn zeros(shape: Vec<usize>) -> PyResult<Self> {
        Ok(Self {
            inner: Tensor::zeros(CpuBackend::new(), shape, DType::F32).map_err(to_py)?,
        })
    }

    #[staticmethod]
    fn from_list(data: Vec<f64>, shape: Vec<usize>) -> PyResult<Self> {
        let values: Vec<f32> = data.into_iter().map(|x| x as f32).collect();
        Ok(Self {
            inner: Tensor::from_f32(CpuBackend::new(), &values, shape).map_err(to_py)?,
        })
    }

    #[staticmethod]
    fn from_numpy(object: &Bound<'_, PyAny>) -> PyResult<Self> {
        let shape: Vec<usize> = object.getattr("shape")?.extract()?;
        let dtype: String = object.getattr("dtype")?.getattr("name")?.extract()?;
        let (array, kind) = match dtype.as_str() {
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
        };
        let bytes: Vec<u8> = array.call_method0("tobytes")?.extract()?;
        Ok(Self {
            inner: Tensor::from_bytes(CpuBackend::new(), &bytes, shape, kind).map_err(to_py)?,
        })
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().dims().to_vec()
    }

    #[getter]
    fn dtype(&self) -> String {
        self.inner.dtype().to_string()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.rank()
    }

    fn numel(&self) -> usize {
        self.inner.numel()
    }

    fn tolist(&self) -> PyResult<Vec<f32>> {
        self.inner.to_vec_f32().map_err(to_py)
    }

    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        numpy_array(py, &self.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, dtype={})",
            self.inner.shape().dims(),
            self.inner.dtype()
        )
    }
}

#[pyclass(name = "Variable", module = "chainer")]
#[derive(Clone)]
pub struct PyVariable {
    inner: Variable<CpuBackend>,
}

impl PyVariable {
    fn wrap(inner: Variable<CpuBackend>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyVariable {
    #[new]
    fn new(tensor: PyRef<'_, PyTensor>) -> Self {
        Self::wrap(Variable::new(tensor.inner.clone()))
    }

    #[staticmethod]
    fn constant(tensor: PyRef<'_, PyTensor>) -> Self {
        Self::wrap(Variable::constant(tensor.inner.clone()))
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
    fn grad(&self) -> Option<PyTensor> {
        self.inner.grad().map(|inner| PyTensor { inner })
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.data().shape().dims().to_vec()
    }

    #[getter]
    fn grad_fn(&self) -> Option<String> {
        self.inner.grad_fn_name().map(str::to_owned)
    }

    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    #[setter]
    fn set_requires_grad(&self, value: bool) {
        self.inner.set_requires_grad(value);
    }

    fn backward(&self) -> PyResult<()> {
        self.inner.backward().map_err(to_py)
    }

    fn cleargrad(&self) {
        self.inner.cleargrad();
    }

    fn unchain_backward(&self) {
        self.inner.unchain_backward();
    }

    fn __repr__(&self) -> String {
        format!(
            "Variable(shape={:?}, requires_grad={})",
            self.inner.data().shape().dims(),
            self.inner.requires_grad()
        )
    }
}

#[pyclass(name = "Linear", module = "chainer.links")]
pub struct PyLinear {
    inner: Linear<CpuBackend>,
}

#[pymethods]
impl PyLinear {
    #[new]
    #[pyo3(signature = (in_features, out_features, seed=0))]
    fn new(in_features: usize, out_features: usize, seed: u64) -> PyResult<Self> {
        Ok(Self {
            inner: Linear::new(in_features, out_features, seed).map_err(to_py)?,
        })
    }

    fn forward(&self, x: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
        Ok(PyVariable::wrap(
            self.inner.forward(&x.inner).map_err(to_py)?,
        ))
    }

    fn parameters(&self) -> Vec<PyVariable> {
        self.inner
            .parameters()
            .into_iter()
            .map(|parameter| PyVariable::wrap(parameter.variable))
            .collect()
    }

    fn cleargrads(&self) {
        for parameter in self.inner.parameters() {
            parameter.variable.cleargrad();
        }
    }

    #[getter]
    #[allow(non_snake_case)]
    fn W(&self) -> PyVariable {
        PyVariable::wrap(self.inner.weight().clone())
    }

    #[getter]
    fn b(&self) -> Option<PyVariable> {
        self.inner.bias().cloned().map(PyVariable::wrap)
    }

    fn __call__(&self, x: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
        self.forward(x)
    }

    fn __repr__(&self) -> String {
        format!(
            "Linear(in_features={}, out_features={})",
            self.inner.in_features(),
            self.inner.out_features()
        )
    }
}

#[pyfunction]
fn relu(x: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
    Ok(PyVariable::wrap(F::relu(&x.inner).map_err(to_py)?))
}

#[pyfunction]
fn add(a: PyRef<'_, PyVariable>, b: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
    Ok(PyVariable::wrap(F::add(&a.inner, &b.inner).map_err(to_py)?))
}

#[pyfunction]
fn sub(a: PyRef<'_, PyVariable>, b: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
    Ok(PyVariable::wrap(F::sub(&a.inner, &b.inner).map_err(to_py)?))
}

#[pyfunction]
fn mul(a: PyRef<'_, PyVariable>, b: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
    Ok(PyVariable::wrap(F::mul(&a.inner, &b.inner).map_err(to_py)?))
}

#[pyfunction]
fn matmul(a: PyRef<'_, PyVariable>, b: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
    Ok(PyVariable::wrap(
        F::matmul(&a.inner, &b.inner).map_err(to_py)?,
    ))
}

#[pyfunction]
fn div(a: PyRef<'_, PyVariable>, b: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
    Ok(PyVariable::wrap(F::div(&a.inner, &b.inner).map_err(to_py)?))
}

#[pyfunction]
fn sqrt(x: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
    Ok(PyVariable::wrap(F::sqrt(&x.inner).map_err(to_py)?))
}

#[pyfunction]
fn transpose(x: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
    Ok(PyVariable::wrap(F::transpose(&x.inner).map_err(to_py)?))
}

#[pyfunction]
fn neg(x: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
    Ok(PyVariable::wrap(F::neg(&x.inner).map_err(to_py)?))
}

#[pyfunction]
fn sum(x: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
    Ok(PyVariable::wrap(F::sum(&x.inner).map_err(to_py)?))
}

#[pyfunction]
fn mean(x: PyRef<'_, PyVariable>) -> PyResult<PyVariable> {
    Ok(PyVariable::wrap(F::mean(&x.inner).map_err(to_py)?))
}

#[pyfunction]
#[pyo3(signature = (logits, target))]
fn softmax_cross_entropy(
    logits: PyRef<'_, PyVariable>,
    target: PyRef<'_, PyVariable>,
) -> PyResult<PyVariable> {
    Ok(PyVariable::wrap(
        F::softmax_cross_entropy(&logits.inner, &target.inner).map_err(to_py)?,
    ))
}

#[pyclass(name = "SGD", module = "chainer.optimizers")]
pub struct PySGD {
    inner: SGD,
    params: Vec<Parameter<CpuBackend>>,
}

#[pymethods]
impl PySGD {
    #[new]
    fn new(lr: f64) -> Self {
        Self {
            inner: SGD::new(lr),
            params: Vec::new(),
        }
    }

    fn setup(&mut self, params: Vec<PyRef<'_, PyVariable>>) {
        self.params = params
            .into_iter()
            .enumerate()
            .map(|(i, variable)| Parameter::new(format!("param{i}"), variable.inner.clone()))
            .collect();
    }

    fn update(&mut self) -> PyResult<()> {
        self.inner.update(&self.params).map_err(to_py)
    }

    fn zero_grad(&self) {
        modern_chainer::zero_grad(&self.params);
    }
}

#[pyclass(name = "Adam", module = "chainer.optimizers")]
pub struct PyAdam {
    inner: Adam<CpuBackend>,
    params: Vec<Parameter<CpuBackend>>,
}

#[pymethods]
impl PyAdam {
    #[new]
    #[pyo3(signature = (lr=0.001))]
    fn new(lr: f64) -> Self {
        Self {
            inner: Adam::new(lr),
            params: Vec::new(),
        }
    }

    fn setup(&mut self, params: Vec<PyRef<'_, PyVariable>>) {
        self.params = params
            .into_iter()
            .enumerate()
            .map(|(i, variable)| Parameter::new(format!("param{i}"), variable.inner.clone()))
            .collect();
    }

    fn update(&mut self) -> PyResult<()> {
        self.inner.update(&self.params).map_err(to_py)
    }

    fn zero_grad(&self) {
        modern_chainer::zero_grad(&self.params);
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
    module.add_function(wrap_pyfunction!(version, module)?)?;
    Ok(())
}
