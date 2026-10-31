use std::collections::HashMap;

use modern_chainer_autograd::Variable;
use modern_chainer_core::{Backend, DType, Device, Error, Result, Shape, Tensor};

use crate::key::{StateId, StateKey};
use crate::spec::{GradPolicy, StateLayout, StateSpec};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScalarValue {
    F32(f32),
    F64(f64),
    I64(i64),
    Bool(bool),
}

impl ScalarValue {
    pub fn default_for(dtype: DType) -> Result<Self> {
        Ok(match dtype {
            DType::F32 => ScalarValue::F32(0.0),
            DType::F64 => ScalarValue::F64(0.0),
            DType::I32 | DType::I64 => ScalarValue::I64(0),
            DType::Bool => ScalarValue::Bool(false),
            other => {
                return Err(Error::UnsupportedDType(other));
            }
        })
    }

    pub fn dtype(self) -> DType {
        match self {
            ScalarValue::F32(_) => DType::F32,
            ScalarValue::F64(_) => DType::F64,
            ScalarValue::I64(_) => DType::I64,
            ScalarValue::Bool(_) => DType::Bool,
        }
    }

    pub fn as_f64(self) -> Result<f64> {
        match self {
            ScalarValue::F32(v) => Ok(v as f64),
            ScalarValue::F64(v) => Ok(v),
            ScalarValue::I64(v) => Ok(v as f64),
            ScalarValue::Bool(v) => Ok(u8::from(v) as f64),
        }
    }

    pub fn as_i64(self) -> Result<i64> {
        match self {
            ScalarValue::F32(v) => Ok(v as i64),
            ScalarValue::F64(v) => Ok(v as i64),
            ScalarValue::I64(v) => Ok(v),
            ScalarValue::Bool(v) => Ok(i64::from(v)),
        }
    }

    pub fn as_bool(self) -> Result<bool> {
        Ok(self.as_i64()? != 0)
    }
}

#[derive(Clone)]
pub struct PagedState<B: Backend> {
    pub blocks: Tensor<B>,
    pub block_table: Tensor<B>,
    pub block_size: usize,
    pub len: usize,
}

#[derive(Clone)]
pub enum StateValue<B: Backend> {
    Tensor(Tensor<B>),
    Variable(Variable<B>),
    Scalar(ScalarValue),
    Paged(PagedState<B>),
}

impl<B: Backend> StateValue<B> {
    pub fn kind(&self) -> &'static str {
        match self {
            StateValue::Tensor(_) => "tensor",
            StateValue::Variable(_) => "variable",
            StateValue::Scalar(_) => "scalar",
            StateValue::Paged(_) => "paged",
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            StateValue::Tensor(t) => t.dtype(),
            StateValue::Variable(v) => v.data().dtype(),
            StateValue::Scalar(s) => s.dtype(),
            StateValue::Paged(p) => p.blocks.dtype(),
        }
    }

    pub fn as_tensor(&self) -> Option<Tensor<B>> {
        match self {
            StateValue::Tensor(t) => Some(t.clone()),
            StateValue::Variable(v) => Some(v.data()),
            _ => None,
        }
    }

    pub fn as_scalar(&self) -> Option<ScalarValue> {
        match self {
            StateValue::Scalar(s) => Some(*s),
            _ => None,
        }
    }
}

#[derive(Clone)]
pub struct StateSlot<B: Backend> {
    spec: StateSpec,
    value: StateValue<B>,
    generation: u32,
}

impl<B: Backend> StateSlot<B> {
    pub fn spec(&self) -> &StateSpec {
        &self.spec
    }

    pub fn value(&self) -> &StateValue<B> {
        &self.value
    }
}

/// An arena of explicitly declared run state.
#[derive(Clone)]
pub struct StateStore<B: Backend> {
    backend: B,
    slots: Vec<Option<StateSlot<B>>>,
    index: HashMap<StateKey, StateId>,
    version: u64,
}

impl<B: Backend> StateStore<B> {
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            slots: Vec::new(),
            index: HashMap::new(),
            version: 0,
        }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn device(&self) -> Device {
        self.backend.device()
    }

    pub fn declare(&mut self, key: impl Into<StateKey>, spec: StateSpec) -> Result<StateId> {
        let key = key.into();
        if let Some(id) = self.index.get(&key).copied() {
            let slot = self.resolve(id)?;
            if !slot.spec.is_compatible(&spec) {
                return Err(Error::invalid(format!(
                    "state {key} already declared with an incompatible spec"
                )));
            }
            return Ok(id);
        }

        let value = self.allocate(&spec)?;
        let index = self.slots.len() as u32;
        let id = StateId::new(index, 0);
        self.slots.push(Some(StateSlot {
            spec,
            value,
            generation: 0,
        }));
        self.index.insert(key, id);
        Ok(id)
    }

    fn allocate(&self, spec: &StateSpec) -> Result<StateValue<B>> {
        match &spec.layout {
            StateLayout::Dense(shape) => {
                let tensor = Tensor::zeros(self.backend.clone(), shape.clone(), spec.dtype)?;
                Ok(match spec.grad {
                    GradPolicy::Detach => StateValue::Tensor(tensor),
                    GradPolicy::Differentiable => StateValue::Variable(Variable::new(tensor)),
                })
            }
            StateLayout::Scalar => Ok(StateValue::Scalar(ScalarValue::default_for(spec.dtype)?)),
            StateLayout::Paged { .. } => Err(Error::unsupported(
                "paged state allocation is reserved for the PagedAttention milestone",
            )),
        }
    }

    pub fn id(&self, key: impl AsRef<str>) -> Option<StateId> {
        self.index.get(&StateKey::new(key.as_ref())).copied()
    }

    pub fn resolve(&self, id: StateId) -> Result<&StateSlot<B>> {
        let slot = self
            .slots
            .get(id.index as usize)
            .and_then(Option::as_ref)
            .ok_or_else(|| Error::invalid(format!("unknown state handle {id}")))?;
        if slot.generation != id.generation {
            return Err(Error::invalid(format!(
                "stale state handle {id} (generation {})",
                slot.generation
            )));
        }
        Ok(slot)
    }

    fn resolve_mut(&mut self, id: StateId) -> Result<&mut StateSlot<B>> {
        let slot = self
            .slots
            .get_mut(id.index as usize)
            .and_then(Option::as_mut)
            .ok_or_else(|| Error::invalid(format!("unknown state handle {id}")))?;
        if slot.generation != id.generation {
            return Err(Error::invalid(format!(
                "stale state handle {id} (generation {})",
                slot.generation
            )));
        }
        Ok(slot)
    }

    pub fn spec(&self, id: StateId) -> Result<&StateSpec> {
        Ok(self.resolve(id)?.spec())
    }

    pub fn value(&self, id: StateId) -> Result<&StateValue<B>> {
        Ok(self.resolve(id)?.value())
    }

    pub fn tensor(&self, id: StateId) -> Result<Tensor<B>> {
        let value = self.value(id)?;
        value
            .as_tensor()
            .ok_or_else(|| Error::invalid(format!("state {id} is not a tensor")))
    }

    pub fn variable(&self, id: StateId) -> Result<Variable<B>> {
        match self.value(id)? {
            StateValue::Variable(v) => Ok(v.clone()),
            StateValue::Tensor(t) => Ok(Variable::constant(t.clone())),
            _ => Err(Error::invalid(format!("state {id} is not a variable"))),
        }
    }

    pub fn scalar(&self, id: StateId) -> Result<ScalarValue> {
        self.value(id)?
            .as_scalar()
            .ok_or_else(|| Error::invalid(format!("state {id} is not a scalar")))
    }

    pub fn set_tensor(&mut self, id: StateId, tensor: Tensor<B>) -> Result<()> {
        let dtype = self.spec(id)?.dtype;
        if tensor.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                expected: dtype,
                actual: tensor.dtype(),
            });
        }
        match &mut self.resolve_mut(id)?.value {
            StateValue::Tensor(current) => {
                *current = tensor;
                Ok(())
            }
            StateValue::Variable(variable) => {
                variable.set_data(tensor);
                Ok(())
            }
            _ => Err(Error::invalid(format!("state {id} does not hold a tensor"))),
        }
    }

    pub fn set_variable(&mut self, id: StateId, variable: Variable<B>) -> Result<()> {
        let dtype = self.spec(id)?.dtype;
        if variable.data().dtype() != dtype {
            return Err(Error::DTypeMismatch {
                expected: dtype,
                actual: variable.data().dtype(),
            });
        }
        match &mut self.resolve_mut(id)?.value {
            StateValue::Variable(current) => {
                *current = variable;
                Ok(())
            }
            StateValue::Tensor(current) => {
                *current = variable.data();
                Ok(())
            }
            _ => Err(Error::invalid(format!(
                "state {id} does not hold a variable"
            ))),
        }
    }

    pub fn set_scalar(&mut self, id: StateId, scalar: ScalarValue) -> Result<()> {
        let dtype = self.spec(id)?.dtype;
        let compatible = matches!(
            (dtype, scalar),
            (DType::F32, ScalarValue::F32(_))
                | (DType::F64, ScalarValue::F64(_))
                | (DType::I32, ScalarValue::I64(_))
                | (DType::I64, ScalarValue::I64(_))
                | (DType::Bool, ScalarValue::Bool(_))
        );
        if !compatible {
            return Err(Error::DTypeMismatch {
                expected: dtype,
                actual: scalar.dtype(),
            });
        }
        match &mut self.resolve_mut(id)?.value {
            StateValue::Scalar(current) => {
                *current = scalar;
                Ok(())
            }
            _ => Err(Error::invalid(format!("state {id} is not a scalar"))),
        }
    }

    pub fn set_value(&mut self, id: StateId, value: StateValue<B>) -> Result<()> {
        match value {
            StateValue::Tensor(t) => self.set_tensor(id, t),
            StateValue::Variable(v) => self.set_variable(id, v),
            StateValue::Scalar(s) => self.set_scalar(id, s),
            StateValue::Paged(p) => {
                let spec = self.spec(id)?.clone();
                match &spec.layout {
                    StateLayout::Paged { .. } => {
                        if p.blocks.dtype() != spec.dtype {
                            return Err(Error::DTypeMismatch {
                                expected: spec.dtype,
                                actual: p.blocks.dtype(),
                            });
                        }
                        match &mut self.resolve_mut(id)?.value {
                            StateValue::Paged(current) => {
                                *current = p;
                                Ok(())
                            }
                            _ => Err(Error::invalid(format!("state {id} is not paged"))),
                        }
                    }
                    _ => Err(Error::invalid(format!("state {id} is not paged"))),
                }
            }
        }
    }

    pub fn copy_slot_from(&mut self, other: &Self, id: StateId) -> Result<()> {
        let value = other.value(id)?.clone();
        self.set_value(id, value)
    }

    pub fn keys(&self) -> impl Iterator<Item = &StateKey> {
        self.index.keys()
    }

    pub fn plan(&self) -> StatePlan {
        let mut entries: Vec<(StateKey, StateId)> = self
            .index
            .iter()
            .map(|(key, id)| (key.clone(), *id))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        StatePlan { entries }
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn bump_version(&mut self) {
        self.version += 1;
    }

    pub fn memory_usage(&self) -> usize {
        self.slots
            .iter()
            .flatten()
            .map(|slot| match &slot.value {
                StateValue::Tensor(t) => t.numel() * t.dtype().size_in_bytes(),
                StateValue::Variable(v) => v.data().numel() * v.data().dtype().size_in_bytes(),
                StateValue::Scalar(s) => s.dtype().size_in_bytes().max(1),
                StateValue::Paged(p) => {
                    p.blocks.numel() * p.blocks.dtype().size_in_bytes()
                        + p.block_table.numel() * p.block_table.dtype().size_in_bytes()
                }
            })
            .sum()
    }
}

#[derive(Clone, Debug)]
pub struct StatePlan {
    entries: Vec<(StateKey, StateId)>,
}

impl StatePlan {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&StateKey, StateId)> {
        self.entries.iter().map(|(key, id)| (key, *id))
    }

    pub fn get(&self, key: impl AsRef<str>) -> Option<StateId> {
        let key = key.as_ref();
        self.entries
            .iter()
            .find(|(candidate, _)| candidate.as_str() == key)
            .map(|(_, id)| *id)
    }
}

pub struct StateBuilder<'a, B: Backend> {
    store: &'a mut StateStore<B>,
}

impl<'a, B: Backend> StateBuilder<'a, B> {
    pub fn new(store: &'a mut StateStore<B>) -> Self {
        Self { store }
    }

    pub fn declare(&mut self, key: impl Into<StateKey>, spec: StateSpec) -> Result<StateId> {
        self.store.declare(key, spec)
    }

    pub fn declare_dense(
        &mut self,
        key: impl Into<StateKey>,
        shape: impl Into<Shape>,
        dtype: DType,
    ) -> Result<StateId> {
        self.declare(key, StateSpec::dense(shape, dtype))
    }

    pub fn declare_scalar(&mut self, key: impl Into<StateKey>, dtype: DType) -> Result<StateId> {
        self.declare(key, StateSpec::scalar(dtype))
    }

    pub fn backend(&self) -> &B {
        self.store.backend()
    }

    pub fn device(&self) -> Device {
        self.store.device()
    }
}
