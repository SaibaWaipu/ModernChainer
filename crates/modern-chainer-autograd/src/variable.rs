use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use modern_chainer_core::{Backend, Ops, Result, Tensor};

use crate::node::Node;

struct VarInner<B: Backend> {
    data: RwLock<Tensor<B>>,
    grad: Mutex<Option<Tensor<B>>>,
    node: RwLock<Option<Arc<dyn Node<B>>>>,
    requires_grad: AtomicBool,
    name: RwLock<Option<String>>,
}

/// A define-by-run variable.
///
/// Cloning a `Variable` shares the same underlying data and gradient, which is
/// what makes Chainer's graph handles cheap to pass around.
pub struct Variable<B: Backend> {
    inner: Arc<VarInner<B>>,
}

impl<B: Backend> Clone for Variable<B> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<B: Backend> Variable<B> {
    pub fn new(tensor: Tensor<B>) -> Self {
        Self::from_parts(tensor, true, None)
    }

    pub fn constant(tensor: Tensor<B>) -> Self {
        Self::from_parts(tensor, false, None)
    }

    pub fn from_parts(
        tensor: Tensor<B>,
        requires_grad: bool,
        node: Option<Arc<dyn Node<B>>>,
    ) -> Self {
        Self {
            inner: Arc::new(VarInner {
                data: RwLock::new(tensor),
                grad: Mutex::new(None),
                node: RwLock::new(node),
                requires_grad: AtomicBool::new(requires_grad),
                name: RwLock::new(None),
            }),
        }
    }

    pub fn id(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }

    pub fn data(&self) -> Tensor<B> {
        self.inner
            .data
            .read()
            .expect("variable data lock poisoned")
            .clone()
    }

    pub fn set_data(&self, tensor: Tensor<B>) {
        *self
            .inner
            .data
            .write()
            .expect("variable data lock poisoned") = tensor;
    }

    pub fn grad(&self) -> Option<Tensor<B>> {
        self.inner
            .grad
            .lock()
            .expect("variable grad lock poisoned")
            .clone()
    }

    pub fn cleargrad(&self) {
        *self.inner.grad.lock().expect("variable grad lock poisoned") = None;
    }

    pub fn set_grad(&self, grad: Option<Tensor<B>>) {
        *self.inner.grad.lock().expect("variable grad lock poisoned") = grad;
    }

    pub fn requires_grad(&self) -> bool {
        self.inner.requires_grad.load(Ordering::Relaxed)
    }

    pub fn set_requires_grad(&self, value: bool) {
        self.inner.requires_grad.store(value, Ordering::Relaxed);
    }

    pub fn name(&self) -> Option<String> {
        self.inner
            .name
            .read()
            .expect("variable name lock poisoned")
            .clone()
    }

    pub fn set_name(&self, name: impl Into<String>) {
        *self
            .inner
            .name
            .write()
            .expect("variable name lock poisoned") = Some(name.into());
    }

    pub fn is_leaf(&self) -> bool {
        self.node().is_none()
    }

    pub(crate) fn node(&self) -> Option<Arc<dyn Node<B>>> {
        self.inner
            .node
            .read()
            .expect("variable node lock poisoned")
            .clone()
    }

    pub fn grad_fn_name(&self) -> Option<&'static str> {
        self.node().map(|node| node.name())
    }

    /// Detach from the graph, keeping the same data.
    pub fn detach(&self) -> Self {
        Self::from_parts(self.data(), false, None)
    }

    /// Drop the graph behind this variable.
    pub fn unchain_backward(&self) {
        *self
            .inner
            .node
            .write()
            .expect("variable node lock poisoned") = None;
    }

    pub fn backward(&self) -> Result<()> {
        if !self.requires_grad() || !crate::is_grad_enabled() {
            return Ok(());
        }
        let seed = self.data();
        let backend = seed.backend().clone();
        let ops = backend.ops();

        let mut order: Vec<Variable<B>> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![(self.clone(), false)];
        while let Some((variable, expanded)) = stack.pop() {
            if expanded {
                order.push(variable);
                continue;
            }
            if !seen.insert(variable.id()) {
                continue;
            }
            stack.push((variable.clone(), true));
            if let Some(node) = variable.node() {
                for input in node.inputs() {
                    stack.push((input.clone(), false));
                }
            }
        }

        let mut grads: HashMap<usize, Tensor<B>> = HashMap::new();
        grads.insert(self.id(), ops.ones_like(&seed)?);

        for variable in order.into_iter().rev() {
            let Some(grad) = grads.remove(&variable.id()) else {
                continue;
            };
            variable.accumulate_grad(&grad, ops)?;
            if let Some(node) = variable.node() {
                let input_grads = node.backward(&grad)?;
                for (input, input_grad) in node.inputs().iter().zip(input_grads) {
                    let Some(input_grad) = input_grad else {
                        continue;
                    };
                    if !input.requires_grad() {
                        continue;
                    }
                    match grads.entry(input.id()) {
                        Entry::Occupied(mut entry) => {
                            let sum = ops.add(entry.get(), &input_grad)?;
                            entry.insert(sum);
                        }
                        Entry::Vacant(entry) => {
                            entry.insert(input_grad);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn accumulate_grad(&self, grad: &Tensor<B>, ops: &B::Ops) -> Result<()> {
        let mut slot = self.inner.grad.lock().expect("variable grad lock poisoned");
        let next = match slot.as_ref() {
            Some(current) => ops.add(current, grad)?,
            None => grad.clone(),
        };
        *slot = Some(next);
        Ok(())
    }
}

impl<B: Backend> PartialEq for Variable<B> {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl<B: Backend> Eq for Variable<B> {}

impl<B: Backend> fmt::Debug for Variable<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Variable")
            .field("shape", self.data().shape())
            .field("requires_grad", &self.requires_grad())
            .field("grad_fn", &self.grad_fn_name())
            .finish()
    }
}
