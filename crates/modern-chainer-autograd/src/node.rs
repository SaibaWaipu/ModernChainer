use std::sync::Arc;

use modern_chainer_core::{Backend, Result, Tensor};

use crate::variable::Variable;

/// A node in the backward graph.
pub trait Node<B: Backend>: Send + Sync {
    fn name(&self) -> &'static str;
    fn inputs(&self) -> &[Variable<B>];
    fn backward(&self, grad: &Tensor<B>) -> Result<Vec<Option<Tensor<B>>>>;
}

/// A differentiable operation.
pub trait Op<B: Backend>: Send + Sync {
    const NAME: &'static str;
    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>>;
    fn backward(
        &self,
        grad: &Tensor<B>,
        inputs: &[&Tensor<B>],
        outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>>;
}

pub(crate) struct OpNode<B: Backend, O: Op<B>> {
    op: O,
    inputs: Vec<Variable<B>>,
    outputs: Vec<Tensor<B>>,
}

impl<B: Backend, O: Op<B>> OpNode<B, O> {
    fn new(op: O, inputs: Vec<Variable<B>>, outputs: Vec<Tensor<B>>) -> Self {
        Self {
            op,
            inputs,
            outputs,
        }
    }
}

impl<B: Backend, O: Op<B>> Node<B> for OpNode<B, O> {
    fn name(&self) -> &'static str {
        O::NAME
    }

    fn inputs(&self) -> &[Variable<B>] {
        &self.inputs
    }

    fn backward(&self, grad: &Tensor<B>) -> Result<Vec<Option<Tensor<B>>>> {
        let input_data: Vec<Tensor<B>> = self.inputs.iter().map(|v| v.data()).collect();
        let input_refs: Vec<&Tensor<B>> = input_data.iter().collect();
        let output_refs: Vec<&Tensor<B>> = self.outputs.iter().collect();
        self.op.backward(grad, &input_refs, &output_refs)
    }
}

/// Run `op` on `inputs`, building a graph unless gradients are disabled or no
/// input requires them.
pub fn apply<B: Backend, O: Op<B> + 'static>(
    op: O,
    inputs: &[Variable<B>],
) -> Result<Vec<Variable<B>>> {
    let data: Vec<Tensor<B>> = inputs.iter().map(|v| v.data()).collect();
    let refs: Vec<&Tensor<B>> = data.iter().collect();
    let outputs = op.forward(&refs)?;

    let track = crate::is_grad_enabled() && inputs.iter().any(|v| v.requires_grad());
    if !track {
        return Ok(outputs.into_iter().map(Variable::new).collect());
    }

    let node: Arc<dyn Node<B>> = Arc::new(OpNode::new(op, inputs.to_vec(), outputs.clone()));
    Ok(outputs
        .into_iter()
        .map(|tensor| Variable::from_parts(tensor, true, Some(Arc::clone(&node))))
        .collect())
}

pub(crate) fn apply1<B: Backend, O: Op<B> + 'static>(
    op: O,
    input: &Variable<B>,
) -> Result<Variable<B>> {
    let mut out = apply(op, std::slice::from_ref(input))?;
    debug_assert_eq!(out.len(), 1);
    Ok(out.remove(0))
}
