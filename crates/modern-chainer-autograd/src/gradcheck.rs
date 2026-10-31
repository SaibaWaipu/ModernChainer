use modern_chainer_core::{Backend, DType, Error, Ops, Result, Tensor};

use crate::variable::Variable;

/// Central-difference gradient check.
///
/// `f` must return a scalar variable.  Only inputs that require gradients and
/// hold `f32` data are checked, so integer label tensors can be passed through
/// unchanged.
pub fn gradcheck<B, F>(f: F, inputs: &[Variable<B>], eps: f32, rtol: f64, atol: f64) -> Result<()>
where
    B: Backend,
    F: Fn(&[Variable<B>]) -> Result<Variable<B>>,
{
    let checkable: Vec<usize> = inputs
        .iter()
        .enumerate()
        .filter(|(_, input)| input.requires_grad())
        .map(|(index, _)| index)
        .collect();

    for &index in &checkable {
        let input = &inputs[index];
        if input.data().dtype() != DType::F32 {
            return Err(Error::UnsupportedDType(input.data().dtype()));
        }
        input.cleargrad();
    }

    let loss = f(inputs)?;
    loss.backward()?;

    let mut analytic: Vec<(usize, Vec<f32>)> = Vec::with_capacity(checkable.len());
    for &index in &checkable {
        let input = &inputs[index];
        let values = match input.grad() {
            Some(grad) => {
                let bytes = grad.to_bytes()?;
                bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
            }
            None => vec![0.0; input.data().numel()],
        };
        analytic.push((index, values));
    }

    for (input_index, expected_values) in analytic {
        let input = &inputs[input_index];
        let tensor = input.data();
        let base_bytes = tensor.to_bytes()?;
        let dtype = tensor.dtype();
        let shape = tensor.shape().clone();
        let backend = tensor.backend().clone();

        for (element, &expected) in expected_values.iter().enumerate() {
            let mut plus = base_bytes.clone();
            let mut minus = base_bytes.clone();
            let offset = element * dtype.size_in_bytes();
            let original = f32::from_ne_bytes(
                base_bytes[offset..offset + 4]
                    .try_into()
                    .expect("f32 element"),
            );
            plus[offset..offset + 4].copy_from_slice(&(original + eps).to_ne_bytes());
            minus[offset..offset + 4].copy_from_slice(&(original - eps).to_ne_bytes());

            let eval = |bytes: &[u8]| -> Result<f64> {
                let replaced = Variable::new(Tensor::from_bytes(
                    backend.clone(),
                    bytes,
                    shape.clone(),
                    dtype,
                )?);
                let mut args = Vec::with_capacity(inputs.len());
                for (i, other) in inputs.iter().enumerate() {
                    if i == input_index {
                        args.push(replaced.clone());
                    } else {
                        args.push(other.clone());
                    }
                }
                let value = crate::no_grad(|| f(&args))?;
                let data = value.data();
                data.backend().ops().to_scalar(&data)
            };

            let numeric = (eval(&plus)? - eval(&minus)?) / (2.0 * eps as f64);
            let expected = expected as f64;
            let scale = expected.abs().max(numeric.abs()).max(1.0);
            if (expected - numeric).abs() > atol + rtol * scale {
                return Err(Error::invalid(format!(
                    "gradcheck failed at input {input_index} element {element}: \
                     analytic={expected}, numeric={numeric}, eps={eps}"
                )));
            }
        }
    }
    Ok(())
}
