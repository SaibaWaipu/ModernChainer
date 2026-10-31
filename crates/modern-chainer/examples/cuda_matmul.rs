use modern_chainer::cuda;
use modern_chainer::prelude::*;

fn main() -> Result<()> {
    if !cuda::is_available() {
        println!("CUDA is not available: {}", cuda::probe().reason);
        return Ok(());
    }
    let backend = CudaBackend::new(0)?;
    println!("adapter: {}", backend.name());

    let a = Tensor::from_f32(backend.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])?;
    let b = Tensor::from_f32(backend.clone(), &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], [3, 2])?;
    let c = backend.ops().matmul(&a, &b)?;
    let bytes = c.to_bytes()?;
    let values: &[f32] = bytemuck::cast_slice(&bytes);
    println!("matmul result: {values:?}");
    Ok(())
}
