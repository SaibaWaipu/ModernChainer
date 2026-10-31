use modern_chainer::vulkan::{VulkanBackend, probe};
use modern_chainer::{Backend, Ops};

fn main() -> modern_chainer::Result<()> {
    let capability = probe();
    if !capability.available {
        println!("Vulkan is not available: {}", capability.reason);
        return Ok(());
    }
    let backend = VulkanBackend::new()?;
    println!("adapter: {}", backend.adapter_name());

    let a =
        modern_chainer::Tensor::from_f32(backend.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])?;
    let b =
        modern_chainer::Tensor::from_f32(backend.clone(), &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], [3, 2])?;
    let c = backend.ops().matmul(&a, &b)?;
    let bytes = c.to_bytes()?;
    let values: &[f32] = bytemuck::cast_slice(&bytes);
    println!("matmul result: {values:?}");
    Ok(())
}
