use modern_chainer_core::{DType, Tensor};
use modern_chainer_cpu::{CpuBackend, CpuTensorExt};
use modern_chainer_serial::{load, save};

#[test]
fn roundtrip_preserves_names_values_and_shapes() {
    let backend = CpuBackend::new();
    let w = Tensor::from_f32(backend, &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [2, 3]).unwrap();
    let b = Tensor::from_f32(backend, &[1.0, 2.0], [2]).unwrap();
    let entries = vec![("W".to_string(), w), ("b".to_string(), b)];
    let path = std::env::temp_dir().join(format!(
        "modern-chainer-test-{}-{}.mcn",
        std::process::id(),
        std::thread::current().name().unwrap_or("main")
    ));
    save(&path, &entries).unwrap();
    let loaded = load(&path, CpuBackend::new()).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded[0].0, "W");
    assert_eq!(loaded[0].1.shape().dims(), &[2, 3]);
    assert_eq!(loaded[0].1.dtype(), DType::F32);
    let values = loaded[0].1.as_slice::<f32>().unwrap();
    assert_eq!(values, &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
}
