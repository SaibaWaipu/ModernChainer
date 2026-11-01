//! CIFAR-10 binary batch loader.

use std::fs;
use std::path::Path;

use modern_chainer_core::{Error, Result};

pub const CIFAR_IMAGE_SIZE: usize = 32 * 32 * 3;

pub struct Cifar10 {
    data: Vec<u8>,
    labels: Vec<u8>,
}

impl Cifar10 {
    pub fn load_batches(dir: impl AsRef<Path>, files: &[&str]) -> Result<Self> {
        let dir = dir.as_ref();
        let mut data = Vec::new();
        let mut labels = Vec::new();
        for file in files {
            let bytes = fs::read(dir.join(file)).map_err(|e| Error::invalid(e.to_string()))?;
            if bytes.len() % (CIFAR_IMAGE_SIZE + 1) != 0 {
                return Err(Error::invalid(format!("invalid CIFAR batch {file}")));
            }
            let (records, _) = bytes.as_chunks::<3073>();
            for record in records {
                labels.push(record[0]);
                data.extend_from_slice(&record[1..]);
            }
        }
        Ok(Self { data, labels })
    }

    pub fn load_train(dir: impl AsRef<Path>) -> Result<Self> {
        Self::load_batches(
            dir,
            &[
                "data_batch_1.bin",
                "data_batch_2.bin",
                "data_batch_3.bin",
                "data_batch_4.bin",
                "data_batch_5.bin",
            ],
        )
    }

    pub fn load_test(dir: impl AsRef<Path>) -> Result<Self> {
        Self::load_batches(dir, &["test_batch.bin"])
    }

    /// Loads the fast.ai CIFAR-10 PNG mirror (`root/train/<class>/*.png`).
    pub fn load_image_dir(root: impl AsRef<Path>, split: &str) -> Result<Self> {
        let root = root.as_ref().join(split);
        let mut classes: Vec<_> = fs::read_dir(&root)
            .map_err(|e| Error::invalid(e.to_string()))?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().is_dir())
            .collect();
        classes.sort_by_key(|entry| entry.file_name());
        let mut data = Vec::new();
        let mut labels = Vec::new();
        for (label, class) in classes.into_iter().enumerate() {
            let mut files: Vec<_> = fs::read_dir(class.path())
                .map_err(|e| Error::invalid(e.to_string()))?
                .filter_map(|entry| entry.ok())
                .filter(|entry| {
                    entry.path().extension().is_some_and(|ext| {
                        ext.eq_ignore_ascii_case("png") || ext.eq_ignore_ascii_case("jpg")
                    })
                })
                .collect();
            files.sort_by_key(|entry| entry.file_name());
            for file in files {
                let image = image::open(file.path())
                    .map_err(|e| Error::invalid(e.to_string()))?
                    .to_rgb8();
                if image.width() != 32 || image.height() != 32 {
                    return Err(Error::invalid("CIFAR image must be 32x32"));
                }
                for channel in 0..3 {
                    for pixel in image.pixels() {
                        data.push(pixel[channel]);
                    }
                }
                labels.push(label as u8);
            }
        }
        if labels.is_empty() {
            return Err(Error::invalid("no CIFAR images found"));
        }
        Ok(Self { data, labels })
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    pub fn sample(&self, index: usize) -> Vec<f32> {
        self.data[index * CIFAR_IMAGE_SIZE..(index + 1) * CIFAR_IMAGE_SIZE]
            .iter()
            .map(|&v| v as f32 / 255.0)
            .collect()
    }

    pub fn label(&self, index: usize) -> u8 {
        self.labels[index]
    }
}
