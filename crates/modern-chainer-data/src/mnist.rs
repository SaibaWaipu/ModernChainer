//! IDX MNIST loader.

use std::fs;
use std::path::Path;

use modern_chainer_core::{Error, Result};

pub struct Mnist {
    images: Vec<u8>,
    labels: Vec<u8>,
    pub rows: usize,
    pub cols: usize,
}

impl Mnist {
    pub fn load(images_path: impl AsRef<Path>, labels_path: impl AsRef<Path>) -> Result<Self> {
        let images = fs::read(images_path).map_err(|e| Error::invalid(e.to_string()))?;
        let labels = fs::read(labels_path).map_err(|e| Error::invalid(e.to_string()))?;
        if images.len() < 16 || u32::from_be_bytes(images[0..4].try_into().unwrap()) != 2051 {
            return Err(Error::invalid("invalid MNIST image file"));
        }
        if labels.len() < 8 || u32::from_be_bytes(labels[0..4].try_into().unwrap()) != 2049 {
            return Err(Error::invalid("invalid MNIST label file"));
        }
        let count = u32::from_be_bytes(images[4..8].try_into().unwrap()) as usize;
        let rows = u32::from_be_bytes(images[8..12].try_into().unwrap()) as usize;
        let cols = u32::from_be_bytes(images[12..16].try_into().unwrap()) as usize;
        if labels.len() < 8 + count || images.len() < 16 + count * rows * cols {
            return Err(Error::invalid("truncated MNIST file"));
        }
        Ok(Self {
            images: images[16..16 + count * rows * cols].to_vec(),
            labels: labels[8..8 + count].to_vec(),
            rows,
            cols,
        })
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    pub fn image(&self, index: usize) -> &[u8] {
        let n = self.rows * self.cols;
        &self.images[index * n..(index + 1) * n]
    }

    pub fn sample(&self, index: usize) -> Vec<f32> {
        self.image(index)
            .iter()
            .map(|&v| v as f32 / 255.0)
            .collect()
    }

    pub fn label(&self, index: usize) -> u8 {
        self.labels[index]
    }
}
