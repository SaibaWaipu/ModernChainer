//! Minimal image primitives used by the vision examples.

use modern_chainer_core::{Error, Result};

#[derive(Clone, Debug)]
pub struct GrayImage {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<f32>,
}

impl GrayImage {
    pub fn new(width: usize, height: usize, pixels: Vec<f32>) -> Result<Self> {
        if width == 0 || height == 0 || pixels.len() != width * height {
            return Err(Error::invalid("invalid grayscale image dimensions"));
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    pub fn zeros(width: usize, height: usize) -> Result<Self> {
        Self::new(width, height, vec![0.0; width * height])
    }

    pub fn flatten(&self) -> &[f32] {
        &self.pixels
    }

    pub fn pixel(&self, x: usize, y: usize) -> f32 {
        self.pixels[y * self.width + x]
    }

    pub fn set_pixel(&mut self, x: usize, y: usize, value: f32) {
        self.pixels[y * self.width + x] = value;
    }

    pub fn resize_nearest(&self, width: usize, height: usize) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::invalid("resize dimensions must be positive"));
        }
        let mut out = vec![0.0; width * height];
        for y in 0..height {
            let src_y = y * self.height / height;
            for x in 0..width {
                let src_x = x * self.width / width;
                out[y * width + x] = self.pixel(src_x, src_y);
            }
        }
        Self::new(width, height, out)
    }

    pub fn read_ppm(bytes: &[u8]) -> Result<Self> {
        let mut fields = Vec::new();
        let mut i = 0;
        while fields.len() < 4 && i < bytes.len() {
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i >= bytes.len() {
                break;
            }
            let start = i;
            while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            fields.push(
                std::str::from_utf8(&bytes[start..i])
                    .map_err(|_| Error::invalid("invalid PPM header"))?,
            );
        }
        if fields.len() != 4 || fields[0] != "P6" {
            return Err(Error::invalid("only binary P6 PPM is supported"));
        }
        let width: usize = fields[1]
            .parse()
            .map_err(|_| Error::invalid("invalid PPM width"))?;
        let height: usize = fields[2]
            .parse()
            .map_err(|_| Error::invalid("invalid PPM height"))?;
        let max: f32 = fields[3]
            .parse()
            .map_err(|_| Error::invalid("invalid PPM max value"))?;
        i += 1;
        let expected = width * height * 3;
        if bytes.len() < i + expected {
            return Err(Error::invalid("truncated PPM payload"));
        }
        let mut pixels = vec![0.0; width * height];
        let (chunks, _) = bytes[i..i + expected].as_chunks::<3>();
        for (p, rgb) in pixels.iter_mut().zip(chunks) {
            *p = (rgb[0] as f32 + rgb[1] as f32 + rgb[2] as f32) / (3.0 * max);
        }
        Self::new(width, height, pixels)
    }

    pub fn write_ppm(&self) -> Vec<u8> {
        let mut out = format!("P6\n{} {}\n255\n", self.width, self.height).into_bytes();
        for &value in &self.pixels {
            let v = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
            out.extend_from_slice(&[v, v, v]);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ppm_roundtrip() {
        let image = GrayImage::new(2, 2, vec![0.0, 0.5, 1.0, 0.25]).unwrap();
        let parsed = GrayImage::read_ppm(&image.write_ppm()).unwrap();
        assert_eq!(parsed.width, 2);
        assert_eq!(parsed.height, 2);
        for (a, b) in image.pixels.iter().zip(&parsed.pixels) {
            assert!((a - b).abs() < 0.01);
        }
    }
}

/// Splits a grayscale image into flattened square patches.
#[derive(Clone, Debug)]
pub struct PatchTokenizer {
    pub patch: usize,
}

impl PatchTokenizer {
    pub fn new(patch: usize) -> Result<Self> {
        if patch == 0 {
            return Err(Error::invalid("patch size must be positive"));
        }
        Ok(Self { patch })
    }

    pub fn tokenize(&self, image: &GrayImage) -> Result<Vec<Vec<f32>>> {
        if !image.width.is_multiple_of(self.patch) || !image.height.is_multiple_of(self.patch) {
            return Err(Error::invalid("image size must be divisible by patch size"));
        }
        let mut tokens = Vec::new();
        for y in (0..image.height).step_by(self.patch) {
            for x in (0..image.width).step_by(self.patch) {
                let mut patch = Vec::with_capacity(self.patch * self.patch);
                for py in 0..self.patch {
                    for px in 0..self.patch {
                        patch.push(image.pixel(x + px, y + py));
                    }
                }
                tokens.push(patch);
            }
        }
        Ok(tokens)
    }
}

/// Loads `.ppm` images from one subdirectory per class.
pub fn load_ppm_dir(root: impl AsRef<std::path::Path>) -> Result<Vec<(GrayImage, usize)>> {
    let root = root.as_ref();
    let mut classes: Vec<_> = std::fs::read_dir(root)
        .map_err(|e| Error::invalid(e.to_string()))?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().is_dir())
        .collect();
    classes.sort_by_key(|entry| entry.file_name());
    let mut out = Vec::new();
    for (label, class) in classes.into_iter().enumerate() {
        let mut files: Vec<_> = std::fs::read_dir(class.path())
            .map_err(|e| Error::invalid(e.to_string()))?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("ppm"))
            })
            .collect();
        files.sort_by_key(|entry| entry.file_name());
        for file in files {
            let bytes = std::fs::read(file.path()).map_err(|e| Error::invalid(e.to_string()))?;
            out.push((GrayImage::read_ppm(&bytes)?, label));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod extra_tests {
    use super::*;

    #[test]
    fn patch_tokenizer() {
        let image = GrayImage::new(4, 4, (0..16).map(|v| v as f32).collect()).unwrap();
        let tokens = PatchTokenizer::new(2).unwrap().tokenize(&image).unwrap();
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0], vec![0.0, 1.0, 4.0, 5.0]);
    }
}
