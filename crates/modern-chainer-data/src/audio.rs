//! Small WAV reader/writer and hand-crafted audio features for the examples.

use std::fs;
use std::path::Path;

use modern_chainer_core::{Error, Result};

#[derive(Clone, Debug)]
pub struct Wav {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
}

impl Wav {
    pub fn new(sample_rate: u32, samples: Vec<f32>) -> Self {
        Self {
            sample_rate,
            samples,
        }
    }

    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_bytes(&fs::read(path).map_err(|e| Error::invalid(e.to_string()))?)
    }

    pub fn write(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, self.to_bytes()).map_err(|e| Error::invalid(e.to_string()))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
            return Err(Error::invalid("not a RIFF/WAVE file"));
        }
        let mut pos = 12;
        let mut sample_rate = 0u32;
        let mut channels = 1usize;
        let mut bits = 16usize;
        let mut data: Option<&[u8]> = None;
        while pos + 8 <= bytes.len() {
            let id = &bytes[pos..pos + 4];
            let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
            let start = pos + 8;
            let end = (start + len).min(bytes.len());
            match id {
                b"fmt " => {
                    if end - start < 16 {
                        return Err(Error::invalid("short WAV fmt chunk"));
                    }
                    let format = u16::from_le_bytes(bytes[start..start + 2].try_into().unwrap());
                    if format != 1 && format != 3 {
                        return Err(Error::invalid("only PCM/float WAV is supported"));
                    }
                    channels = u16::from_le_bytes(bytes[start + 2..start + 4].try_into().unwrap())
                        as usize;
                    sample_rate =
                        u32::from_le_bytes(bytes[start + 4..start + 8].try_into().unwrap());
                    bits = u16::from_le_bytes(bytes[start + 14..start + 16].try_into().unwrap())
                        as usize;
                }
                b"data" => data = Some(&bytes[start..end]),
                _ => {}
            }
            pos = end + (len & 1);
        }
        let data = data.ok_or_else(|| Error::invalid("WAV has no data chunk"))?;
        let channels = channels.max(1);
        let mut samples = Vec::new();
        match bits {
            16 => {
                for frame in data.chunks_exact(2 * channels) {
                    let mut mono = 0f32;
                    for c in 0..channels {
                        let v =
                            i16::from_le_bytes([frame[c * 2], frame[c * 2 + 1]]) as f32 / 32768.0;
                        mono += v;
                    }
                    samples.push(mono / channels as f32);
                }
            }
            32 => {
                for frame in data.chunks_exact(4 * channels) {
                    let mut mono = 0f32;
                    for c in 0..channels {
                        let v = f32::from_le_bytes(frame[c * 4..c * 4 + 4].try_into().unwrap());
                        mono += v;
                    }
                    samples.push(mono / channels as f32);
                }
            }
            _ => return Err(Error::invalid("only 16/32-bit WAV is supported")),
        }
        Ok(Self::new(sample_rate, samples))
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let data_len = self.samples.len() * 2;
        let mut out = Vec::with_capacity(44 + data_len);
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data_len as u32).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&self.sample_rate.to_le_bytes());
        out.extend_from_slice(&(self.sample_rate * 2).to_le_bytes());
        out.extend_from_slice(&2u16.to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data_len as u32).to_le_bytes());
        for &sample in &self.samples {
            let v = (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16;
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }
}

pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|v| v * v).sum::<f32>() / samples.len() as f32).sqrt()
}

pub fn zero_crossing_rate(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let crossings = samples
        .windows(2)
        .filter(|pair| (pair[0] >= 0.0) != (pair[1] >= 0.0))
        .count();
    crossings as f32 / (samples.len() - 1) as f32
}

/// Goertzel power at one frequency, useful for small tone/keyword tasks.
pub fn goertzel_power(samples: &[f32], sample_rate: u32, frequency: f32) -> f32 {
    if samples.is_empty() || sample_rate == 0 {
        return 0.0;
    }
    let omega = 2.0 * std::f32::consts::PI * frequency / sample_rate as f32;
    let coeff = 2.0 * omega.cos();
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    for &x in samples {
        let s0 = x + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    (s1 * s1 + s2 * s2 - coeff * s1 * s2).max(0.0) / samples.len() as f32
}

pub fn tone_features(samples: &[f32], sample_rate: u32, frequencies: &[f32]) -> Vec<f32> {
    let mut features = vec![rms(samples), zero_crossing_rate(samples)];
    for &frequency in frequencies {
        features.push(goertzel_power(samples, sample_rate, frequency));
    }
    features
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_roundtrip() {
        let wav = Wav::new(8_000, vec![0.0, 0.5, -0.5, 0.25]);
        let parsed = Wav::from_bytes(&wav.to_bytes()).unwrap();
        assert_eq!(parsed.sample_rate, 8_000);
        assert_eq!(parsed.samples.len(), 4);
        assert!((parsed.samples[1] - 0.5).abs() < 0.001);
    }

    #[test]
    fn detects_tone() {
        let rate = 8_000;
        let samples: Vec<f32> = (0..800)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / rate as f32).sin())
            .collect();
        let at = goertzel_power(&samples, rate, 440.0);
        let off = goertzel_power(&samples, rate, 880.0);
        assert!(at > off * 10.0);
    }
}
