use std::fmt;
use std::str::FromStr;

use crate::error::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Cuda(u32),
    Vulkan(u32),
}

impl Device {
    pub const fn cpu() -> Self {
        Device::Cpu
    }

    pub const fn cuda(index: u32) -> Self {
        Device::Cuda(index)
    }

    pub const fn vulkan(index: u32) -> Self {
        Device::Vulkan(index)
    }

    pub const fn index(self) -> Option<u32> {
        match self {
            Device::Cpu => None,
            Device::Cuda(i) | Device::Vulkan(i) => Some(i),
        }
    }

    pub const fn kind(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Cuda(_) => "cuda",
            Device::Vulkan(_) => "vulkan",
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => f.write_str("cpu"),
            Device::Cuda(i) => write!(f, "cuda:{i}"),
            Device::Vulkan(i) => write!(f, "vulkan:{i}"),
        }
    }
}

impl FromStr for Device {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("cpu") {
            return Ok(Device::Cpu);
        }
        let (kind, index) = match s.split_once(':') {
            Some((kind, index)) => (kind, Some(index)),
            None => (s, None),
        };
        let index = match index {
            Some(raw) => raw
                .parse::<u32>()
                .map_err(|_| Error::InvalidArgument(format!("invalid device index in {s:?}")))?,
            None => 0,
        };
        match kind.to_ascii_lowercase().as_str() {
            "cuda" | "gpu" => Ok(Device::Cuda(index)),
            "vulkan" | "vk" => Ok(Device::Vulkan(index)),
            "cpu" => Ok(Device::Cpu),
            other => Err(Error::InvalidArgument(format!(
                "unknown device kind {other:?} (expected cpu, cuda or vulkan)"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_devices() {
        assert_eq!("cpu".parse::<Device>().unwrap(), Device::Cpu);
        assert_eq!("cuda".parse::<Device>().unwrap(), Device::Cuda(0));
        assert_eq!("CUDA:1".parse::<Device>().unwrap(), Device::Cuda(1));
        assert_eq!("vulkan:2".parse::<Device>().unwrap(), Device::Vulkan(2));
        assert!("tpu".parse::<Device>().is_err());
    }
}
