use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    F16,
    BF16,
    I32,
    I64,
    U8,
    Bool,
}

impl DType {
    pub const fn size_in_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::F16 | DType::BF16 => 2,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::U8 | DType::Bool => 1,
        }
    }

    pub const fn is_float(self) -> bool {
        matches!(self, DType::F32 | DType::F64 | DType::F16 | DType::BF16)
    }

    pub const fn is_int(self) -> bool {
        matches!(self, DType::I32 | DType::I64 | DType::U8)
    }

    pub const fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::U8 => "u8",
            DType::Bool => "bool",
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl DType {
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "f32" | "float32" => DType::F32,
            "f64" | "float64" => DType::F64,
            "f16" | "float16" => DType::F16,
            "bf16" | "bfloat16" => DType::BF16,
            "i32" | "int32" => DType::I32,
            "i64" | "int64" => DType::I64,
            "u8" | "uint8" => DType::U8,
            "bool" => DType::Bool,
            _ => return None,
        })
    }
}
