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

    pub const fn is_float16(self) -> bool {
        matches!(self, DType::F16 | DType::BF16)
    }
}

pub fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i16;
    let frac = bits & 0x7fffff;
    if exp == 0xff {
        return sign | 0x7c00 | if frac == 0 { 0 } else { 0x200 };
    }
    let h_exp = exp - 112;
    if h_exp >= 0x1f {
        return sign | 0x7c00;
    }
    if h_exp <= 0 {
        if h_exp < -10 {
            return sign;
        }
        let m = frac | 0x800000;
        let shift = 14 - h_exp;
        let rounding_bias = ((m >> (shift - 1)) & 1) + 0xfff;
        let h = (m + rounding_bias) >> shift;
        let h = h as u16;
        h | sign
    } else {
        let h = sign | ((h_exp as u16) << 10) | ((frac >> 13) as u16);
        let rounding = (frac >> 12) & 1;
        let sticky = (frac & 0xfff) != 0;
        let lsb = h & 1;
        if rounding == 1 && (sticky || lsb == 1) {
            h + 1
        } else {
            h
        }
    }
}

pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let out = if exp == 0x1f {
        sign | 0x7f800000 | (frac << 13)
    } else if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mut e = 0u32;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e += 1;
            }
            f &= 0x3ff;
            let exp = 0x7f - 14 - e;
            sign | (exp << 23) | (f << 13)
        }
    } else {
        sign | ((exp + 112) << 23) | (frac << 13)
    };
    f32::from_bits(out)
}

pub fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounding = (bits >> 15) & 1;
    let sticky = (bits & 0x7fff) != 0;
    let mut out = (bits >> 16) as u16;
    if rounding == 1 && (sticky || lsb == 1) {
        out = out.wrapping_add(1);
    }
    out
}

pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
