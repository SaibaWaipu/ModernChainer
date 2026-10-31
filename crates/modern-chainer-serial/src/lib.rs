//! Serialization.
//!
//! The native `.mcn` format is a small, versioned container.  Reading legacy
//! Chainer `.npz` files is planned for a later milestone.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use modern_chainer_core::{Backend, DType, Error, Result, Tensor};

const MAGIC: &[u8; 8] = b"MCHN0001";

pub fn save<B: Backend, P: AsRef<Path>>(path: P, tensors: &[(String, Tensor<B>)]) -> Result<()> {
    let file = File::create(path)?;
    let mut out = BufWriter::new(file);
    out.write_all(MAGIC)?;
    out.write_all(&(tensors.len() as u32).to_le_bytes())?;
    for (name, tensor) in tensors {
        write_string(&mut out, name)?;
        out.write_all(&[dtype_tag(tensor.dtype())])?;
        let dims = tensor.shape().dims();
        out.write_all(&(dims.len() as u32).to_le_bytes())?;
        for &dim in dims {
            out.write_all(&(dim as u64).to_le_bytes())?;
        }
        let bytes = tensor.to_bytes()?;
        out.write_all(&(bytes.len() as u64).to_le_bytes())?;
        out.write_all(&bytes)?;
    }
    out.flush()?;
    Ok(())
}

pub fn load<B: Backend, P: AsRef<Path>>(path: P, backend: B) -> Result<Vec<(String, Tensor<B>)>> {
    let file = File::open(path)?;
    let mut input = BufReader::new(file);
    let mut magic = [0u8; 8];
    input.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(Error::invalid("not a Modern Chainer .mcn file"));
    }
    let count = read_u32(&mut input)? as usize;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let name = read_string(&mut input)?;
        let mut tag = [0u8; 1];
        input.read_exact(&mut tag)?;
        let dtype = dtype_from_tag(tag[0])?;
        let rank = read_u32(&mut input)? as usize;
        let mut dims = Vec::with_capacity(rank);
        for _ in 0..rank {
            dims.push(read_u64(&mut input)? as usize);
        }
        let len = read_u64(&mut input)? as usize;
        let mut bytes = vec![0u8; len];
        input.read_exact(&mut bytes)?;
        let tensor = Tensor::from_bytes(backend.clone(), &bytes, dims, dtype)?;
        out.push((name, tensor));
    }
    Ok(out)
}

fn write_string<W: Write>(out: &mut W, value: &str) -> Result<()> {
    let bytes = value.as_bytes();
    out.write_all(&(bytes.len() as u32).to_le_bytes())?;
    out.write_all(bytes)?;
    Ok(())
}

fn read_string<R: Read>(input: &mut R) -> Result<String> {
    let len = read_u32(input)? as usize;
    let mut bytes = vec![0u8; len];
    input.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|e| Error::invalid(format!("invalid name: {e}")))
}

fn read_u32<R: Read>(input: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    input.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(input: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    input.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn dtype_tag(dtype: DType) -> u8 {
    match dtype {
        DType::F32 => 0,
        DType::F64 => 1,
        DType::F16 => 2,
        DType::BF16 => 3,
        DType::I32 => 4,
        DType::I64 => 5,
        DType::U8 => 6,
        DType::Bool => 7,
    }
}

fn dtype_from_tag(tag: u8) -> Result<DType> {
    Ok(match tag {
        0 => DType::F32,
        1 => DType::F64,
        2 => DType::F16,
        3 => DType::BF16,
        4 => DType::I32,
        5 => DType::I64,
        6 => DType::U8,
        7 => DType::Bool,
        other => return Err(Error::invalid(format!("unknown dtype tag {other}"))),
    })
}
