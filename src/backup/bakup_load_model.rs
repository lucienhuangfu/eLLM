use std::{io::Read, path::Path, fs};
use linked_hash_map::LinkedHashMap;
use anyhow::{anyhow, bail, Ok, Result};
use half::{f16, bf16};
use zip;
use repugnant_pickle::{parse_ops, RepugnantTorchTensor};

// use std::{borrow::Cow,  mem, path::Path, fs};
// use half::{f16, bf16, vec::HalfBitsVecExt};
// use nom;

pub fn load_part<P: AsRef<Path>>(filename: P, state_dict: &mut LinkedHashMap<String, Storage>) -> Result<()> {
    let mut zip_files = zip::ZipArchive::new(fs::File::open(filename)?)?;
    // let pklfn = zip_files
    //        .file_names()
    //        .find(|s| s.ends_with("/data.pkl"))
    //        .map(str::to_owned)
    //        .ok_or_else(|| anyhow!("Could not find data.pkl in archive"))?;
    // let (pfx, _) = pklfn.rsplit_once('/').unwrap();

    let data_file_name = P + "/data.pkl";
    let mut zip_data = zip_files.by_name(data_file_name)?;
    // print!("{} {}", pfx, pklfn);
    let mut buf = Vec::with_capacity(zip_data.size() as usize);
    let _ = zip_data.read_to_end(&mut buf)?;
    drop(zf);
    
    
    let tensors = parse_pkl(buf)?;
    /* 
    // println!("{:#?}", tensors);
    // let mut state_dict: LinkedHashMap<String, Storage> = LinkedHashMap::with_capacity(tensors.len());
    for tensor in tensors.into_iter() {
        let dataname = format!("{}/data/{}", pfx, tensor.storage);
        let mut zf = zp.by_name(&dataname)?;
        let mut buf = Vec::with_capacity(zf.size() as usize);
        let _ = zf.read_to_end(&mut buf);
        drop(zf);
        let tensordata = parse_tensor(buf, &tensor);
        state_dict.insert(String::from(tensor.name), tensordata);
    }*/
    Ok(())
}


fn parse_pkl(buf: Vec<u8>) -> Result<Vec<RepugnantTorchTensor>> {
    let (_remain, ops) = parse_ops::<nom::error::VerboseError<&[u8]>>(&buf)
            .map_err(|e| anyhow!("Parse error: {:?}", e))?;
    let (vals, _memo) = evaluate(&ops, true)?;
    let val = match vals.as_slice() {
        [Value::Seq(SequenceType::Dict, seq1), ..] => match seq1.as_slice() {
            [Value::Seq(SequenceType::Tuple, seq), ..] => seq,
            _ => bail!("Unexpected value in Tuple"),
        },
        _ => bail!("Unexpected toplevel type"),
    };
    let mut tensors = Vec::with_capacity(16);
    for di in val.iter() {
        let (k, v) = match di {
            Value::Seq(SequenceType::Tuple, seq) if seq.len() == 2 => (&seq[0], &seq[1]),
            _ => bail!("Could not get key/value for dictionary item"),
        };
        let k = if let Value::String(s) = k {
            *s
        } else {
            bail!("Dictionary key is not a string");
        };
        let v = match v {
            Value::Global(g, seq)
                if g.as_ref()
                    == &Value::Raw(Cow::Owned(ops::PickleOp::GLOBAL(
                        "torch._utils",
                        "_rebuild_tensor_v2",
                    ))) =>
            {
                seq
            }
            _ => bail!("error value in dict")
        };
        // println!("\nKey: {k:?}\n{v:?}");

        let (pidval, offs, shape, stride, grad) = match v.as_slice() {
            [Value::Seq(SequenceType::Tuple, seq)] => match seq.as_slice() {
                [Value::PersId(pidval), Value::Int(offs), Value::Seq(SequenceType::Tuple, shape), Value::Seq(SequenceType::Tuple, stride), Value::Bool(grad), ..] => {
                    (pidval.as_ref(), *offs as u64, shape, stride, *grad)
                }
                _ => bail!("Unexpected value in call to torch._utils._rebuild_tensor_v2"),
            },
            _ => bail!("Unexpected type in call to torch._utils._rebuild_tensor_v2"),
        };
        // println!("PID: {pidval:?}");
        let fixdim = |v: &[Value]| {
            v.iter()
                .map(|x| match x {
                    Value::Int(n) => Ok(*n as usize),
                    _ => bail!("Bad value for shape/stride item"),
                })
                .collect::<Result<Vec<_>>>()
        };
        let shape = fixdim(shape)?;
        let stride = fixdim(stride)?;
        // println!("Tensor: shape={shape:?}, stride={stride:?}, offs={offs}, grad={grad:?}");
        let (stype, sfile, sdev, slen) = match pidval {
            Value::Seq(SequenceType::Tuple, seq) => match seq.as_slice() {
                [Value::String("storage"), Value::Raw(op), Value::String(sfile), Value::String(sdev), Value::Int(slen)] => {
                    match &**op {
                        ops::PickleOp::GLOBAL("torch", styp) if styp.ends_with("Storage") => {
                            (&styp[..styp.len() - 7], *sfile, *sdev, *slen as u64)
                        }
                        _ => bail!("Unexpected storage type part of persistant ID"),
                    }
                }
                _ => bail!("Unexpected sequence in persistant ID"),
            },
            _ => bail!("Unexpected value for persistant ID"),
        };
        let stype: TensorType = stype
            .parse()
            .expect("Impossible: Parsing tensor type failed");
        let sfile = format!("{sfile}");

        // println!("PID: file={sfile}, len={slen}, type={stype:?}, dev={sdev}");
        let offs = offs * stype.size() as u64;
        tensors.push(RepugnantTorchTensor {
            name: k.to_string(),
            device: sdev.to_string(),
            tensor_type: stype,
            storage: sfile,
            storage_len: slen,
            storage_offset: offs,
            absolute_offset: offs,
            shape,
            stride,
            requires_grad: grad,
        })
    }
    Ok(tensors)
}






#[derive(Debug)]
pub enum Storage {
    Float64(Vec<f64>),
    Float32(Vec<f32>),
    Float16(Vec<f16>),
    BFloat16(Vec<bf16>),
    Int64(Vec<i64>),
    Int32(Vec<i32>),
    Int16(Vec<i16>),
    Int8(Vec<i8>),
    UInt8(Vec<u8>),
}

impl Storage {
    pub fn len(&self) -> usize {
        match self {
            Self::Float64(v) => v.len(),
            Self::Float32(v) => v.len(),
            Self::Float16(v) => v.len(),
            Self::BFloat16(v) => v.len(),
            Self::Int64(v) => v.len(),
            Self::Int32(v) => v.len(),
            Self::Int16(v) => v.len(),
            Self::Int8(v) => v.len(),
            Self::UInt8(v) => v.len(),
        }
    }

    pub fn to_f64(self) -> Vec<f64> {
        match self {
            Self::Float64(v) => v,
            Self::Float32(v) => v.into_iter().map(|x| x as f64).collect(),
            Self::Float16(v) => v.into_iter().map(|x| x.to_f64()).collect(),
            Self::BFloat16(v) => v.into_iter().map(|x| x.to_f64()).collect(),
            Self::Int64(v) => v.into_iter().map(|x| x as f64).collect(),
            Self::Int32(v) => v.into_iter().map(|x| x as f64).collect(),
            Self::Int16(v) => v.into_iter().map(|x| x as f64).collect(),
            Self::Int8(v) => v.into_iter().map(|x| x as f64).collect(),
            Self::UInt8(v) => v.into_iter().map(|x| x as f64).collect(),
        }
    }

    pub fn to_f32(self) -> Vec<f32> {
        match self {
            Self::Float64(v) => v.into_iter().map(|x| x as f32).collect(),
            Self::Float32(v) => v,
            Self::Float16(v) => v.into_iter().map(|x| x.to_f32()).collect(),
            Self::BFloat16(v) => v.into_iter().map(|x| x.to_f32()).collect(),
            Self::Int64(v) => v.into_iter().map(|x| x as f32).collect(),
            Self::Int32(v) => v.into_iter().map(|x| x as f32).collect(),
            Self::Int16(v) => v.into_iter().map(|x| x as f32).collect(),
            Self::Int8(v) => v.into_iter().map(|x| x as f32).collect(),
            Self::UInt8(v) => v.into_iter().map(|x| x as f32).collect(),
        }
    }

    pub fn to_f16(self) -> Vec<f16> {
        match self {
            Self::Float64(v) => v.into_iter().map(|x| f16::from_f64(x)).collect(),
            Self::Float32(v) => v.into_iter().map(|x| x).collect(),
            Self::Float16(v) => v,
            _ => panic!("unsupport type"),
        }
    }

    pub fn valuetype(&self) -> String {
        match self {
            Self::Float64(v) => String::from("f64"),
            Self::Float32(v) => String::from("f32"),
            Self::Float16(v) => String::from("f16"),
            Self::BFloat16(v) => String::from("bf16"),
            Self::Int64(v) => String::from("i64"),
            Self::Int32(v) => String::from("i32"),
            Self::Int16(v) => String::from("i16"),
            Self::Int8(v) => String::from("i8"),
            Self::UInt8(v) => String::from("u8"),
        }
    }
}


#[cfg(test)]
mod tests {
    // use std::collections::BTreeMap;
    // use std::io::{BufWriter, Write};
    // use std::path::Path;
    // use std::fs::{self, File};
    // use serde_pickle::Value;

    use super::*;
    // use linked_hash_map::LinkedHashMap;

    #[test]
    fn test_load() {
        let torch_file = String::from("D:/llama-7b/pytorch_model-00002-of-00002.bin");
     
        let mut state_dict: LinkedHashMap<String, Storage> = LinkedHashMap::new();
        load_part(torch_file, &mut state_dict);
    }

}