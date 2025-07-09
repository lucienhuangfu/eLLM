use std::collections::HashMap;
use std::{borrow::Cow, io::Read, mem, path::Path, fs};
use anyhow::{anyhow, bail, Ok, Result};
use half::{f16, bf16, vec::HalfBitsVecExt};
use zip;
use repugnant_pickle::*;
use linked_hash_map::LinkedHashMap;
use nom;

pub fn load_from_dir_f16<P: AsRef<Path>>(dirpath: P, state_dict: &mut HashMap<String, Vec<f16>>) -> Result<()> {
    let mut tmp: LinkedHashMap<String, Storage> = LinkedHashMap::new();
    load_from_dir(dirpath, &mut tmp);
    for (k,v) in tmp {
        state_dict.insert(k, v.to_f16());
    }
    Ok(())
}

pub fn load_from_dir<P: AsRef<Path>>(dirpath: P, state_dict: &mut LinkedHashMap<String, Storage>) -> Result<()> {
    let mut files: Vec<String> = fs::read_dir(dirpath)?.into_iter().filter(|entry| {
        let p = entry.as_ref().unwrap().path();
        p.is_file() && p.to_str().unwrap().ends_with(".bin")
    }).map(|x| String::from(x.unwrap().path().to_str().unwrap())).collect();
    files.sort();
    for f in files {
        load(f, state_dict);
    }
    Ok(())
}

pub fn load<P: AsRef<Path>>(filename: P, state_dict: &mut LinkedHashMap<String, Storage>) -> Result<()> {
    let mut zp = zip::ZipArchive::new(fs::File::open(filename)?)?;
    let pklfn = zp
            .file_names()
            .find(|s| s.ends_with("/data.pkl"))
            .map(str::to_owned)
            .ok_or_else(|| anyhow!("Could not find data.pkl in archive"))?;
    let (pfx, _) = pklfn.rsplit_once('/').unwrap();
    let mut zf = zp.by_name(&pklfn)?;
    let mut buf = Vec::with_capacity(zf.size() as usize);
    let _ = zf.read_to_end(&mut buf)?;
    drop(zf);
    let tensors = parse_pkl(buf)?;
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
    }
    Ok(())
}

/*
pub fn load_meta_from_dir<P: AsRef<Path>>(dirpath: P, meta_dict: &mut LinkedHashMap<String, RepugnantTorchTensor>) -> Result<()> {
    let mut files: Vec<String> = fs::read_dir(dirpath)?.into_iter().filter(|entry| {
        let p = entry.as_ref().unwrap().path();
        p.is_file() && p.to_str().unwrap().ends_with(".bin")
    }).map(|x| String::from(x.unwrap().path().to_str().unwrap())).collect();
    files.sort();
    for f in files {
        load_meta(f, meta_dict);
    }
    Ok(())
}

pub fn load_meta<P: AsRef<Path>>(filename: P, meta_dict: &mut LinkedHashMap<String, RepugnantTorchTensor>) -> Result<()> {
    let mut zp = zip::ZipArchive::new(fs::File::open(filename)?)?;
    let pklfn = zp
            .file_names()
            .find(|s| s.ends_with("/data.pkl"))
            .map(str::to_owned)
            .ok_or_else(|| anyhow!("Could not find data.pkl in archive"))?;
    let mut zf = zp.by_name(&pklfn)?;
    let mut buf = Vec::with_capacity(zf.size() as usize);
    let _ = zf.read_to_end(&mut buf)?;
    drop(zf);
    let tensors = parse_pkl(buf)?;
    for tensor in tensors.into_iter() {
        meta_dict.insert(String::from(tensor.name.clone()), tensor);
    }
    Ok(())
}
 */

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

fn parse_tensor(data: Vec<u8>, tensor: &RepugnantTorchTensor) -> Storage {
    let mut data = mem::ManuallyDrop::new(data);
    let tensor_type = &tensor.tensor_type;
    let size = tensor.storage_len as usize;
    unsafe {
        let ptr = data.as_mut_ptr(); 
        match tensor_type {
            TensorType::Float32 =>
                Storage::Float32(Vec::from_raw_parts(ptr as *mut f32, size, size)),
            TensorType::Float64 =>
                Storage::Float64(Vec::from_raw_parts(ptr as *mut f64, size, size)),
            TensorType::Float16 => {
                let ptr2 = ptr as *mut u16;
                let v = Vec::from_raw_parts(ptr2, size, size);
                Storage::Float16(v.reinterpret_into::<f16>())
            }
            TensorType::BFloat16 => {
                let ptr2 = ptr as *mut u16;
                let v = Vec::from_raw_parts(ptr2, size, size);
                Storage::BFloat16(v.reinterpret_into::<bf16>())
            }
            TensorType::Int16 => 
                Storage::Int16(Vec::from_raw_parts(ptr as *mut i16, size, size)),
            TensorType::Int32 => 
                Storage::Int32(Vec::from_raw_parts(ptr as *mut i32, size, size)),
            TensorType::Int64 => 
                Storage::Int64(Vec::from_raw_parts(ptr as *mut i64, size, size)),
            TensorType::Int8 => 
                Storage::Int8(Vec::from_raw_parts(ptr as *mut i8, size, size)),
            TensorType::UInt8 => 
                Storage::UInt8(Vec::from_raw_parts(ptr as *mut u8, size, size)),
            _ => panic!("cannot load tensor {:#?}",  tensor_type),
        }
    }
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

#[cfg(test)]
mod tests {
    // use std::collections::BTreeMap;
    use std::io::{BufWriter, Write};
    use std::path::Path;
    use std::fs::{self, File};
    use serde_pickle::Value;
    use super::*;
    use linked_hash_map::LinkedHashMap;

    #[test]
    fn test_load() {
        let torch_file = String::from("D:/llama-7b/pytorch_model-00002-of-00002.bin");
        let convert_file = String::from("D:/convert.pkl");
        let mut state_dict: LinkedHashMap<String, Storage> = LinkedHashMap::new();
        load(torch_file, &mut state_dict);


        /* 
        let state_dict2 = read_test_file(convert_file);
        assert_eq!(state_dict.len(), state_dict2.len());

        let mut iter = state_dict2.into_iter();
        for (k,v) in state_dict.into_iter() {
            let (k2,v2) = iter.next().unwrap();
            assert_eq!(k,k2);
            assert_eq!(v.to_f64(), v2);
        }

        fn read_test_file<P: AsRef<Path>>(filename: P) -> Vec<(String, Vec<f64>)> {
            let content = fs::read(filename).unwrap();
            let deserialized: Vec<(String, Vec<f64>)> = serde_pickle::from_slice(&content, Default::default()).unwrap();
            deserialized
        }*/
    }

    /*
    #[test]
    fn test_load_from_dir() {
        let mut state_dict: LinkedHashMap<String, Storage> = LinkedHashMap::new();
        load_from_dir("/home/ubuntu/llama2-7B-hf", &mut state_dict);
        let keys = vec!["model.embed_tokens.weight","model.layers.0.input_layernorm.weight","model.layers.0.post_attention_layernorm.weight","model.layers.9.post_attention_layernorm.weight","model.layers.29.post_attention_layernorm.weight","model.layers.4.post_attention_layernorm.weight","model.layers.5.self_attn.k_proj.weight","model.layers.30.post_attention_layernorm.weight","model.layers.20.post_attention_layernorm.weight"];
        let content = fs::read("/home/ubuntu/llama2-7B-hf/values.pickle").unwrap();
        let deserialized: Vec<Vec<f64>> = serde_pickle::from_slice(&content, Default::default()).unwrap();
        for (v1,v2) in keys.iter().map(|k| state_dict.remove(*k).unwrap().to_f64()).zip(deserialized.into_iter()) {
            assert_eq!(v1, v2);
        }
        // let data: Vec<(String, Vec<f32>)> = state_dict.into_iter().map(|x| (x.0, x.1.to_f32())).collect();
        // let file = File::create("example.pickle").unwrap();
        // let mut writer = BufWriter::new(file);
        // let buffer = serde_pickle::to_vec(&data, Default::default()).unwrap();
        // writer.write_all(&buffer);
        
        // println!("{}", state_dict.len());
        // for (k,v) in state_dict.iter() {
        //     println!("{}", v.valuetype());
        // }
    } */
}