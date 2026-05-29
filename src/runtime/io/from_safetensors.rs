use std::f16;

use anyhow::{anyhow, Result};
use safetensors::tensor::TensorView;
use safetensors::Dtype;

pub trait FromSafetensors: Sized {
    fn from_tensor_view(tensor_view: &TensorView) -> Result<Vec<Self>>;
}

impl FromSafetensors for f16 {
    fn from_tensor_view(tensor_view: &TensorView) -> Result<Vec<Self>> {
        match tensor_view.dtype() {
            Dtype::F16 => Ok(tensor_view
                .data()
                .chunks_exact(2)
                .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]))
                .collect()),
            Dtype::F32 => Ok(tensor_view
                .data()
                .chunks_exact(4)
                .map(|chunk| {
                    let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    val as f16
                })
                .collect()),
            Dtype::BF16 => Ok(tensor_view
                .data()
                .chunks_exact(2)
                .map(|chunk| {
                    let val_u16 = u16::from_le_bytes([chunk[0], chunk[1]]);
                    let val_f32 = f32::from_bits((val_u16 as u32) << 16);
                    val_f32 as f16
                })
                .collect()),
            _ => Err(anyhow!(
                "Unsupported tensor dtype for f16: {:?}",
                tensor_view.dtype()
            )),
        }
    }
}

impl FromSafetensors for f32 {
    fn from_tensor_view(tensor_view: &TensorView) -> Result<Vec<Self>> {
        match tensor_view.dtype() {
            Dtype::F16 => Ok(tensor_view
                .data()
                .chunks_exact(2)
                .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]) as f32)
                .collect()),
            Dtype::F32 => Ok(tensor_view
                .data()
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()),
            Dtype::BF16 => Ok(tensor_view
                .data()
                .chunks_exact(2)
                .map(|chunk| {
                    let val_u16 = u16::from_le_bytes([chunk[0], chunk[1]]);
                    f32::from_bits((val_u16 as u32) << 16)
                })
                .collect()),
            _ => Err(anyhow!(
                "Unsupported tensor dtype for f32: {:?}",
                tensor_view.dtype()
            )),
        }
    }
}

impl FromSafetensors for f64 {
    fn from_tensor_view(tensor_view: &TensorView) -> Result<Vec<Self>> {
        match tensor_view.dtype() {
            Dtype::F16 => Ok(tensor_view
                .data()
                .chunks_exact(2)
                .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]) as f64)
                .collect()),
            Dtype::F32 => Ok(tensor_view
                .data()
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64)
                .collect()),
            Dtype::BF16 => Ok(tensor_view
                .data()
                .chunks_exact(2)
                .map(|chunk| {
                    let val_u16 = u16::from_le_bytes([chunk[0], chunk[1]]);
                    f32::from_bits((val_u16 as u32) << 16) as f64
                })
                .collect()),
            _ => Err(anyhow!(
                "Unsupported tensor dtype for f64: {:?}",
                tensor_view.dtype()
            )),
        }
    }
}
