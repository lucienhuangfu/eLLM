use std::{f16, mem};

use anyhow::{anyhow, Result};
use safetensors::tensor::TensorView;
use safetensors::Dtype;

pub trait FromSafetensors: Sized {
    fn from_tensor_view(tensor_view: &TensorView) -> Result<Vec<Self>>;
}

impl FromSafetensors for f16 {
    fn from_tensor_view(tensor_view: &TensorView) -> Result<Vec<Self>> {
        match tensor_view.dtype() {
            Dtype::F16 => copy_le_bytes_as_f16(tensor_view.data()),
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

#[inline]
fn copy_le_bytes_as_f16(data: &[u8]) -> Result<Vec<f16>> {
    if data.len() % mem::size_of::<f16>() != 0 {
        return Err(anyhow!(
            "Invalid F16 tensor byte length: {} is not divisible by {}",
            data.len(),
            mem::size_of::<f16>()
        ));
    }

    #[cfg(target_endian = "little")]
    {
        let len = data.len() / mem::size_of::<f16>();
        let mut out = Vec::<f16>::with_capacity(len);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), out.as_mut_ptr().cast::<u8>(), data.len());
            out.set_len(len);
        }
        Ok(out)
    }

    #[cfg(not(target_endian = "little"))]
    {
        Ok(data
            .chunks_exact(2)
            .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]))
            .collect())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_le_bytes_as_f16_preserves_values() {
        let values = [1.0f16, -2.5f16, 0.0f16, f16::INFINITY];
        let mut bytes = Vec::with_capacity(values.len() * mem::size_of::<f16>());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        let copied = copy_le_bytes_as_f16(&bytes).unwrap();
        assert_eq!(copied, values);
    }

    #[test]
    fn copy_le_bytes_as_f16_rejects_odd_byte_count() {
        let err = copy_le_bytes_as_f16(&[0, 1, 2]).unwrap_err();
        assert!(err.to_string().contains("Invalid F16 tensor byte length"));
    }
}
