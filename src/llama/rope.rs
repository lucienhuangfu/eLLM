use itertools::Itertools;
use std::iter::zip;

use num_traits::Float;
use num::complex::Complex;

pub fn precompute_freqs_cis<T:Float>(dim: usize, 
            max_sequence_length: usize, 
            theta: T) -> Vec<T> {
    let end = max_sequence_length * 2;
    let freqs: Vec<_> = (0..dim).step_by(2).map(
        |x| {
            let p =  T::from(x as f32  / dim as f32).unwrap();
            theta.powf(p).recip()
        }
    ).collect();
    let t: Vec<_> = (0..end).collect();
    let freqs: Vec<_> = t.into_iter().cartesian_product(freqs.into_iter()).map(|(a, b)| T::from(a as f32).unwrap()*b).collect();
    
    // println!("{}", freqs_cis.len());
    let ones = vec![T::one(); freqs.len()];
    let freqs_cis: Vec<_> = zip(ones, freqs).map(|(a, b)| {
        let a= T::from(a).unwrap();
        Complex::new(a, b.into()).to_polar()
    }).flat_map(|tup| vec![tup.0.clone(),tup.1]).collect();
    
    // println!("{}", freqs_cis.len());
    // [max_sequence_length, head_num, head_size]
    // let mut output_tensor = Tensor::from_vec(vec![end, 1, dim], tensor_name, is_parameter);
    // output_tensor.data = (freqs_cis.as_mut_ptr() as *mut f32).to_owned();
    //***** 这里是因为ptensor的的to_owned被写死了 就是f32 会报错 Jason 2024.5.16
    freqs_cis

}

#[cfg(test)]
mod test {
    // use std::f16;
    use super::*;
    use super::super::super::compiler::zip_map::complex_zip::ComplexZipMap;
    #[test]
    fn test_freqs() {
        precompute_freqs_cis(64, 512, 10000.0);
    }

}