use itertools::Itertools;
use std::iter::zip;

// use num_traits::Float;
use num::complex::Complex;

pub fn precompute_freqs_cis(dim: usize, 
            max_sequence_length: usize, 
            theta: f32) -> Vec<f32> {
    let end = max_sequence_length * 2;
    let freqs: Vec<_> = (0..dim).step_by(2).map(
        |x| {
            let p =  x as f32 / dim as f32;
            theta.powf(p).recip()
        }
    ).collect();
    let t: Vec<_> = (0..end).collect();
    let freqs: Vec<_> = t.into_iter().cartesian_product(freqs.into_iter()).map(|(a, b)| a as f32 * b).collect();
    
    // println!("{}", freqs_cis.len());
    let ones = vec![1.0; freqs.len()];
    let freqs_cis: Vec<_> = zip(ones, freqs).map(|(a, b)| {
        Complex::new(a, b).to_polar()
    }).flat_map(|tup| vec![tup.0.clone(),tup.1]).collect();
    
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