use std::ops::{Add, Sub, Div, Mul, AddAssign, Neg };


use itertools::Itertools;
use std::iter::zip;
// use num_traits::Float;
// use num::complex::Complex;

use crate::kernel::generic::from_usize::FromUsize;
use crate::kernel::generic::powf::Powf;

pub fn precompute_freqs_cis<T>(dim: usize, 
            max_sequence_length: usize, 
            theta: T) -> Vec<T> 
where T: Copy 
    + Default 
    + Sub<Output = T>
    + Neg<Output = T>
    + Div<Output = T>
    + Mul<Output = T>
    + FromUsize
    + Powf
   
    {
    let end = max_sequence_length * 2;
    let freqs: Vec<_> = (0..dim).step_by(2).map(
        |x| {
            let p =   T::from_usize(x)  / T::from_usize(dim) ;
            // theta.powf(p).recip()
            1 / T::powf(p)
        }
    ).collect();
    let t: Vec<_> = (0..end).collect();
    let freqs: Vec<_> = t.into_iter().cartesian_product(freqs.into_iter()).map(|(a, b)| T::from_usize(a)*b).collect();
    let ones = vec![T::from_usize(1); freqs.len()];
    let freqs_cis: Vec<_> = zip(ones, freqs).map(|(a, b)| {
        // let a= T::from(a).unwrap();
        // Complex::new(a, b.into()).to_polar()
    }).flat_map(|tup| vec![tup.0.clone(),tup.1]).collect();
    
    freqs_cis

}

#[cfg(test)]
mod test {
    // use std::f16;
    use super::*;
    // use super::super::super::compiler::zip_map::complex_zip::ComplexZipMap;
    #[test]
    fn test_freqs() {
        precompute_freqs_cis(64, 512, 10000.0);
    }

}