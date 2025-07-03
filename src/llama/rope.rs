use std::ops::{Add, Sub, Div, Mul, AddAssign, Neg };


use itertools::Itertools;
use std::iter::zip;

use crate::kernel::generic::from_usize::FromUsize;
use crate::kernel::generic::powf::Powf;



fn complex_to_polar<T>(re: T, im: T) -> (T, T)
where
    T: Copy + From<f64> + Into<f64> + PartialOrd,
{
    let re_f: f64 = re.into();
    let im_f: f64 = im.into();
    let r = (re_f * re_f + im_f * im_f).sqrt();
    let theta = f64::atan2(im_f, re_f);
    (T::from(r), T::from(theta))
}

pub fn precompute_freqs_cis<T>(dim: usize, 
            max_sequence_length: usize, 
            theta: f64) -> Vec<T> 
where T: Copy 
    + Default 
    + Sub<Output = T>
    + Neg<Output = T>
    + Div<Output = T>
    + Mul<Output = T>
    + FromUsize
    + Powf
    + From<f64>
    + Into<f64>
    {
    let end = max_sequence_length * 2;
    let freqs: Vec<_> = (0..dim).step_by(2).map(
        |x| {
            let p =   T::from_usize(x)  / T::from_usize(dim) ;
            // theta.powf(p).recip()
            T::from_usize(1) / T::powf(theta, p)
        }
    ).collect();
    let t: Vec<_> = (0..end).collect();
    let freqs: Vec<_> = t.into_iter().cartesian_product(freqs.into_iter()).map(|(a, b)| T::from_usize(a)*b).collect();
    let ones = vec![T::from_usize(1); freqs.len()];
    let freqs_cis: Vec<_> = zip(ones, freqs).map(|(a, b)| {
        let (r, theta) = complex_to_polar(a, b);
        (r, theta)
    }).flat_map(|tup| vec![tup.0.clone(), tup.1]).collect();
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