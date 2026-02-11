// use num_traits::real::Real;
// use std::ops::{Add, Sub, Mul,  Div, Neg};
// use crate::common::num_traits::Sigmoid;

pub trait ZipMapTrait<T> 
// where
//     T: Copy + Default + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Neg<Output = T> + Sigmoid,
{
    fn compute(&self, 
            input_ptr1: *const T, 
            input_ptr_2: *const T,
            output_ptr: *mut T, 
    );
}

//unsafe impl Send for ZipMapTrait {}
//unsafe impl Sync for ZipMapTrait {}

















