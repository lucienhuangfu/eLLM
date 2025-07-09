// use super::asmsimd::*;
use std::f16;
use std::ptr;

use std::arch::x86_64::{_mm512_fmadd_ph, _mm512_load_ph, _mm512_reduce_add_ph, _mm512_setzero_ph};
#[inline(always)]
pub fn _dot_product(ptr1: *const f16, ptr2: *const f16, length: usize) -> f16 {
    // print!("dot product length {}", length);
    unsafe {
        let mut chunks_simd = _mm512_setzero_ph();
        for x in (0..length).step_by(32) {
            let x1 = _mm512_load_ph(ptr1.add(x));
            let x2 = _mm512_load_ph(ptr2.add(x));
            chunks_simd = _mm512_fmadd_ph(x1, x2, chunks_simd);
        }
        let product = _mm512_reduce_add_ph(chunks_simd);
        product
    }
}

#[inline(always)]
pub fn dot_product(
    input_ptr1: *const f16,
    input_ptr2: *const f16,
    output_ptr: *mut f16,
    length: usize,
) {
    unsafe {
        /*
        let mut chunks_sum: f16 = 0.0;
        let mut chunks_simd = _mm512_setzero_ph();
        for (ptr1, ptr2) in (0..length).step_by(32).map(|x| (input_ptr1.add(x), input_ptr2.add(x))) {
            let x1 = _mm512_load_ph(ptr1);
            let x2 = _mm512_load_ph(ptr2);
            chunks_simd = _mm512_fmadd_ph(x1, x2, chunks_simd);
        }
        chunks_sum = _mm512_reduce_add_ph(chunks_simd);
        */
        let product = _dot_product(input_ptr1, input_ptr2, length);
        ptr::write(output_ptr, product);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::allocator::allocate_init;

    #[test]
    fn test_dot_product() {
        let length = 128;
        let mut x1 = allocate_init::<f16>(length, 1.0);
        let mut x2 = allocate_init::<f16>(length, 1.0);
        let mut result: f16 = 0.0;
        dot_product(x1, x2, &mut result, length);
        assert_eq!(result, 128.0);
    }
}
