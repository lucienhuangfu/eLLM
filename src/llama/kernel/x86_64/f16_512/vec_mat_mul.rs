use std::arch::x86_64::{
    _mm512_fmadd_ph, _mm512_load_ph, _mm512_reduce_add_ph, _mm512_setzero_ph, _mm512_store_ph,
    _mm_prefetch, _MM_HINT_T0,
};
use std::ops::{Add, Mul};
use std::ptr;
// use std::arch::asm;

use super::dot_product::dot_product;

pub fn vec_mat_mul(
    a_ptr: *const f16,
    b_ptr: *const f16,
    c_ptr: *mut f16,
    col_size: usize,
    row_size: usize,
) {
    unsafe {
        let mut vector_chunks = [
            _mm512_setzero_ph(),
            _mm512_setzero_ph(),
            _mm512_setzero_ph(),
            _mm512_setzero_ph(),
        ];

        for (i, v) in vector_chunks.iter_mut().enumerate() {
            *v = _mm512_load_ph(a_ptr.add(i * 32));
        }

        for i in 0..row_size {
            let m_ptr = b_ptr.add(i * col_size);
            let mut chunks_simd = _mm512_setzero_ph();

            for (k, vec_chunk) in vector_chunks.iter().enumerate() {
                _mm_prefetch(m_ptr.add((k + 1) * 32) as *const i8, _MM_HINT_T0);
                let x2 = _mm512_load_ph(m_ptr.add(k * 32));
                chunks_simd = _mm512_fmadd_ph(*vec_chunk, x2, chunks_simd);
            }

            let result = _mm512_reduce_add_ph(chunks_simd);
            ptr::write(c_ptr.add(i), result);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_mat_vec_mul() {
        let col_size = 64;
        let row_size = 32;

        let v1 = vec![1.0; col_size];

        let v2 = vec![1.0; row_size * col_size];
        let mut v3 = vec![0.0; row_size];
        let result = vec![64.0; row_size];
        vec_mat_mul(
            v1.as_ptr(),
            v2.as_ptr(),
            v3.as_mut_ptr(),
            col_size,
            row_size,
        );
        assert_eq!(v3[..], result[..]);
    }
}
