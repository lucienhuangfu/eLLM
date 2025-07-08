use std::f16;
// use num::{complex::ComplexFloat, Zero};
use std::arch::x86_64::{
    __m512h, _mm512_add_ph, _mm512_fmadd_ph, _mm512_load_ph, _mm512_mul_ph, _mm512_reduce_add_ph,
    _mm512_set1_ph, _mm512_setzero_ph, _mm512_store_ph, _mm_prefetch, _MM_HINT_T1, _MM_HINT_T2,
}; // Add this line

use super::dot_product::_dot_product;

// 标量和向量相乘并累加到结果向量
#[inline(always)]
unsafe fn scalar_vector_mul_and_acc(
    scalar1: f16,
    o: &mut [__m512h; 4],
    scalar2: f16,
    v: *const f16,
) {
    let scalar1_chunk = _mm512_set1_ph(scalar1);
    let scalar2_chunk = _mm512_set1_ph(scalar2);

    for (offset, res_chunk) in (0..128).step_by(32).zip(o.iter_mut()) {
        let v_chunk = _mm512_load_ph(v.add(offset));
        // let res1_chunk = _mm512_mul_ph(scalar1_chunk, *res_chunk);
        let res2_chunk = _mm512_mul_ph(scalar2_chunk, v_chunk);
        // *res_chunk = _mm512_add_ph(res1_chunk, res2_chunk);
        *res_chunk = _mm512_fmadd_ph(scalar1_chunk, *res_chunk, res2_chunk);
    }
}
#[inline(always)]
pub fn flash_attention(
    q: *const f16,
    K: *const f16,
    V: *const f16,
    o: *mut f16,
    inverse_sqrt_head: f16,
    length: usize,
    stride: usize,
    position: usize,
) {
    unsafe {
        // 预取下一块数据 (4 cache lines ahead)
        _mm_prefetch(K as *const i8, _MM_HINT_T2);
        _mm_prefetch(K.add(64) as *const i8, _MM_HINT_T2);
        _mm_prefetch(V as *const i8, _MM_HINT_T2);
        _mm_prefetch(V.add(64) as *const i8, _MM_HINT_T2);

        let mut m_i_1 = f16::NEG_INFINITY;
        let mut d_i_1: f16 = 0.0;

        // 初始化结果寄存器为0
        let mut o_chunks = [
            _mm512_setzero_ph(),
            _mm512_setzero_ph(),
            _mm512_setzero_ph(),
            _mm512_setzero_ph(),
        ];

        let end = position * stride;
        for offset in (0..=end).step_by(stride) {
            // 提前预取下一次循环的数据
            if offset < end {
                let next_offset = offset + stride;
                _mm_prefetch(K.add(next_offset) as *const i8, _MM_HINT_T2);
                _mm_prefetch(K.add(next_offset + 64) as *const i8, _MM_HINT_T2);
                _mm_prefetch(V.add(next_offset) as *const i8, _MM_HINT_T2);
                _mm_prefetch(V.add(next_offset + 64) as *const i8, _MM_HINT_T2);
            }

            // dot product
            let x_i = _dot_product(q, K.add(offset), length);
            // 计算缩放后的 x_i
            let x_i = x_i * inverse_sqrt_head;
            let m_i = if x_i > m_i_1 { x_i } else { m_i_1 };

            // 合并 update_exp 和 add_exp 的计算
            let update_exp = d_i_1 * f16::exp(m_i_1 - m_i);
            let add_exp = f16::exp(x_i - m_i);
            let d_i = update_exp + add_exp;
            let u = update_exp / d_i;
            let a = add_exp / d_i;

            scalar_vector_mul_and_acc(u, &mut o_chunks, a, V.add(offset));

            m_i_1 = m_i;
            d_i_1 = d_i;
        }

        for (step, res_chunk) in (0..128).step_by(32).zip(o_chunks.into_iter()) {
            _mm512_store_ph(o.add(step), res_chunk);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::slice;
    // use std::f16;
    use crate::memory::allocator::allocate_init;

    #[test]
    fn test_scalar_vector_mul_and_acc() {
        let scalar1 = 1.0;
        let scalar2 = 2.0;
        // let v: Vec<f16> = vec![2.0; 128];
        let length = 128;
        let v = allocate_init::<f16>(length, 2.0);
        let mut o = unsafe {
            [
                _mm512_set1_ph(1.0),
                _mm512_set1_ph(1.0),
                _mm512_set1_ph(1.0),
                _mm512_set1_ph(1.0),
            ]
        };
        unsafe {
            scalar_vector_mul_and_acc(scalar1, &mut o, scalar2, v);
        }
        for i in 0..4 {
            // let mut res: Vec<f16> = [0.0; 32].into_iter().map(|x| x).collect();
            let res = allocate_init::<f16>(length, 0.0);
            let res_slice = unsafe { slice::from_raw_parts(res, 32) };
            unsafe {
                _mm512_store_ph(res, o[i]);
            }
            let mut exp: Vec<f16> = vec![5.0; 32];
            // unsafe {
            //    _mm512_store_ph(exp.as_mut_ptr(), expected[i]);
            // }
            for j in 0..32 {
                assert!(f16::abs(res_slice[j] - exp[j]) < 1e-6);
            }
        }
    }

    #[test]
    fn test_flash_attention() {
        let row_size = 8;
        let length = 128;

        // let q: Vec<f16> = vec![1.0; length];
        // let K: Vec<f16> = vec![1.0; row_size * length];
        // let V: Vec<f16> = vec![1.0; row_size * length];
        // let mut o: Vec<f16> = vec![0.0; length];

        let q = allocate_init::<f16>(length, 1.0);
        let K = allocate_init::<f16>(row_size * length, 1.0);
        let V = allocate_init::<f16>(row_size * length, 1.0);
        let o = allocate_init::<f16>(length, 0.0);
        let o_slice = unsafe { slice::from_raw_parts(o, length) };
        flash_attention(q, K, V, o, 1.0, length, length, row_size - 1);
        println!("Result: {:?}", o);

        let expected: Vec<f16> = vec![1.0; length];
        for i in 0..length {
            assert!((o_slice[i] - expected[i]).abs() < 1e-6);
        }
    }
}
