#![feature(stdsimd)]

use std::f16;
use std::arch::x86_64::*;

/// 使用 AVX-512 指令进行向量矩阵乘法，并在寄存器中完成累加操作。
///
/// 说明:
/// 1. **寄存器利用**：在每个循环中使用 16 个寄存器（`sum0` 到 `sum15`）来存储中间结果，以充分利用 SIMD 寄存器。
/// 2. **分块处理**：每次处理 16 行，分成 16 个块进行计算。
/// 3. **向量乘法**：使用 `_mm512_mul_ph` 进行向量乘法，然后使用 `_mm512_add_ph` 累加结果。
/// 4. **横向加法**：使用 `_mm512_reduce_add_ph` 在寄存器中完成最后的累加操作。
/// 5. **结果存储**：将 16 个结果归集到一个寄存器中，然后一次性写入结果。
/// 6. **缓存预取**：在外循环中使用 `_mm_prefetch` 指令预取下一块数据，以提高内存访问效率。
fn vector_matrix_multiply(vector: &[f16], matrix: &[f16], rows: usize) -> Vec<f16> {
    assert_eq!(vector.len(), 128);
    assert_eq!(matrix.len(), rows * 128);

    let mut result = vec![0.0); rows];

    unsafe {
        for i in (0..rows).step_by(16) {
            // Prefetch the next blocks of the matrix
            if i + 16 < rows {
                let base_prefetch_ptr = matrix.as_ptr().add((i + 16) * 128);
                _mm_prefetch(base_prefetch_ptr as *const i8, _MM_HINT_T0);
                _mm_prefetch(base_prefetch_ptr.add(128) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base_prefetch_ptr.add(256) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base_prefetch_ptr.add(384) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base_prefetch_ptr.add(512) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base_prefetch_ptr.add(640) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base_prefetch_ptr.add(768) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base_prefetch_ptr.add(896) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base_prefetch_ptr.add(1024) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base_prefetch_ptr.add(1152) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base_prefetch_ptr.add(1280) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base_prefetch_ptr.add(1408) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base_prefetch_ptr.add(1536) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base_prefetch_ptr.add(1664) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base_prefetch_ptr.add(1792) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base_prefetch_ptr.add(1920) as *const i8, _MM_HINT_T0);
            }

            let mut sum0 = _mm512_setzero_ph();
            let mut sum1 = _mm512_setzero_ph();
            let mut sum2 = _mm512_setzero_ph();
            let mut sum3 = _mm512_setzero_ph();
            let mut sum4 = _mm512_setzero_ph();
            let mut sum5 = _mm512_setzero_ph();
            let mut sum6 = _mm512_setzero_ph();
            let mut sum7 = _mm512_setzero_ph();
            let mut sum8 = _mm512_setzero_ph();
            let mut sum9 = _mm512_setzero_ph();
            let mut sum10 = _mm512_setzero_ph();
            let mut sum11 = _mm512_setzero_ph();
            let mut sum12 = _mm512_setzero_ph();
            let mut sum13 = _mm512_setzero_ph();
            let mut sum14 = _mm512_setzero_ph();
            let mut sum15 = _mm512_setzero_ph();

            for j in (0..128).step_by(32) {
                let vec_chunk = _mm512_loadu_ph(vector.as_ptr().add(j) as *const _);

                let base_ptr = matrix.as_ptr().add(i * 128 + j);
                let mat_chunk0 = _mm512_loadu_ph(base_ptr as *const _);
                let mat_chunk1 = _mm512_loadu_ph(base_ptr.add(128) as *const _);
                let mat_chunk2 = _mm512_loadu_ph(base_ptr.add(256) as *const _);
                let mat_chunk3 = _mm512_loadu_ph(base_ptr.add(384) as *const _);
                let mat_chunk4 = _mm512_loadu_ph(base_ptr.add(512) as *const _);
                let mat_chunk5 = _mm512_loadu_ph(base_ptr.add(640) as *const _);
                let mat_chunk6 = _mm512_loadu_ph(base_ptr.add(768) as *const _);
                let mat_chunk7 = _mm512_loadu_ph(base_ptr.add(896) as *const _);
                let mat_chunk8 = _mm512_loadu_ph(base_ptr.add(1024) as *const _);
                let mat_chunk9 = _mm512_loadu_ph(base_ptr.add(1152) as *const _);
                let mat_chunk10 = _mm512_loadu_ph(base_ptr.add(1280) as *const _);
                let mat_chunk11 = _mm512_loadu_ph(base_ptr.add(1408) as *const _);
                let mat_chunk12 = _mm512_loadu_ph(base_ptr.add(1536) as *const _);
                let mat_chunk13 = _mm512_loadu_ph(base_ptr.add(1664) as *const _);
                let mat_chunk14 = _mm512_loadu_ph(base_ptr.add(1792) as *const _);
                let mat_chunk15 = _mm512_loadu_ph(base_ptr.add(1920) as *const _);

                sum0 = _mm512_add_ph(sum0, _mm512_mul_ph(vec_chunk, mat_chunk0));
                sum1 = _mm512_add_ph(sum1, _mm512_mul_ph(vec_chunk, mat_chunk1));
                sum2 = _mm512_add_ph(sum2, _mm512_mul_ph(vec_chunk, mat_chunk2));
                sum3 = _mm512_add_ph(sum3, _mm512_mul_ph(vec_chunk, mat_chunk3));
                sum4 = _mm512_add_ph(sum4, _mm512_mul_ph(vec_chunk, mat_chunk4));
                sum5 = _mm512_add_ph(sum5, _mm512_mul_ph(vec_chunk, mat_chunk5));
                sum6 = _mm512_add_ph(sum6, _mm512_mul_ph(vec_chunk, mat_chunk6));
                sum7 = _mm512_add_ph(sum7, _mm512_mul_ph(vec_chunk, mat_chunk7));
                sum8 = _mm512_add_ph(sum8, _mm512_mul_ph(vec_chunk, mat_chunk8));
                sum9 = _mm512_add_ph(sum9, _mm512_mul_ph(vec_chunk, mat_chunk9));
                sum10 = _mm512_add_ph(sum10, _mm512_mul_ph(vec_chunk, mat_chunk10));
                sum11 = _mm512_add_ph(sum11, _mm512_mul_ph(vec_chunk, mat_chunk11));
                sum12 = _mm512_add_ph(sum12, _mm512_mul_ph(vec_chunk, mat_chunk12));
                sum13 = _mm512_add_ph(sum13, _mm512_mul_ph(vec_chunk, mat_chunk13));
                sum14 = _mm512_add_ph(sum14, _mm512_mul_ph(vec_chunk, mat_chunk14));
                sum15 = _mm512_add_ph(sum15, _mm512_mul_ph(vec_chunk, mat_chunk15));
            }

            // Horizontal add within each register
            let sum0 = _mm512_reduce_add_ph(sum0);
            let sum1 = _mm512_reduce_add_ph(sum1);
            let sum2 = _mm512_reduce_add_ph(sum2);
            let sum3 = _mm512_reduce_add_ph(sum3);
            let sum4 = _mm512_reduce_add_ph(sum4);
            let sum5 = _mm512_reduce_add_ph(sum5);
            let sum6 = _mm512_reduce_add_ph(sum6);
            let sum7 = _mm512_reduce_add_ph(sum7);
            let sum8 = _mm512_reduce_add_ph(sum8);
            let sum9 = _mm512_reduce_add_ph(sum9);
            let sum10 = _mm512_reduce_add_ph(sum10);
            let sum11 = _mm512_reduce_add_ph(sum11);
            let sum12 = _mm512_reduce_add_ph(sum12);
            let sum13 = _mm512_reduce_add_ph(sum13);
            let sum14 = _mm512_reduce_add_ph(sum14);
            let sum15 = _mm512_reduce_add_ph(sum15);

            // Combine all sums into one register
            let combined_sum = _mm512_set_ph(
                _mm512_cvtss_f32(sum15),
                _mm512_cvtss_f32(sum14),
                _mm512_cvtss_f32(sum13),
                _mm512_cvtss_f32(sum12),
                _mm512_cvtss_f32(sum11),
                _mm512_cvtss_f32(sum10),
                _mm512_cvtss_f32(sum9),
                _mm512_cvtss_f32(sum8),
                _mm512_cvtss_f32(sum7),
                _mm512_cvtss_f32(sum6),
                _mm512_cvtss_f32(sum5),
                _mm512_cvtss_f32(sum4),
                _mm512_cvtss_f32(sum3),
                _mm512_cvtss_f32(sum2),
                _mm512_cvtss_f32(sum1),
                _mm512_cvtss_f32(sum0),
            );

            // Store the combined result
            _mm512_storeu_ph(result.as_mut_ptr().add(i) as *mut _, combined_sum);
        }
    }

    result
}

fn main() {
    // Example usage
    let vector: Vec<f16> = vec![1.0); 128];
    let rows = 256;
    let matrix: Vec<f16> = vec![1.0); rows * 128];

    let result = vector_matrix_multiply(&vector, &matrix, rows);

    for val in result {
        println!("{}", val.to_f32());
    }
}
