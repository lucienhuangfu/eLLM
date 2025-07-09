use std::arch::x86_64::*;

const VECTOR_SIZE: usize = 64;

#[target_feature(enable = "avx2")]
unsafe fn vector_matrix_multiply(
    vector: &[f32; VECTOR_SIZE],
    matrix: &[f32; VECTOR_SIZE * VECTOR_SIZE],
) -> [f32; VECTOR_SIZE] {
    let mut result = [0.0f32; VECTOR_SIZE];
    // 为什么不在寄存器里累加最后的结果
    for i in 0..VECTOR_SIZE {
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        let row_offset = i * VECTOR_SIZE;

        for j in (0..VECTOR_SIZE).step_by(32) {
            // Prefetch the next chunks of data
            _mm_prefetch(
                matrix.as_ptr().add(row_offset + j) as *const i8,
                _MM_HINT_T0,
            );
            _mm_prefetch(
                matrix.as_ptr().add(row_offset + j + 8) as *const i8,
                _MM_HINT_T0,
            );
            _mm_prefetch(
                matrix.as_ptr().add(row_offset + j + 16) as *const i8,
                _MM_HINT_T0,
            );
            _mm_prefetch(
                matrix.as_ptr().add(row_offset + j + 24) as *const i8,
                _MM_HINT_T0,
            );

            let vec_chunk0 = _mm256_loadu_ps(&vector[j]);
            let vec_chunk1 = _mm256_loadu_ps(&vector[j + 8]);
            let vec_chunk2 = _mm256_loadu_ps(&vector[j + 16]);
            let vec_chunk3 = _mm256_loadu_ps(&vector[j + 24]);

            let mat_chunk0 = _mm256_loadu_ps(matrix.as_ptr().add(row_offset + j));
            let mat_chunk1 = _mm256_loadu_ps(matrix.as_ptr().add(row_offset + j + 8));
            let mat_chunk2 = _mm256_loadu_ps(matrix.as_ptr().add(row_offset + j + 16));
            let mat_chunk3 = _mm256_loadu_ps(matrix.as_ptr().add(row_offset + j + 24));

            sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(vec_chunk0, mat_chunk0));
            sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(vec_chunk1, mat_chunk1));
            sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(vec_chunk2, mat_chunk2));
            sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(vec_chunk3, mat_chunk3));
        }

        let sum4 = _mm256_add_ps(sum0, sum1);
        let sum5 = _mm256_add_ps(sum2, sum3);
        let sum6 = _mm256_add_ps(sum4, sum5);

        // Horizontal addition of the elements in the AVX registers
        let sum7 = _mm256_hadd_ps(sum6, sum6);
        let sum8 = _mm256_hadd_ps(sum7, sum7);

        // Extract the lower 128 bits and add the two 128-bit halves
        let low128 = _mm256_extractf128_ps(sum8, 0);
        let high128 = _mm256_extractf128_ps(sum8, 1);
        let final_sum = _mm_add_ps(low128, high128);

        // Store the result
        result[i] = _mm_cvtss_f32(final_sum);
    }

    result
}

fn main() {
    let vector: [f32; VECTOR_SIZE] = [1.0; VECTOR_SIZE];
    let matrix: [f32; VECTOR_SIZE * VECTOR_SIZE] = [1.0; VECTOR_SIZE * VECTOR_SIZE];

    unsafe {
        let result = vector_matrix_multiply(&vector, &matrix);
        println!("{:?}", result);
    }
}
