use std::arch::x86_64::*;

const VECTOR_SIZE: usize = 64;
const CACHE_LINE_SIZE: usize = 8; // 8 floats (32 bytes) per cache line

#[target_feature(enable = "avx2")]
unsafe fn vector_matrix_multiply(
    vector: &[f32; VECTOR_SIZE],
    matrix: &[f32; VECTOR_SIZE * VECTOR_SIZE],
) -> [f32; VECTOR_SIZE] {
    let mut result = [0.0f32; VECTOR_SIZE];

    // 先遍历矩阵的行，然后在每行中遍历向量的列
    for i in (0..VECTOR_SIZE).step_by(CACHE_LINE_SIZE) {
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();
        let mut sum4 = _mm256_setzero_ps();
        let mut sum5 = _mm256_setzero_ps();
        let mut sum6 = _mm256_setzero_ps();
        let mut sum7 = _mm256_setzero_ps();

        for j in 0..VECTOR_SIZE {
            let vec_chunk = _mm256_set1_ps(vector[j]);

            let mat_chunk0 = _mm256_loadu_ps(
                matrix
                    .as_ptr()
                    .add((i + 0) * VECTOR_SIZE + j * CACHE_LINE_SIZE),
            );
            let mat_chunk1 = _mm256_loadu_ps(
                matrix
                    .as_ptr()
                    .add((i + 1) * VECTOR_SIZE + j * CACHE_LINE_SIZE),
            );
            let mat_chunk2 = _mm256_loadu_ps(
                matrix
                    .as_ptr()
                    .add((i + 2) * VECTOR_SIZE + j * CACHE_LINE_SIZE),
            );
            let mat_chunk3 = _mm256_loadu_ps(
                matrix
                    .as_ptr()
                    .add((i + 3) * VECTOR_SIZE + j * CACHE_LINE_SIZE),
            );
            let mat_chunk4 = _mm256_loadu_ps(
                matrix
                    .as_ptr()
                    .add((i + 4) * VECTOR_SIZE + j * CACHE_LINE_SIZE),
            );
            let mat_chunk5 = _mm256_loadu_ps(
                matrix
                    .as_ptr()
                    .add((i + 5) * VECTOR_SIZE + j * CACHE_LINE_SIZE),
            );
            let mat_chunk6 = _mm256_loadu_ps(
                matrix
                    .as_ptr()
                    .add((i + 6) * VECTOR_SIZE + j * CACHE_LINE_SIZE),
            );
            let mat_chunk7 = _mm256_loadu_ps(
                matrix
                    .as_ptr()
                    .add((i + 7) * VECTOR_SIZE + j * CACHE_LINE_SIZE),
            );

            sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(vec_chunk, mat_chunk0));
            sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(vec_chunk, mat_chunk1));
            sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(vec_chunk, mat_chunk2));
            sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(vec_chunk, mat_chunk3));
            sum4 = _mm256_add_ps(sum4, _mm256_mul_ps(vec_chunk, mat_chunk4));
            sum5 = _mm256_add_ps(sum5, _mm256_mul_ps(vec_chunk, mat_chunk5));
            sum6 = _mm256_add_ps(sum6, _mm256_mul_ps(vec_chunk, mat_chunk6));
            sum7 = _mm256_add_ps(sum7, _mm256_mul_ps(vec_chunk, mat_chunk7));
        }

        // Store the accumulated results back to the result array
        _mm256_storeu_ps(result.as_mut_ptr().add(i + 0), sum0);
        _mm256_storeu_ps(result.as_mut_ptr().add(i + 1), sum1);
        _mm256_storeu_ps(result.as_mut_ptr().add(i + 2), sum2);
        _mm256_storeu_ps(result.as_mut_ptr().add(i + 3), sum3);
        _mm256_storeu_ps(result.as_mut_ptr().add(i + 4), sum4);
        _mm256_storeu_ps(result.as_mut_ptr().add(i + 5), sum5);
        _mm256_storeu_ps(result.as_mut_ptr().add(i + 6), sum6);
        _mm256_storeu_ps(result.as_mut_ptr().add(i + 7), sum7);
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
