use std::arch::x86_64::*;

// 标量和向量相乘并累加到结果向量
// 使用 AVX-256 指令集将标量加载到寄存器中
// 将向量分成八个部分，并行处理每个部分
// 将标量和向量的每个部分相乘，并累加到结果寄存器中
unsafe fn scalar_vector_mul_and_add(scalar: f32, vector: &[f32], res_chunks: &mut [_m256; 8]) {
    let scalar_vec = _mm256_set1_ps(scalar);
    for i in 0..8 {
        let vec_chunk = _mm256_loadu_ps(vector.as_ptr().add(i * 8) as *const _);
        let res_chunk = _mm256_mul_ps(scalar_vec, vec_chunk);
        res_chunks[i] = _mm256_add_ps(res_chunks[i], res_chunk);
    }
}

// 加权平均向量计算
// 接受列向量、矩阵、行数、列数和结果向量
// 确保矩阵的列数是 64，并且列向量和矩阵的大小是正确的
// 分成2个核进行计算
// 初始化结果寄存器为 0
// 遍历每一行，调用 scalar_vector_mul_and_add 函数进行标量乘法和累加操作
// 将结果寄存器中的值写入内存
unsafe fn weighted_average_vector(
    column_vector: &[f32],
    matrix: &[f32],
    rows: usize,
    cols: usize,
    result: &mut [f32],
) {
    assert_eq!(cols, 128); // 确保矩阵的列数是128
    assert_eq!(column_vector.len(), rows);
    assert_eq!(matrix.len(), rows * cols);
    assert_eq!(result.len(), cols);

    // 初始化结果寄存器为0
    let mut res_chunks = [
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    ];

    // 计算加权平均向量
    for i in 0..rows {
        let scalar = column_vector[i];
        let matrix_row = &matrix[i * cols..(i + 1) * cols];
        scalar_vector_mul_and_add(scalar, matrix_row, &mut res_chunks);
    }

    // 将结果寄存器写入内存
    for i in 0..8 {
        _mm256_storeu_ps(result.as_mut_ptr().add(i * 8) as *mut _, res_chunks[i]);
    }
}

fn main() {
    // 示例数据
    let column_vector: Vec<F32> = vec![1.0; 512]; // 1.0 in f32
    let matrix: Vec<F32> = vec![1.0; 512 * 128]; // 1.0 in f32
    let mut result: Vec<F32> = vec![0.0; 128];

    unsafe {
        weighted_average_vector(&column_vector, &matrix, 512, 128, &mut result);
    }

    // 打印结果
    for val in result {
        println!("{}", val);
    }
}