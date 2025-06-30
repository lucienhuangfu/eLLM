use std::arch::x86_64::*;

use std::f16;
// 定义数据类型
// type F16 = u16; // f16在Rust中没有直接的类型，需要使用u16表示
// 标量和向量相乘并累加到结果向量
unsafe fn scalar_vector_mul_and_add(scalar: F16, vector: &[F16], res_chunks: &mut [_m512bh; 4]) {
    let scalar_f32 = _mm512_cvtph_ps(_mm512_set1_ph(scalar));
    for i in 0..4 {
        let vec_chunk = _mm512_loadu_ph(vector.as_ptr().add(i * 32) as *const _);
        let res_chunk = _mm512_mul_ps(scalar_f32, _mm512_cvtph_ps(vec_chunk));
        let res_chunk_f16 = _mm512_cvtps_ph(res_chunk);
        res_chunks[i] = _mm512_add_ph(res_chunks[i], res_chunk_f16);
    }
}

// 加权平均向量计算
unsafe fn weighted_average_vector(
    column_vector: &[F16],
    matrix: &[F16],
    rows: usize,
    cols: usize,
    result: &mut [F16],
) {
    assert_eq!(cols, 128); // 确保矩阵的列数是128
    assert_eq!(column_vector.len(), rows);
    assert_eq!(matrix.len(), rows * cols);
    assert_eq!(result.len(), cols);

    // 初始化结果寄存器为0
    let mut res_chunks = [
        _mm512_setzero_ph(),
        _mm512_setzero_ph(),
        _mm512_setzero_ph(),
        _mm512_setzero_ph(),
    ];

    // 计算加权平均向量
    for i in 0..rows {
        let scalar = column_vector[i];
        let matrix_row = &matrix[i * cols..(i + 1) * cols];
        scalar_vector_mul_and_add(scalar, matrix_row, &mut res_chunks);
    }

    // 将结果寄存器写入内存
    for i in 0..4 {
        _mm512_storeu_ph(result.as_mut_ptr().add(i * 32) as *mut _, res_chunks[i]);
    }
}

fn main() {
    // 示例数据
    let column_vector: Vec<F16> = vec![0x3c00; 512]; // 1.0 in f16
    let matrix: Vec<F16> = vec![0x3c00; 512 * 128]; // 1.0 in f16
    let mut result: Vec<F16> = vec![0; 128];

    unsafe {
        weighted_average_vector(&column_vector, &matrix, 512, 128, &mut result);
    }

    // 打印结果
    for val in result {
        println!("{:x}", val);
    }
}
