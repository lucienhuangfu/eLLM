// === kernel/x86_64/f16_512/matmul_block.rs ===
#![allow(non_snake_case)]

use std::arch::x86_64::{_mm512_fmadd_ph, _mm512_loadu_ph, _mm512_set1_ph, _mm512_storeu_ph};
use std::f16;

use crate::init::matmul_params::MatMulParams;

/// 广播式 3x32 FP16 AVX-512 微核：
/// 约定把 (lda/ldc/kc) 映射进 matmulParams 的 5 字段中：
/// - a_row_step_micro = MR (=3)
/// - b_row_step_micro = NR (=32)
/// - column_step_macro = kc
/// - a_row_step_macro = lda
/// - b_row_step_macro = ldc
///
/// A_tile: 3 x kc（行主，行距=lda）
/// B_panel: kc x 32（行主打包，每行 32 连续）
/// C_tile: 3 x 32（行主，行距=ldc）
#[inline(always)]
pub unsafe fn matmul_block(
    a: *const f16,       // A tile base: 3xkc
    b_panel: *const f16, // packed B panel: kc x 32
    c: *mut f16,         // C tile base: 3x32
    param: &MatMulParams,
) {
    // 形状校验
    debug_assert_eq!(param.a_row_step_micro, 3);
    debug_assert_eq!(param.b_row_step_micro, 32);
    debug_assert!(param.column_step_macro > 0);

    // 取 stride/尺寸（元素计）
    let lda = param.a_row_step_macro; // A 行距
    let ldc = param.b_row_step_macro; // C 行距
    let kc = param.column_step_macro; // K 面板长度
    let b_stride = 32usize; // B_panel 每行 32

    // A 三行基址
    let a0 = a;
    let a1 = a.add(lda);
    let a2 = a.add(2 * lda);

    // 读入 C 累加器（unaligned load，方便上层随意对齐）
    let mut c_row0 = _mm512_loadu_ph(c.add(0 * ldc));
    let mut c_row1 = _mm512_loadu_ph(c.add(1 * ldc));
    let mut c_row2 = _mm512_loadu_ph(c.add(2 * ldc));

    // 主循环：按 k 广播 A 的标量 * B_panel 的向量做 FMA
    for k in 0..kc {
        let bvec = _mm512_loadu_ph(b_panel.add(k * b_stride));
        let a0b = _mm512_set1_ph(*a0.add(k));
        let a1b = _mm512_set1_ph(*a1.add(k));
        let a2b = _mm512_set1_ph(*a2.add(k));

        c_row0 = _mm512_fmadd_ph(a0b, bvec, c_row0);
        c_row1 = _mm512_fmadd_ph(a1b, bvec, c_row1);
        c_row2 = _mm512_fmadd_ph(a2b, bvec, c_row2);
    }

    // 写回
    _mm512_storeu_ph(c.add(0 * ldc), c_row0);
    _mm512_storeu_ph(c.add(1 * ldc), c_row1);
    _mm512_storeu_ph(c.add(2 * ldc), c_row2);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_matmul_block_kernel() {
        if !is_x86_feature_detected!("avx512fp16") {
            println!("Skipping avx512fp16 test on unsupported hardware");
            return;
        }

        let kc = 64;
        let lda = 128;
        let ldc = 64;

        let params = MatMulParams {
            a_row_step_micro: 3,
            b_row_step_micro: 32,
            column_step_macro: kc,
            a_row_step_macro: lda,
            b_row_step_macro: ldc,
        };

        let mut a_vec = vec![0.0f16; 3 * lda];
        let mut b_vec = vec![0.0f16; kc * 32];
        let mut c_vec = vec![0.0f16; 3 * ldc];

        // 初始化 A
        for i in 0..3 {
            for k in 0..kc {
                let val = (i + k) as f32 * 0.1;
                a_vec[i * lda + k] = val as f16;
            }
        }

        // 初始化 B
        for k in 0..kc {
            for j in 0..32 {
                let val = (k + j) as f32 * 0.1;
                b_vec[k * 32 + j] = val as f16;
            }
        }

        // 初始化 C (累加基底)
        for i in 0..3 {
            for j in 0..32 {
                c_vec[i * ldc + j] = 1.0f16;
            }
        }

        unsafe {
            matmul_block(a_vec.as_ptr(), b_vec.as_ptr(), c_vec.as_mut_ptr(), &params);
        }

        // 验证结果
        for i in 0..3 {
            for j in 0..32 {
                let mut sum = 1.0f32;
                for k in 0..kc {
                    let a_val = a_vec[i * lda + k] as f32;
                    let b_val = b_vec[k * 32 + j] as f32;
                    sum += a_val * b_val;
                }
                let got = c_vec[i * ldc + j] as f32;
                assert_abs_diff_eq!(got, sum, epsilon = 2.0);
            }
        }
    }

    #[test]
    fn test_matmul_block_large_scale() {
        if !is_x86_feature_detected!("avx512fp16") {
            println!("Skipping avx512fp16 test on unsupported hardware");
            return;
        }

        // 模拟用户场景：Left 128x2048, Right 2048x2048
        // 我们测试其中一个 3x32 的 block 计算
        let m = 128;
        let k_dim = 2048;
        let n = 2048;

        let kc = k_dim; // K 维度全长
        let lda = k_dim; // A 的行距 (128x2048)
        let ldc = n; // C 的行距 (128x2048)

        let params = MatMulParams {
            a_row_step_micro: 3,
            b_row_step_micro: 32,
            column_step_macro: kc,
            a_row_step_macro: lda,
            b_row_step_macro: ldc,
        };

        let mut a_vec = vec![0.0f16; m * k_dim];
        // B 需要是 packed panel: K x 32
        let mut b_packed = vec![0.0f16; k_dim * 32];
        let mut c_vec = vec![0.0f16; m * n];

        // 初始化 A (使用较小数值防止溢出/精度问题)
        for i in 0..m {
            for k in 0..k_dim {
                let val = ((i + k) % 17) as f32 * 0.01;
                a_vec[i * lda + k] = val as f16;
            }
        }

        // 初始化 B packed
        for k in 0..k_dim {
            for j in 0..32 {
                let val = ((k + j) % 19) as f32 * 0.01;
                b_packed[k * 32 + j] = val as f16;
            }
        }

        // 初始化 C
        for i in 0..m {
            for j in 0..n {
                c_vec[i * ldc + j] = 0.0f16;
            }
        }

        // 计算左上角 3x32 块
        unsafe {
            matmul_block(
                a_vec.as_ptr(),
                b_packed.as_ptr(),
                c_vec.as_mut_ptr(),
                &params,
            );
        }

        // 验证结果
        for i in 0..3 {
            for j in 0..32 {
                let mut sum = 0.0f32;
                for k in 0..kc {
                    let a_val = a_vec[i * lda + k] as f32;
                    let b_val = b_packed[k * 32 + j] as f32;
                    sum += a_val * b_val;
                }
                let got = c_vec[i * ldc + j] as f32;
                // K=2048 累加误差容忍度
                assert_abs_diff_eq!(got, sum, epsilon = 1.0);
            }
        }
    }
}
