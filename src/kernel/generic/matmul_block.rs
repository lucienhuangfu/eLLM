use super::super::super::init::matmul_params::MatmulParams;
use std::ops::{Add, Mul};

/// 通用微核（与 AVX-512 版本对齐的“广播式”语义）
///
/// 计算一个  (MR = param.a_row_step_micro) × (NR = param.b_row_step_micro)  的 C 子块：
///
///  - A_tile:  [MR x Kc]，行主，行距 = `lda = param.a_row_step_macro`
///  - B_panel: [Kc x NR]，行主打包，行距 = `NR = param.b_row_step_micro`
///  - C_tile:  [MR x NR]，行主，行距 = `ldc = param.b_row_step_macro`
///  - Kc = `param.column_step_macro`
///
/// 注意：这里的 `b` 必须是 **打包面板**（Kc×NR，行主），与 AVX-512 微核一致。
pub fn matmul_block<T>(a: *const T, b: *const T, c: *mut T, param: &MatmulParams)
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    let mr = param.a_row_step_micro;     // 行数
    let nr = param.b_row_step_micro;     // 列数
    let kc = param.column_step_macro;    // K 面板长度

    let lda = param.a_row_step_macro;    // A 行距（元素计）
    let ldc = param.b_row_step_macro;    // C 行距（元素计）

    // B_panel 的行距就是 NR（行主打包；每行存 NR 个元素）
    let ldb_panel = nr;

    for i in 0..mr {
        for j in 0..nr {
            // 读 C[i,j] 作为累加起点（保持 += 语义）
            let mut acc = unsafe { *c.add((i * ldc + j) as usize) };
            for k in 0..kc {
                // A[i,k]
                let a_value = unsafe { *a.add((i * lda + k) as usize) };
                // B_panel[k,j]  —— 行主面板，偏移 = k * NR + j
                let b_value = unsafe { *b.add((k * ldb_panel + j) as usize) };
                acc = acc + (a_value * b_value);
            }
            // 回写 C[i,j]
            unsafe { *c.add((i * ldc + j) as usize) = acc };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f32：MR=3, NR=2, Kc=3，A/B 都是 1 → C 全部等于 Kc
    #[test]
    fn test_f32_panel_microkernel_like() {
        // 参数：MR, NR, Kc
        let mr = 3usize;
        let nr = 2usize;
        let kc = 3usize;

        // 行距：这里构造紧致 tile/panel，取 lda=Kc，ldc=NR
        let lda = kc;
        let ldc = nr;

        // 仅 5 字段（把 lda/ldc/kc 映射进去）
        let param = MatmulParams {
            a_row_step_macro: lda,    // ← lda
            b_row_step_macro: ldc,    // ← ldc
            column_step_macro: kc,    // ← kc
            a_row_step_micro: mr,     // ← MR
            b_row_step_micro: nr,     // ← NR
        };

        // A_tile: [MR x Kc]，全 1
        let a: Vec<f32> = vec![1.0; mr * kc];
        // B_panel: [Kc x NR]，行主，行距=NR，全 1
        let b_panel: Vec<f32> = vec![1.0; kc * nr];
        // C_tile: [MR x NR]，行主，初始 0
        let mut c: Vec<f32> = vec![0.0; mr * ldc];

        // 调用
        matmul_block(a.as_ptr(), b_panel.as_ptr(), c.as_mut_ptr(), &param);

        // 期望全部为 kc
        let expected: Vec<f32> = vec![kc as f32; mr * ldc];
        let tol = 1e-6f32;
        assert!(
            c.iter()
                .zip(expected.iter())
                .all(|(x, y)| (*x - *y).abs() < tol),
            "f32 panel microkernel expected all {}: got {:?}",
            kc,
            &c[..]
        );
    }

    /// f64：MR=3, NR=2, Kc=3，同上
    #[test]
    fn test_f64_panel_microkernel_like() {
        let mr = 3usize;
        let nr = 2usize;
        let kc = 3usize;

        let lda = kc;
        let ldc = nr;

        let param = MatmulParams {
            a_row_step_macro: lda,
            b_row_step_macro: ldc,
            column_step_macro: kc,
            a_row_step_micro: mr,
            b_row_step_micro: nr,
        };

        let a: Vec<f64> = vec![1.0; mr * kc];
        let b_panel: Vec<f64> = vec![1.0; kc * nr];
        let mut c: Vec<f64> = vec![0.0; mr * ldc];

        matmul_block(a.as_ptr(), b_panel.as_ptr(), c.as_mut_ptr(), &param);

        let expected: Vec<f64> = vec![kc as f64; mr * ldc];
        let tol = 1e-9f64;
        assert!(
            c.iter()
                .zip(expected.iter())
                .all(|(x, y)| (*x - *y).abs() < tol),
            "f64 panel microkernel expected all {}: got {:?}",
            kc,
            &c[..]
        );
    }
}