// === kernel/x86_64/f16_512/matmul_block.rs ===
#![allow(non_snake_case)]

use std::arch::x86_64::{_mm512_fmadd_ph, _mm512_loadu_ph, _mm512_set1_ph, _mm512_storeu_ph};
use std::f16;

use crate::init::matmul_params::MatMulParams;

/// 广播式 3x32 FP16 AVX-512 微核：
/// 约定把 (lda/ldc/kc) 映射进 MatMulParams 的 5 字段中：
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
    a: *const f16,        // A tile base: 3xkc
    b_panel: *const f16,  // packed B panel: kc x 32
    c: *mut f16,          // C tile base: 3x32
    param: &MatMulParams,
) {
    // 形状校验
    debug_assert_eq!(param.a_row_step_micro, 3);
    debug_assert_eq!(param.b_row_step_micro, 32);
    debug_assert!(param.column_step_macro > 0);

    // 取 stride/尺寸（元素计）
    let lda = param.a_row_step_macro;    // A 行距
    let ldc = param.b_row_step_macro;    // C 行距
    let kc  = param.column_step_macro;   // K 面板长度
    let b_stride = 32usize;              // B_panel 每行 32

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
    use std::arch::is_x86_feature_detected;
    use std::mem;
    use std::ptr;
    use std::slice;
    use std::f16;

    use crate::kernel::generic::from_f32::FromF32;      // 提供 f16::from_f32
    use crate::memory::allocator::allocate_init;        // 你们自家的对齐 allocator

    #[inline]
    fn f16_from_f32(x: f32) -> f16 {
        f16::from_f32(x)
    }

    /// 手写 f16 -> f32 转换，只用于测试里的误差检查/打印
    #[inline]
    fn f32_from_f16(x: f16) -> f32 {
        let bits: u16 = unsafe { mem::transmute(x) };
        let sign = ((bits & 0x8000) as u32) << 16;
        let exp = (bits & 0x7C00) >> 10;
        let mant = bits & 0x03FF;

        let f_bits: u32 = if exp == 0 {
            if mant == 0 {
                sign
            } else {
                let mut e: i32 = -14;
                let mut m = mant as u32;
                while (m & 0x0400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x03FF;
                let exp_f = (e + 127) as u32;
                sign | (exp_f << 23) | (m << 13)
            }
        } else if exp == 0x1F {
            let exp_f = 0xFFu32;
            sign | (exp_f << 23) | ((mant as u32) << 13)
        } else {
            let exp_f = (exp as i32 - 15 + 127) as u32;
            sign | (exp_f << 23) | ((mant as u32) << 13)
        };

        f32::from_bits(f_bits)
    }

    fn approx_eq(a: f16, b: f16, tol: f32) -> bool {
        let da = f32_from_f16(a);
        let db = f32_from_f16(b);
        (da - db).abs() <= tol
    }

    #[test]
    fn test_broadcast_microkernel_3x32_k64_ones() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("Skipping test_broadcast_microkernel_3x32_k64_ones: avx512fp16 not detected");
            return;
        }

        unsafe {
            const MR: usize = 3;
            const NR: usize = 32;
            const KC: usize = 64;
            const LDA: usize = KC;
            const LDC: usize = NR;

            let param = MatMulParams {
                a_row_step_macro: LDA,
                b_row_step_macro: LDC,
                column_step_macro: KC,
                a_row_step_micro: MR,
                b_row_step_micro: NR,
            };

            // 上层“默认对齐”的版本：用你们的 allocate_init 分配 A/B/C
            let a_ptr = allocate_init::<f16>(MR * LDA, f16_from_f32(1.0));
            let b_ptr = allocate_init::<f16>(KC * NR, f16_from_f32(1.0));
            let c_ptr = allocate_init::<f16>(MR * LDC, f16_from_f32(0.0));

            // 调用 AVX-512 微核
            matmul_block(a_ptr, b_ptr, c_ptr, &param);

            // 读回 C: 每个元素都应该等于 KC（因为 sum_{k} 1*1，初始 C=0）
            let c_slice = slice::from_raw_parts(c_ptr, MR * LDC);
            let expected = f16_from_f32(KC as f32);

            for (i, &v) in c_slice.iter().enumerate() {
                assert!(
                    approx_eq(v, expected, 1e-2),
                    "mismatch at index {}: got {:?} (f32={}), expected {:?} (f32={})",
                    i,
                    v,
                    f32_from_f16(v),
                    expected,
                    f32_from_f16(expected),
                );
            }
        }
    }
}