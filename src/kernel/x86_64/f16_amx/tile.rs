#![allow(non_snake_case)]

use std::arch::x86_64::{
    _tile_dpfp16ps, _tile_loadconfig, _tile_loadd, _tile_release, _tile_stored, _tile_zero,
};
use std::f16;

pub const AMX_MR: usize = 3;
pub const AMX_NR: usize = 16;
const AMX_KR: usize = 32;
const AMX_TILE_ROWS: usize = 16;

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct TileConfig {
    palette: u8,
    start_row: u8,
    reserved_a0: [u8; 14],
    colsb: [u16; 8],
    reserved_b0: [u16; 8],
    rows: [u8; 8],
    reserved_c0: [u8; 8],
}

impl Default for TileConfig {
    fn default() -> Self {
        Self {
            palette: 0,
            start_row: 0,
            reserved_a0: [0; 14],
            colsb: [0; 8],
            reserved_b0: [0; 8],
            rows: [0; 8],
            reserved_c0: [0; 8],
        }
    }
}

impl TileConfig {
    #[inline(always)]
    fn fp16_dot_product() -> Self {
        let mut cfg = Self::default();
        cfg.palette = 1;

        for tile in 0..=2 {
            cfg.colsb[tile] = 64;
            cfg.rows[tile] = AMX_TILE_ROWS as u8;
        }

        cfg
    }

    #[inline(always)]
    fn as_ptr(&self) -> *const u8 {
        self as *const Self as *const u8
    }
}

#[cfg(target_os = "linux")]
unsafe fn request_amx_permission() -> bool {
    const SYS_ARCH_PRCTL: isize = 158;
    const ARCH_GET_XCOMP_PERM: usize = 0x1022;
    const ARCH_REQ_XCOMP_PERM: usize = 0x1023;
    const XFEATURE_XTILEDATA: usize = 18;

    unsafe extern "C" {
        fn syscall(num: isize, ...) -> isize;
    }

    let mut xfeatures = 0usize;
    if unsafe {
        syscall(
            SYS_ARCH_PRCTL,
            ARCH_GET_XCOMP_PERM,
            &mut xfeatures as *mut usize,
        )
    } != 0
    {
        return false;
    }

    match 0b11 & (xfeatures >> 17) {
        3 => true,
        1 => unsafe { syscall(SYS_ARCH_PRCTL, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0 },
        _ => false,
    }
}

#[cfg(not(target_os = "linux"))]
unsafe fn request_amx_permission() -> bool {
    true
}

#[target_feature(enable = "amx-tile")]
#[inline]
pub unsafe fn ensure_amx_ready() {
    assert!(
        unsafe { request_amx_permission() },
        "AMX tile data permission is not available for this thread"
    );
}

#[target_feature(enable = "amx-tile,amx-fp16")]
pub unsafe fn gemm_3x16_to_f32(
    a: *const f16,
    b_panel: *const f16,
    out: *mut f32,
    lda: usize,
    b_stride: usize,
    kc: usize,
) {
    let out_slice = unsafe { core::slice::from_raw_parts_mut(out, AMX_MR * AMX_NR) };
    out_slice.fill(0.0);

    let mut a_tile = [[0.0f16; AMX_KR]; AMX_TILE_ROWS];
    let mut b_tile = [[0.0f16; AMX_KR]; AMX_TILE_ROWS];
    let mut partial = [[0.0f32; AMX_NR]; AMX_TILE_ROWS];
    let mut k0 = 0usize;

    while k0 < kc {
        let kb = (kc - k0).min(AMX_KR);
        let cfg = TileConfig::fp16_dot_product();

        a_tile.fill([0.0f16; AMX_KR]);
        b_tile.fill([0.0f16; AMX_KR]);

        for r in 0..AMX_MR {
            for k in 0..kb {
                a_tile[r][k] = unsafe { *a.add(r * lda + k0 + k) };
            }
        }

        // TDPFP16PS consumes B in a pair-interleaved layout:
        // B[pair_k][2 * n]     = original B[2 * pair_k][n]
        // B[pair_k][2 * n + 1] = original B[2 * pair_k + 1][n]
        // Existing f16_512 panels are K x N, so pack the current 16-column
        // half into that AMX layout and zero-pad odd/tail K.
        for n in 0..AMX_NR {
            for pair_k in 0..((kb + 1) / 2) {
                let k_even = 2 * pair_k;
                b_tile[pair_k][2 * n] = unsafe { *b_panel.add((k0 + k_even) * b_stride + n) };
                if k_even + 1 < kb {
                    b_tile[pair_k][2 * n + 1] =
                        unsafe { *b_panel.add((k0 + k_even + 1) * b_stride + n) };
                }
            }
        }

        unsafe {
            _tile_loadconfig(cfg.as_ptr());
            _tile_zero::<0>();
            _tile_loadd::<1>(a_tile.as_ptr().cast(), 64);
            _tile_loadd::<2>(b_tile.as_ptr().cast(), 64);
            _tile_dpfp16ps::<0, 1, 2>();
            _tile_stored::<0>(partial.as_mut_ptr().cast(), AMX_NR * size_of::<f32>());
            _tile_release();
        }

        for r in 0..AMX_MR {
            for n in 0..AMX_NR {
                out_slice[r * AMX_NR + n] += partial[r][n];
            }
        }

        k0 += kb;
    }
}
