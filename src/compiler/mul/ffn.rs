#[derive(Clone)]
pub struct FusedFFN<T> {
    // 输入/输出
    ptr1:       crate::init::send_sync_ptr::ConstPtr<T>, // A[S×M×K]
    output_ptr: crate::init::send_sync_ptr::MutPtr<T>,   // C[S×M×K]

    // 预转置权重缓存与只读指针（保持你的风格）
    wdown_nt_buf: Vec<T>,  wdown_nt: crate::init::send_sync_ptr::ConstPtr<T>, // [N×K]
    wgate_nt_buf: Vec<T>,  wgate_nt: crate::init::send_sync_ptr::ConstPtr<T>, // [N×K]
    wup_nt_buf:   Vec<T>,  wup_nt:   crate::init::send_sync_ptr::ConstPtr<T>, // [K×N]

    pub params: crate::init::matmul_params::MatMulParams, // a_row=M, b_row=N, column=K
    _marker: core::marker::PhantomData<T>,
}

impl<T> FusedFFN<T>
where
    T: Copy + Default + core::ops::Add<Output = T> + core::ops::Mul<Output = T>,
{
    pub unsafe fn new(
        ptr1: *const T,
        w_down_ptr: *const T,   // 原始 [K×N]
        w_gate_ptr: *const T,   // 原始 [K×N]
        w_up_ptr:   *const T,   // 原始 [N×K]
        output_ptr: *mut T,

        a_row: usize, b_row: usize, column: usize, // M, N, K
        a_row_step_macro: usize,
        b_row_step_macro: usize,
        column_step_macro: usize,
        a_row_step_micro: usize,
        b_row_step_micro: usize,
    ) -> Self {
        let m = a_row; let n = b_row; let k = column;

        // 预转置：W_down[K×N] → [N×K]
        let mut wdown_nt_buf = vec![T::default(); n*k];
        for kk in 0..k {
            let src = w_down_ptr.add(kk * n);
            for jj in 0..n {
                *wdown_nt_buf.as_mut_ptr().add(jj*k + kk) = *src.add(jj);
            }
        }
        // 预转置：W_gate[K×N] → [N×K]
        let mut wgate_nt_buf = vec![T::default(); n*k];
        for kk in 0..k {
            let src = w_gate_ptr.add(kk * n);
            for jj in 0..n {
                *wgate_nt_buf.as_mut_ptr().add(jj*k + kk) = *src.add(jj);
            }
        }
        // 预转置：W_up[N×K] → [K×N]
        let mut wup_nt_buf = vec![T::default(); k*n];
        for nn in 0..n {
            let src = w_up_ptr.add(nn * k);
            for jj in 0..k {
                *wup_nt_buf.as_mut_ptr().add(jj*n + nn) = *src.add(jj);
            }
        }

        let params = crate::init::matmul_params::MatMulParams {
            a_row: m, b_row: n, column: k,
            a_row_step_macro,
            b_row_step_macro,
            column_step_macro,
            a_row_step_micro,
            b_row_step_micro,
        };

        Self {
            ptr1: crate::init::send_sync_ptr::ConstPtr { ptr: ptr1 },
            output_ptr: crate::init::send_sync_ptr::MutPtr { ptr: output_ptr },
            wdown_nt: crate::init::send_sync_ptr::ConstPtr { ptr: wdown_nt_buf.as_ptr() },
            wdown_nt_buf,
            wgate_nt: crate::init::send_sync_ptr::ConstPtr { ptr: wgate_nt_buf.as_ptr() },
            wgate_nt_buf,
            wup_nt: crate::init::send_sync_ptr::ConstPtr { ptr: wup_nt_buf.as_ptr() },
            wup_nt_buf,
            params,
            _marker: core::marker::PhantomData,
        }
    }

    pub unsafe fn run(
        &self,
        position_index: usize,    // s_begin
        position_interval: usize, // 这次处理的序列帧数
        batch_size: usize,        // M
        cpu_num: usize,
        thread_id: usize,
    )
    where
        Self: FusedKernels<T>, // 关键：要求实现 compute 接口（默认或特化）
    {
        let m = batch_size;
        let n = self.params.b_row;   // 中间维
        let k = self.params.column;  // 归约维/输入维

        let mb = self.params.a_row_step_macro.max(1);
        let nb = self.params.b_row_step_macro.max(1);
        let kc = self.params.column_step_macro.max(1);
        let mr = self.params.a_row_step_micro.max(1); // 3
        let nr = self.params.b_row_step_micro.max(1); // 32

        // 基址/行距
        let a_base = self.ptr1.ptr;
        let c_base = self.output_ptr.ptr;
        let lda_a = k; // A 每行跨度
        let ldc_c = k; // C 每行跨度（最终输出列 K）

        // 序列 stride
        let s_begin = position_index;
        let s_end   = position_index + position_interval;
        let a_seq_stride = m * k;
        let c_seq_stride = m * k;

        // 任务切分：S × tiles_m × tiles_n（N 是中间维）
        let tiles_m = (m + mb - 1) / mb;
        let tiles_n = (n + nb - 1) / nb;
        let tiles_sn = (s_end - s_begin) * tiles_m * tiles_n;

        if let Some((tb, te)) = crate::compiler::assign::assign(tiles_sn, cpu_num, thread_id) {
            // 线程私有面板
            let mut down_panel = vec![T::default(); kc * nr];
            let mut gate_panel = vec![T::default(); kc * nr];
            let mut up_panel   = vec![T::default(); nb * nr];

            // pack 函数（保持 T 泛型）
            #[inline(always)]
            unsafe fn pack_nk_to_kcnr<T: Copy>(
                w_nt: *const T, n: usize, k: usize,
                n0: usize, k0: usize, kc: usize, nr: usize,
                out: *mut T,
            ) {
                for p in 0..kc {
                    let dst = out.add(p * nr);
                    let src_col = k0 + p;
                    for lane in 0..nr {
                        let j = n0 + lane;
                        *dst.add(lane) = *w_nt.add(j * k + src_col);
                    }
                }
            }
            #[inline(always)]
            unsafe fn pack_kn_to_nbnr<T: Copy>(
                wup_nt: *const T, k: usize, n: usize,
                n0: usize, k0: usize, nb_blk: usize,
                out: *mut T,
            ) {
                for p in 0..nb_blk {
                    let dst = out.add(p * 32);
                    for lane in 0..32 {
                        let kk = k0 + lane;
                        *dst.add(lane) = *wup_nt.add(kk * n + (n0 + p));
                    }
                }
            }

            for t in tb..te {
                let s_rel = t / (tiles_m * tiles_n);
                let rem   = t % (tiles_m * tiles_n);
                let tm = rem / tiles_n;
                let tn = rem % tiles_n;

                let s  = s_begin + s_rel;
                let m0 = tm * mb;
                let n0 = tn * nb;

                let m_blk = (m - m0).min(mb);
                let n_blk = (n - n0).min(nb);

                let a_base_s = a_base.add(s * a_seq_stride);
                let c_base_s = c_base.add(s * c_seq_stride);

                // === 阶段1/2：down+gate（K 归约到 N 块） ===
                let ldn = n; // 中间缓冲行距
                let mut acc_down = vec![T::default(); m_blk * ldn];
                let mut acc_gate = vec![T::default(); m_blk * ldn];

                let mut k0 = 0;
                while k0 < k {
                    let mut nt = 0;
                    while nt < n_blk {
                        pack_nk_to_kcnr(self.wdown_nt.ptr, n, k, n0 + nt, k0, kc, nr, down_panel.as_mut_ptr());
                        pack_nk_to_kcnr(self.wgate_nt.ptr, n, k, n0 + nt, k0, kc, nr, gate_panel.as_mut_ptr());

                        let mut mi = 0;
                        while mi < m_blk {
                            let a_tile = a_base_s.add((m0 + mi) * lda_a + k0);
                            let d_tile = acc_down.as_mut_ptr().add(mi * ldn + nt);
                            let g_tile = acc_gate.as_mut_ptr().add(mi * ldn + nt);

                            let p12 = crate::init::matmul_params::MatMulParams {
                                a_row: 3, b_row: ldn, column: lda_a,
                                a_row_step_macro: 3, b_row_step_macro: 32, column_step_macro: kc,
                                a_row_step_micro: 3, b_row_step_micro: 32,
                            };
                            // —— 关键：调 trait，具体用哪个内核由特化决定 —— //
                            self.compute_dual_update_3x32(a_tile, down_panel.as_ptr(), gate_panel.as_ptr(), d_tile, g_tile, &p12);

                            mi += mr;
                        }
                        nt += nr;
                    }
                    k0 += kc;
                }

                // === 阶段2：SiLU ⊙（就地） ===
                let mut nt = 0;
                while nt < n_blk {
                    let mut mi = 0;
                    while mi < m_blk {
                        let d_tile = acc_down.as_mut_ptr().add(mi * ldn + nt);
                        let g_tile = acc_gate.as_ptr().add(mi * ldn + nt);
                        self.compute_silu_hadamard_3x32_inplace(d_tile, g_tile, ldn);
                        mi += mr;
                    }
                    nt += nr;
                }

                // === 阶段3：up，到 K 轴 ===
                let tiles_nk = (k + nb - 1) / nb;
                for tnk in 0..tiles_nk {
                    let kc0 = tnk * nb;
                    let k_blk = (k - kc0).min(nb);

                    let mut lanes = 0usize;
                    while lanes < k_blk {
                        // 最小原型：按 32 对齐
                        pack_kn_to_nbnr(self.wup_nt.ptr, k, n, n0, kc0 + lanes, n_blk, up_panel.as_mut_ptr());

                        let mut mi = 0;
                        while mi < m_blk {
                            let n_tile = acc_down.as_ptr().add(mi * ldn + 0);
                            let c_tile = c_base_s.add((m0 + mi) * ldc_c + (kc0 + lanes));

                            let p_up = crate::init::matmul_params::MatMulParams {
                                a_row: 3, b_row: ldc_c, column: ldn,
                                a_row_step_macro: 3, b_row_step_macro: 32, column_step_macro: n_blk,
                                a_row_step_micro: 3, b_row_step_micro: 32,
                            };
                            self.compute_gemm_3x32(n_tile, up_panel.as_ptr(), c_tile, &p_up);

                            mi += mr;
                        }

                        lanes += 32;
                    }
                }
            }
        }
    }
}