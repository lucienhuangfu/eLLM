use super::common::*;
use super::*;
// ExpertsMatMulSilu / ExpertsMatMulDown / ExpertsMergeAdd
// weights now NT: [E, I, H] and [E, H, Hmid] respectively
// ============================================================

/// reference: out[e,b,i] = silu(sum_k a[b,k]*w_gate_nt[e,i,k]) * (sum_k a[b,k]*w_up_nt[e,i,k])
fn ref_experts_silu_f32(
    a: &[f16],                  // [B,H]
    w_gate_nt: &[f16],          // [E,I,H] row-major
    w_up_nt: &[f16],            // [E,I,H]
    experts_indicator: &[bool], // [E]
    indice: &[bool],            // [E,B]
    out: &mut [f32],            // [E,B,I]
    b: usize,
    h: usize,
    i: usize,
    e: usize,
) {
    for v in out.iter_mut() {
        *v = 0.0;
    }

    for ex in 0..e {
        if !experts_indicator[ex] {
            continue;
        }
        for bb in 0..b {
            if !indice[ex * b + bb] {
                continue;
            }
            for ii in 0..i {
                let mut g = 0.0f32;
                let mut u = 0.0f32;
                for kk in 0..h {
                    let a_v = a[bb * h + kk] as f32;
                    let wg = w_gate_nt[ex * (i * h) + ii * h + kk] as f32;
                    let wu = w_up_nt[ex * (i * h) + ii * h + kk] as f32;
                    g += a_v * wg;
                    u += a_v * wu;
                }
                out[ex * (b * i) + bb * i + ii] = silu_f32(g) * u;
            }
        }
    }
}

#[inline]
fn slot_of(topk: &[usize], b: usize, ktop: usize, e: usize) -> usize {
    let row = &topk[b * ktop..b * ktop + ktop];
    row.iter().position(|&x| x == e).unwrap_or(0)
}

fn ref_down_f32(
    nonlin: &[f16],   // [E,B,Hmid]
    wdown_nt: &[f16], // [E,H,Hmid] row-major (NT)
    experts_indicator: &[bool],
    indice: &[bool],     // [E,B]
    weight: &[f16],      // [E,B]
    topk: &[usize],      // [B,Ktop]
    out_ref: &mut [f32], // [B,Ktop,H]
    e: usize,
    b: usize,
    hmid: usize,
    h: usize,
    ktop: usize,
) {
    for v in out_ref.iter_mut() {
        *v = 0.0;
    }

    for ex in 0..e {
        if !experts_indicator[ex] {
            continue;
        }
        for bb in 0..b {
            if !indice[ex * b + bb] {
                continue;
            }
            let s = slot_of(topk, bb, ktop, ex);
            let w = f32_from_f16(weight[ex * b + bb]);

            for j in 0..h {
                let mut acc = 0.0f32;
                for kk in 0..hmid {
                    let a = f32_from_f16(nonlin[(ex * b + bb) * hmid + kk]);
                    // NT: [j * hmid + kk]
                    let bj = f32_from_f16(wdown_nt[ex * (h * hmid) + j * hmid + kk]);
                    acc += a * bj;
                }
                out_ref[(bb * ktop + s) * h + j] += w * acc;
            }
        }
    }
}

#[test]
fn test_experts_matmul_silu_f16_tensor_api() {
    if !std::arch::is_x86_feature_detected!("avx512fp16") {
        println!("AVX512FP16 not supported, skipping test.");
        return;
    }
    init_f16_tensor_test_runtime();

    let batch_size = 12;
    let hidden = 64; // H
    let inter = 64; // I
    let num_experts = 2;

    let input =
        Tensor::<f16>::from_mem_pool(vec![batch_size, hidden], "model.layers.0.input".to_string());

    // shape is [E, I, H], raw mem_mgr also [E, I, H] row-major (NT)
    let gate_w =
        Tensor::<f16>::from_mem_pool(vec![num_experts, inter, hidden], "gate.weight".to_string());
    let up_w =
        Tensor::<f16>::from_mem_pool(vec![num_experts, inter, hidden], "up.weight".to_string());

    let b = batch_size;

    let mut experts_box = AlignedBox::allocate_init(num_experts, false);
    let experts_indicator = experts_box.as_mut_ptr();
    std::mem::forget(experts_box);

    let mut indice_box = AlignedBox::allocate_init(num_experts * b, false);
    let indice_ptr = indice_box.as_mut_ptr();
    std::mem::forget(indice_box);

    unsafe {
        *experts_indicator.add(0) = true;
        *experts_indicator.add(1) = false;

        for bb in 0..b {
            *indice_ptr.add(0 * b + bb) = true;
            *indice_ptr.add(b + bb) = false;
        }
    }
    let mut routing_scores = vec![0.0f16; num_experts * b];
    let mut topk_indices = vec![0usize; b];
    for bb in 0..b {
        routing_scores[bb] = 1.0f16;
        topk_indices[bb] = 0;
    }
    let routing = unsafe {
        routing_from_dense(
            num_experts,
            b,
            1,
            indice_ptr,
            routing_scores.as_ptr(),
            topk_indices.as_ptr(),
        )
    };

    // input init
    let mut a = vec![0.0f16; b * hidden];
    for bb in 0..b {
        for kk in 0..hidden {
            a[bb * hidden + kk] = (((bb * 7 + kk * 3) % 19) as f32 * 0.01) as f16;
        }
    }
    unsafe {
        input.data.copy_from_nonoverlapping(a.as_ptr(), a.len());
    }

    // weights init: [E, I, H] row-major
    let per_elems = inter * hidden;
    let mut wg_nt = vec![0.0f16; num_experts * per_elems];
    let mut wu_nt = vec![0.0f16; num_experts * per_elems];

    for e in 0..num_experts {
        for ii in 0..inter {
            for kk in 0..hidden {
                let base_g = ((kk * 5 + ii * 11 + e * 13) % 23) as f32 * 0.01;
                let base_u = ((kk * 9 + ii * 7 + e * 17) % 29) as f32 * 0.01;
                wg_nt[e * per_elems + ii * hidden + kk] = base_g as f16;
                wu_nt[e * per_elems + ii * hidden + kk] = base_u as f16;
            }
        }
    }

    unsafe {
        gate_w
            .data
            .copy_from_nonoverlapping(wg_nt.as_ptr(), wg_nt.len());
        up_w.data
            .copy_from_nonoverlapping(wu_nt.as_ptr(), wu_nt.len());
    }

    let params = MatMulParams {
        a_row_step_macro: 6,
        b_row_step_macro: 64,
        column_step_macro: 64,
        a_row_step_micro: 3,
        b_row_step_micro: 32,
    };

    let out = input.experts_matmul_silu_mul_matmul(
        &gate_w,
        &up_w,
        routing,
        params,
        false,
        "model.layers.0.experts_silu".to_string(),
    );

    assert_eq!(out.shape, vec![num_experts, batch_size, inter]);
    let op = take_single_f16_operator(|op| matches!(op, Operator::ExpertsMatMulSilu(_)));

    let thread_num = avail_threads();
    run_operator_all_threads(&op, b, 0, thread_num);

    let out_len = num_experts * b * inter;
    let out_got = unsafe { std::slice::from_raw_parts(out.data, out_len) };

    // reference for expert0
    for bb in 0..b {
        for ii in 0..inter {
            let mut g = 0.0f32;
            let mut u = 0.0f32;
            for kk in 0..hidden {
                let a_v = a[bb * hidden + kk] as f32;
                let wg_v = wg_nt[0 * per_elems + ii * hidden + kk] as f32;
                let wu_v = wu_nt[0 * per_elems + ii * hidden + kk] as f32;
                g += a_v * wg_v;
                u += a_v * wu_v;
            }
            let exp = silu_f32(g) * u;

            let got = out_got[0 * (b * inter) + bb * inter + ii] as f32;
            assert!(
                (got - exp).abs() < 0.5,
                "Mismatch expert0 bb {} ii {}: got {}, expected {}",
                bb,
                ii,
                got,
                exp
            );
        }
    }

    // expert1 inactive -> 0
    for bb in 0..b {
        for ii in 0..inter {
            let got = out_got[(b * inter) + bb * inter + ii] as f32;
            assert!(
                got.abs() < 1e-3,
                "Inactive expert1 should be ~0, but got {} at bb {} ii {}",
                got,
                bb,
                ii
            );
        }
    }
}

#[test]
fn test_experts_matmul_down_f16_tensor_api() {
    if !std::arch::is_x86_feature_detected!("avx512fp16") {
        println!("AVX512FP16 not supported, skipping test.");
        return;
    }
    init_f16_tensor_test_runtime();

    let batch_size = 12;
    let num_experts = 2;

    let inter = 64; // K (KC=64)
    let hidden = 32; // N (NR=32)
    let num_experts_per_tok = 1;

    let b = batch_size;

    // input to down: [E, seq, batch, inter]
    let x = Tensor::<f16>::from_mem_pool(
        vec![num_experts, batch_size, inter],
        "model.layers.0.experts.silu_out".to_string(),
    );

    // down weights: shape [E, hidden, inter]
    // ✅ NEW contract: B is already NT (N×K) row-major in mem_mgr per expert:
    // w_nt[j * inter + kk]
    let down_w = Tensor::<f16>::from_mem_pool(
        vec![num_experts, hidden, inter],
        "model.layers.0.down.weight".to_string(),
    );

    let mut experts_box = AlignedBox::allocate_init(num_experts, false);
    let experts_indicator = experts_box.as_mut_ptr();
    std::mem::forget(experts_box);

    let mut indice_box = AlignedBox::allocate_init(num_experts * b, false);
    let indice_ptr = indice_box.as_mut_ptr();
    std::mem::forget(indice_box);

    let mut weight_box = AlignedBox::allocate_init(num_experts * b, 0.0f16);
    let weight_ptr = weight_box.as_mut_ptr();
    std::mem::forget(weight_box);

    let mut topk_indices_box = AlignedBox::allocate_init(b * num_experts_per_tok, 0usize);
    let topk_indices_ptr = topk_indices_box.as_mut_ptr();
    std::mem::forget(topk_indices_box);

    unsafe {
        *experts_indicator.add(0) = true;
        *experts_indicator.add(1) = false;

        for t in 0..b {
            *indice_ptr.add(0 * b + t) = true;
            *indice_ptr.add(b + t) = false;

            *weight_ptr.add(0 * b + t) = 1.0f16;
            *weight_ptr.add(b + t) = 0.0f16;

            *topk_indices_ptr.add(t) = 0usize;
        }
    }

    // init X: expert0 pattern, expert1 zeros
    let mut x_e0 = vec![0.0f16; b * inter];
    for t in 0..b {
        for kk in 0..inter {
            x_e0[t * inter + kk] = (((t * 7 + kk * 3) % 19) as f32 * 0.01) as f16;
        }
    }
    unsafe {
        x.data
            .add(0 * (b * inter))
            .copy_from_nonoverlapping(x_e0.as_ptr(), x_e0.len());
        for i in 0..(b * inter) {
            *x.data.add((b * inter) + i) = 0.0f16;
        }
    }

    // init W_down:
    // ✅ NEW: per expert is NT (N×K) = hidden × inter row-major:
    // w_e0[j * inter + kk]
    let per_w = inter * hidden;
    let mut w_e0 = vec![0.0f16; per_w];
    let mut w_e1 = vec![0.0f16; per_w];

    for j in 0..hidden {
        for kk in 0..inter {
            // 跟以前一样的 deterministic pattern，只是存储索引变了
            w_e0[j * inter + kk] = (((kk * 5 + j * 11) % 23) as f32 * 0.01) as f16;
            w_e1[j * inter + kk] = (((kk * 3 + j * 7) % 29) as f32 * 0.01) as f16;
        }
    }

    unsafe {
        down_w
            .data
            .add(0 * per_w)
            .copy_from_nonoverlapping(w_e0.as_ptr(), per_w);
        down_w
            .data
            .add(per_w)
            .copy_from_nonoverlapping(w_e1.as_ptr(), per_w);
    }

    let params = MatMulParams {
        a_row_step_macro: 6,
        b_row_step_macro: 64,
        column_step_macro: 64,
        a_row_step_micro: 3,
        b_row_step_micro: 32,
    };

    let out = x.experts_matmul_mul(
        &down_w,
        unsafe {
            routing_from_dense(
                num_experts,
                b,
                num_experts_per_tok,
                indice_ptr,
                weight_ptr,
                topk_indices_ptr,
            )
        },
        num_experts_per_tok,
        params,
        false,
        "model.layers.0.experts_down".to_string(),
    );

    assert_eq!(out.shape, vec![batch_size, num_experts_per_tok, hidden]);
    let op = take_single_f16_operator(|op| matches!(op, Operator::ExpertsMatMulDown(_)));

    // ✅ IMPORTANT: down does out += acc * factor, so zero out first
    let out_len = b * num_experts_per_tok * hidden;
    unsafe {
        for i in 0..out_len {
            *out.data.add(i) = 0.0f16;
        }
    }

    let thread_num = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    run_operator_all_threads(&op, b, 0, thread_num);

    // verify reference:
    // out[t, 0, j] = sum_k x_e0[t,k] * w_e0_nt[j,k]
    let out_got = unsafe { std::slice::from_raw_parts(out.data, out_len) };
    for t in 0..b {
        for j in 0..hidden {
            let mut sum = 0.0f32;
            for kk in 0..inter {
                let x_v = x_e0[t * inter + kk] as f32;
                let w_v = w_e0[j * inter + kk] as f32; // ✅ NT indexing
                sum += x_v * w_v;
            }
            let got = out_got[t * hidden + j] as f32;
            assert!(
                (got - sum).abs() < 0.5,
                "Down mismatch token {} col {}: got {}, expected {}",
                t,
                j,
                got,
                sum
            );
        }
    }
}

#[test]
fn test_experts_merge_add_f16_tensor_api_k2_slot1_zero() {
    if !std::arch::is_x86_feature_detected!("avx512fp16") {
        println!("AVX512FP16 not supported, skipping test.");
        return;
    }
    init_f16_tensor_test_runtime();

    let batch_size = 12;
    let num_tokens = batch_size;

    let num_experts = 2; // 仅用于 reset gating（我们这里 reset_gating=false）
    let k = 2usize; // num_experts_per_token == K
    let hidden = 64usize;

    // input ptr layout: [num_tokens, K, H]
    let input = Tensor::<f16>::from_mem_pool(
        vec![batch_size, k, hidden],
        "model.layers.0.moe.down_out".to_string(),
    );

    let residual = Tensor::<f16>::from_mem_pool(
        vec![batch_size, hidden],
        "model.layers.0.residual".to_string(),
    );

    // routing buffers（reset_gating=false 不会用来选择，只会在 reset_gating=true 时清零）
    let mut experts_box = AlignedBox::allocate_init(num_experts, false);
    let experts_indicator = experts_box.as_mut_ptr();
    std::mem::forget(experts_box);

    let mut indice_box = AlignedBox::allocate_init(num_experts * num_tokens, false);
    let indice_ptr = indice_box.as_mut_ptr();
    std::mem::forget(indice_box);

    unsafe {
        *experts_indicator.add(0) = true;
        *experts_indicator.add(1) = true;
        for t in 0..num_tokens {
            *indice_ptr.add(0 * num_tokens + t) = true;
            *indice_ptr.add(num_tokens + t) = true;
        }
    }
    let mut routing_scores = vec![0.0f16; num_experts * num_tokens];
    let mut topk_indices = vec![0usize; num_tokens * k];
    for t in 0..num_tokens {
        routing_scores[t] = 1.0f16;
        routing_scores[num_tokens + t] = 1.0f16;
        topk_indices[t * k] = 0;
        topk_indices[t * k + 1] = 1;
    }
    let routing = unsafe {
        routing_from_dense(
            num_experts,
            num_tokens,
            k,
            indice_ptr,
            routing_scores.as_ptr(),
            topk_indices.as_ptr(),
        )
    };

    // init residual
    let mut r = vec![0.0f16; num_tokens * hidden];
    for t in 0..num_tokens {
        for h in 0..hidden {
            r[t * hidden + h] = (((t * 2 + h * 5) % 17) as f32 * 0.01) as f16;
        }
    }
    unsafe {
        residual.data.copy_from_nonoverlapping(r.as_ptr(), r.len());
    }

    // init input: slot0 = pattern, slot1 = 0
    let mut slot0 = vec![0.0f16; num_tokens * hidden];
    for t in 0..num_tokens {
        for h in 0..hidden {
            slot0[t * hidden + h] = (((t * 7 + h * 3) % 19) as f32 * 0.01) as f16;
        }
    }

    unsafe {
        // 全清零
        let total = num_tokens * k * hidden;
        for i in 0..total {
            *input.data.add(i) = 0.0f16;
        }
        // 写 slot0
        for t in 0..num_tokens {
            let base = t * (k * hidden);
            input
                .data
                .add(base + 0 * hidden)
                .copy_from_nonoverlapping(slot0.as_ptr().add(t * hidden), hidden);
        }
    }

    // build via Tensor API
    let out = input.experts_merge_add(
        &residual,
        routing,
        false,
        "model.layers.0.experts_merge_add".to_string(),
    );

    assert_eq!(out.shape, vec![batch_size, hidden]);
    let op = take_single_f16_operator(|op| matches!(op, Operator::ExpertsMergeAdd(_)));
    match &op {
        Operator::ExpertsMergeAdd(operator) => {
            assert_eq!(operator.num_experts, num_experts);
            assert_eq!(operator.batch_size, batch_size);
        }
        _ => unreachable!(),
    }

    // run
    let thread_num = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    run_operator_all_threads(&op, num_tokens, 0, thread_num);

    // verify: out = residual + slot0 + slot1(0)
    let out_len = num_tokens * hidden;
    let out_got = unsafe { std::slice::from_raw_parts(out.data, out_len) };

    for t in 0..num_tokens {
        for h in 0..hidden {
            let exp = (r[t * hidden + h] as f32) + (slot0[t * hidden + h] as f32);
            let got = out_got[t * hidden + h] as f32;
            assert!(
                (got - exp).abs() < 1e-3,
                "MergeAdd mismatch token {} h {}: got {}, expected {}",
                t,
                h,
                got,
                exp
            );
        }
    }
}
