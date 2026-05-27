#![feature(f16)]

use ellm::operators::transform::RMSMap;
use std::io::Read;
use std::path::Path;

fn load_npy_f32<P: AsRef<Path>>(path: P) -> Vec<f32> {
    let mut buf = vec![];
    std::fs::File::open(path)
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();
    let data: npy::NpyData<f32> = npy::NpyData::from_bytes(&buf).unwrap();
    data.to_vec()
}

fn write_npy_f32<P: AsRef<Path>>(path: P, data: &[f32]) {
    npy::to_file(path.as_ref(), data.iter().copied()).unwrap();
}

fn f32_to_f16(data: &[f32]) -> Vec<f16> {
    data.iter().map(|&x| x as f16).collect()
}

fn f16_to_f32(data: &[f16]) -> Vec<f32> {
    data.iter().map(|&x| x as f32).collect()
}

fn run_test(
    input_f32: &[f32],
    weight_f32: &[f32],
    hidden_size: usize,
    eps: f16,
    num_rows: usize,
    decode_only: bool,
) -> Vec<f32> {
    let total_size = num_rows * hidden_size;
    let input_f16 = f32_to_f16(input_f32);
    let weight_f16 = f32_to_f16(weight_f32);
    let mut output_f16 = vec![0.0f16; total_size];

    let operator = RMSMap::<f16>::new(
        input_f16.as_ptr(),
        weight_f16.as_ptr(),
        output_f16.as_mut_ptr(),
        hidden_size,
        eps,
        decode_only,
    );

    // Run single-threaded
    operator.run(num_rows, 0, 1, 0);

    f16_to_f32(&output_f16)
}

fn compare_and_report(
    name: &str,
    output: &[f32],
    expected: &[f32],
    dump_dir: &Path,
) -> bool {
    let mut max_err = 0.0f32;
    let mut sum_err = 0.0f64;
    let mut dot = 0.0f64;
    let mut norm_rust = 0.0f64;
    let mut norm_py = 0.0f64;
    let mut first_bad_idx = None;

    for i in 0..output.len() {
        let diff = (output[i] - expected[i]).abs();
        if diff > max_err {
            max_err = diff;
        }
        sum_err += diff as f64;
        let r = output[i] as f64;
        let p = expected[i] as f64;
        dot += r * p;
        norm_rust += r * r;
        norm_py += p * p;
        if diff > 1e-3 && first_bad_idx.is_none() {
            first_bad_idx = Some(i);
        }
    }

    let mean_err = sum_err / output.len() as f64;
    let cosine = if norm_rust > 0.0 && norm_py > 0.0 {
        dot / (norm_rust.sqrt() * norm_py.sqrt())
    } else {
        1.0
    };

    write_npy_f32(dump_dir.join(format!("rust_{}.npy", name)), output);

    let passed = max_err < 1e-3 && cosine > 0.9999;
    println!(
        "  {}: max_err={:.2e} mean_err={:.2e} cosine={:.10} {}",
        name,
        max_err,
        mean_err,
        cosine,
        if passed { "PASS" } else { "FAIL" }
    );

    if let Some(idx) = first_bad_idx {
        let row = idx / 1024;
        let col = idx % 1024;
        println!(
            "    first mismatch at [{row}][{col}]: rust={:?} python={:?}",
            output[idx], expected[idx]
        );
    }

    passed
}

fn main() {
    let dump_dir = Path::new("alignment/rms_map/dump");
    let hidden_size = 1024;
    let num_rows = 15;
    let eps: f16 = 1e-6f32 as f16;

    println!("=== RMSMap Alignment Test ===");

    // Test 1: Sequential inputs with weight=1
    println!("\n--- Test 1: Sequential, weight=1 ---");
    let input_seq = load_npy_f32(dump_dir.join("input_seq.npy"));
    let weight_ones = load_npy_f32(dump_dir.join("weight_ones.npy"));
    let expected_seq = load_npy_f32(dump_dir.join("expected_seq.npy"));
    let output_seq = run_test(&input_seq, &weight_ones, hidden_size, eps, num_rows, false);
    compare_and_report("seq", &output_seq, &expected_seq, dump_dir);

    // Test 2: Zeros
    println!("\n--- Test 2: Zeros ---");
    let input_zeros = load_npy_f32(dump_dir.join("input_zeros.npy"));
    let expected_zeros = load_npy_f32(dump_dir.join("expected_zeros.npy"));
    let output_zeros = run_test(&input_zeros, &weight_ones, hidden_size, eps, num_rows, false);
    compare_and_report("zeros", &output_zeros, &expected_zeros, dump_dir);

    // Test 3: Ones
    println!("\n--- Test 3: Ones ---");
    let input_ones = load_npy_f32(dump_dir.join("input_ones.npy"));
    let expected_ones = load_npy_f32(dump_dir.join("expected_ones.npy"));
    let output_ones = run_test(&input_ones, &weight_ones, hidden_size, eps, num_rows, false);
    compare_and_report("ones", &output_ones, &expected_ones, dump_dir);

    // Test 4: Random with weight=1
    println!("\n--- Test 4: Random, weight=1 ---");
    let input_rand = load_npy_f32(dump_dir.join("input_rand.npy"));
    let expected_rand = load_npy_f32(dump_dir.join("expected_rand.npy"));
    let output_rand = run_test(&input_rand, &weight_ones, hidden_size, eps, num_rows, false);
    compare_and_report("rand", &output_rand, &expected_rand, dump_dir);

    // Test 5: Random with random weight
    println!("\n--- Test 5: Random, random weight ---");
    let weight_rand = load_npy_f32(dump_dir.join("weight_rand.npy"));
    let expected_rand_w = load_npy_f32(dump_dir.join("expected_rand_w.npy"));
    let output_rand_w = run_test(&input_rand, &weight_rand, hidden_size, eps, num_rows, false);
    compare_and_report("rand_w", &output_rand_w, &expected_rand_w, dump_dir);

    // Test 6: Decode-only mode (single row)
    println!("\n--- Test 6: Decode-only mode ---");
    let input_one_row = &input_rand[..hidden_size];
    let weight_rand_slice = &weight_rand;
    let input_one_f16 = f32_to_f16(input_one_row);
    let weight_rand_f16 = f32_to_f16(weight_rand_slice);
    let mut output_decode_f16 = vec![0.0f16; hidden_size];
    let op_decode = RMSMap::<f16>::new(
        input_one_f16.as_ptr(),
        weight_rand_f16.as_ptr(),
        output_decode_f16.as_mut_ptr(),
        hidden_size,
        eps,
        true,
    );
    op_decode.run(0, 1, 1, 0);
    let output_decode = f16_to_f32(&output_decode_f16);
    let expected_decode = &expected_rand_w[..hidden_size];
    write_npy_f32(dump_dir.join("rust_decode.npy"), &output_decode);
    compare_and_report("decode", &output_decode, expected_decode, dump_dir);

    println!("\n=== Done ===");
}
