use ellm::operators::transform::SiluMulZipMap;
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

fn main() {
    println!("===== SiluMul Alignment Test =====");

    let dump_dir = Path::new("alignment").join("silu_mul").join("dump");
    std::fs::create_dir_all(&dump_dir).unwrap();

    // Load inputs from NumPy files
    let x1 = load_npy_f32(dump_dir.join("python_silu_mul_x1.npy"));
    let x2 = load_npy_f32(dump_dir.join("python_silu_mul_x2.npy"));
    let python_output = load_npy_f32(dump_dir.join("python_silu_mul_output.npy"));

    // Shape parameters
    let batch_size = 4;
    let hidden_size = 2048;
    let head_num = 1;
    let head_size = hidden_size;
    let prefill_size = batch_size;

    // Initialize output vector
    let mut output = vec![0.0f32; batch_size * hidden_size];

    // Create operator and run
    let operator = SiluMulZipMap::<f32>::new(
        x1.as_ptr(),
        x2.as_ptr(),
        output.as_mut_ptr(),
        head_num,
        head_size,
    );
    operator.run(prefill_size, 1, 0);

    // Write Rust output to NumPy file
    write_npy_f32(dump_dir.join("rust_silu_mul_output.npy"), &output);

    // Compare outputs
    println!("\n--- Comparing outputs ---");
    let mut max_abs_error = 0.0f32;
    let mut mean_abs_error = 0.0f32;
    let mut cosine_numerator = 0.0f32;
    let mut norm_rust = 0.0f32;
    let mut norm_python = 0.0f32;

    for i in 0..batch_size * hidden_size {
        let diff = (output[i] - python_output[i]).abs();
        max_abs_error = max_abs_error.max(diff);
        mean_abs_error += diff;

        cosine_numerator += output[i] * python_output[i];
        norm_rust += output[i] * output[i];
        norm_python += python_output[i] * python_output[i];
    }

    mean_abs_error /= (batch_size * hidden_size) as f32;
    let cosine_similarity = cosine_numerator / (norm_rust.sqrt() * norm_python.sqrt());

    println!("max_abs_error: {:.2e}", max_abs_error);
    println!("mean_abs_error: {:.2e}", mean_abs_error);
    println!("cosine_similarity: {:.8}", cosine_similarity);

    let passed = max_abs_error < 1e-5 && mean_abs_error < 1e-6 && cosine_similarity > 0.999999;
    println!("\n{}", if passed { "PASS" } else { "FAIL" });

    println!("\n--- Done ---");
}
