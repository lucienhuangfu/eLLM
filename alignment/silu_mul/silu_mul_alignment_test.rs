// === alignment/silu_mul/silu_mul_alignment_test.rs ===
use std::fs;
use std::path::PathBuf;

// Import our SiluMulZipMap operator
// Note: We need to adjust the path based on the actual project structure
#[path = "../../src/operators/elementwise/silu_mul_zip.rs"]
mod silu_mul;
use silu_mul::SiluMulZipMap;
#[path = "../../src/operators/traits.rs"]
mod traits;
use traits::ZipMapTrait;

// Helper to write npy files (simplified version)
fn write_npy_f32(path: &str, data: &[f32], shape: &[usize]) {
    use npy::NpyData;
    
    // Create npy data
    let mut file = fs::File::create(path).unwrap();
    NpyData::from_slice(data).write(&mut file, shape).unwrap();
}

fn main() {
    println!("===== SiLU+Mul Alignment Test =====");
    
    // Get dump directory
    let script_dir = PathBuf::from(file!()).parent().unwrap().to_path_buf();
    let dump_dir = script_dir.join("dump");
    fs::create_dir_all(&dump_dir).unwrap();
    
    // Test configuration
    let batch_size = 4;
    let head_num = 32;
    let head_size = 128;
    let total_size = batch_size * head_num * head_size;
    
    // Test 1: Basic test with sequential values
    println!("\n--- Test 1: Basic sequential values ---");
    let mut x1: Vec<f32> = (0..total_size).map(|x| x as f32).collect();
    let x2: Vec<f32> = vec![1.0; total_size];
    let mut y: Vec<f32> = vec![0.0; total_size];
    
    let operator = SiluMulZipMap::new(
        x1.as_ptr(),
        x2.as_ptr(),
        y.as_mut_ptr(),
        head_num,
        head_size,
    );
    
    // Run operator with 1 thread
    operator.run(batch_size, 1, 0);
    
    let rust_path = dump_dir.join("rust_silu_mul_basic.npy");
    write_npy_f32(rust_path.to_str().unwrap(), &y, &[batch_size, head_num, head_size]);
    println!("Saved to {:?}", rust_path);
    println!("Output shape: [{:?}, {:?}, {:?}]", batch_size, head_num, head_size);
    
    // Test 2: All zeros
    println!("\n--- Test 2: All zeros ---");
    let x1_zero: Vec<f32> = vec![0.0; total_size];
    let x2_zero: Vec<f32> = vec![1.0; total_size];
    let mut y_zero: Vec<f32> = vec![0.0; total_size];
    
    let operator_zero = SiluMulZipMap::new(
        x1_zero.as_ptr(),
        x2_zero.as_ptr(),
        y_zero.as_mut_ptr(),
        head_num,
        head_size,
    );
    operator_zero.run(batch_size, 1, 0);
    
    let rust_zero_path = dump_dir.join("rust_silu_mul_zeros.npy");
    write_npy_f32(rust_zero_path.to_str().unwrap(), &y_zero, &[batch_size, head_num, head_size]);
    println!("Saved to {:?}", rust_zero_path);
    
    // Test 3: Small random values (we'll use a fixed seed pattern)
    println!("\n--- Test 3: Small random values ---");
    let mut x1_rand: Vec<f32> = Vec::with_capacity(total_size);
    let mut x2_rand: Vec<f32> = Vec::with_capacity(total_size);
    // Generate simple deterministic "random" values
    for i in 0..total_size {
        let val1 = ((i as f32 * 0.1).sin() * 0.01) as f32;
        let val2 = ((i as f32 * 0.2).cos() * 0.01) as f32;
        x1_rand.push(val1);
        x2_rand.push(val2);
    }
    let mut y_rand: Vec<f32> = vec![0.0; total_size];
    
    let operator_rand = SiluMulZipMap::new(
        x1_rand.as_ptr(),
        x2_rand.as_ptr(),
        y_rand.as_mut_ptr(),
        head_num,
        head_size,
    );
    operator_rand.run(batch_size, 1, 0);
    
    let rust_rand_path = dump_dir.join("rust_silu_mul_rand.npy");
    write_npy_f32(rust_rand_path.to_str().unwrap(), &y_rand, &[batch_size, head_num, head_size]);
    println!("Saved to {:?}", rust_rand_path);
    
    println!("\n--- Done ---");
}
