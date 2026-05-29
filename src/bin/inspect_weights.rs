#![feature(f16)]

use ellm::runtime::io::SafeTensorsLoader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = "models/Qwen3-Coder-30B-A3B-Instruct";
    let params = SafeTensorsLoader::new(model_dir)?.load_all_weights_f16()?;

    let mut total = 0usize;
    let mut non_finite = 0usize;
    let mut worst_name = String::new();
    let mut worst_non_finite = 0usize;

    for (name, data) in &params {
        let bad = data.iter().filter(|v| !((**v as f32).is_finite())).count();
        if bad > worst_non_finite {
            worst_non_finite = bad;
            worst_name = name.clone();
        }
        non_finite += bad;
        total += data.len();
    }

    println!("tensors: {}", params.len());
    println!("values: {}", total);
    println!("non_finite_after_f16_load: {}", non_finite);
    println!("worst_tensor: {} ({})", worst_name, worst_non_finite);

    Ok(())
}
