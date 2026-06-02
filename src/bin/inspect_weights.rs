#![feature(f16)]

use ellm::runtime::io::SafeTensorsLoader;
use std::env;
use std::f16;
use std::time::Instant;

#[derive(Debug)]
struct Args {
    model_dir: String,
    parallel: bool,
    validate: bool,
    fingerprint: bool,
}

fn parse_args() -> Args {
    let mut model_dir = None;
    let mut parallel = false;
    let mut validate = true;
    let mut fingerprint = false;

    for arg in env::args().skip(1) {
        match arg.as_str() {
            "--parallel" => parallel = true,
            "--serial" => parallel = false,
            "--no-validate" => validate = false,
            "--validate" => validate = true,
            "--fingerprint" => fingerprint = true,
            "--help" | "-h" => {
                println!(
                    "usage: inspect_weights [--serial|--parallel] [--validate|--no-validate] [--fingerprint] [model_dir]"
                );
                std::process::exit(0);
            }
            _ if arg.starts_with('-') => {
                eprintln!("unknown option: {arg}");
                std::process::exit(2);
            }
            _ => model_dir = Some(arg),
        }
    }

    Args {
        model_dir: model_dir.unwrap_or_else(|| "models/Qwen3-Coder-30B-A3B-Instruct".to_string()),
        parallel,
        validate,
        fingerprint,
    }
}

fn update_hash(hash: &mut u64, bytes: &[u8]) {
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

fn fingerprint_params(params: &std::collections::HashMap<String, Vec<f16>>) -> u64 {
    let mut names: Vec<&String> = params.keys().collect();
    names.sort();

    let mut hash = 0xcbf2_9ce4_8422_2325;
    for name in names {
        update_hash(&mut hash, name.as_bytes());
        update_hash(&mut hash, &(params[name].len() as u64).to_le_bytes());
        for value in &params[name] {
            update_hash(&mut hash, &value.to_bits().to_le_bytes());
        }
    }
    hash
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let loader = SafeTensorsLoader::new(&args.model_dir)?;

    let started = Instant::now();
    let params = if args.parallel {
        loader.load_all_weights_f16_parallel()?
    } else {
        loader.load_all_weights_f16()?
    };
    let load_elapsed = started.elapsed();

    let mut total = 0usize;
    let mut non_finite = 0usize;
    let mut worst_name = String::new();
    let mut worst_non_finite = 0usize;

    for data in params.values() {
        total += data.len();
    }

    if args.validate {
        for (name, data) in &params {
            let bad = data.iter().filter(|v| !((**v as f32).is_finite())).count();
            if bad > worst_non_finite {
                worst_non_finite = bad;
                worst_name = name.clone();
            }
            non_finite += bad;
        }
    }

    println!(
        "mode: {}",
        if args.parallel { "parallel" } else { "serial" }
    );
    println!("tensors: {}", params.len());
    println!("values: {}", total);
    if args.validate {
        println!("non_finite_after_f16_load: {}", non_finite);
        println!("worst_tensor: {} ({})", worst_name, worst_non_finite);
    }
    if args.fingerprint {
        println!("fingerprint_fnv1a64: {:016x}", fingerprint_params(&params));
    }
    println!("load_weights: {:.3}s", load_elapsed.as_secs_f64());

    Ok(())
}
