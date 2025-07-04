#![feature(f16)]
#![feature(duration_millis_float)]

use std::time::Instant;
// use std::time::SystemTime;
use std::cell::RefCell;
use std::rc::Rc;
use std::f16;

use eLLM::compiler::operator::Operator;
use eLLM::init::config::Config;
use eLLM::memory::cache::Cache;
use eLLM::ptensor::tensor::Tensor;
use eLLM::llama::model::Model;
use eLLM::runtime::start::start;

use eLLM::ptensor::linear::Linear;
use eLLM::llama::rope::precompute_freqs_cis;
// use ScaleFlow::init::model_loader;
// use ScaleFlow::serving::data::generate_data;




fn build_linear(
    cache: Rc<RefCell<Cache<f16>>>,
    operator_queue: Rc<RefCell<Vec<Operator<f16>>>>,) {
    let head_size = 128;
    let head_num = 32;
    let hidden_size = head_num * head_size;
    let batch_size = 32;
    let sequence_length = 1;

    // let cache = Rc::new(RefCell::new(Cache::new()));
    // let operator_queue = Rc::new(RefCell::new(Vec::new()));

    let linear = Linear::<f16>::new(hidden_size, hidden_size, sequence_length, String::from("model.layers.0"), cache.clone(), operator_queue.clone());

    for i in 0..linear.weight.shape.iter().product() {
        unsafe { linear.weight.data.add(i).write(1.0) };
    }
    
    let shape1 = vec![batch_size, hidden_size];

    let input = Tensor::from_cache(shape1, String::from("model.layer.0.input_tensor"), cache.clone(), operator_queue.clone());
    for i in 0..input.shape.iter().product() {
        unsafe {
            input.data.add(i).write(1.0);
        }
    }

    let output_shape = vec![batch_size, hidden_size];
    let size3 = output_shape.iter().product();
    let mut result = vec![0.0; size3];
    for i in 0..hidden_size {
        result[i] = hidden_size as f16;
    }

    let output_tensor = linear.forward(&input, String::from("model.layer.0.self_attn.value_tensor"));
    
    
    /*
    let thread_num: usize = num_cpus::get();
    for i in 0..thread_num {
        output_tensor.operator_queue.borrow()[0].run(1, 0, i);
    }

    let output_slice = unsafe { std::slice::from_raw_parts(output_tensor.data, size3) };
    assert_relative_eq!(output_slice, &result[..], max_relative = 1e-6); 
    */
}


fn main() {
    dbg!(is_x86_feature_detected!("avx512fp16"));
    
    println!("Initializing...");

    let cache = Rc::new(RefCell::new(Cache::new()));
    let operator_queue = Rc::new(RefCell::new(Vec::new()));
    
    let cpu_num = num_cpus::get();
    println!("cpu num {}", cpu_num);
    let mut config: Config = Config::new();
    config.load_model_config(r"models/Llama-2-70b-hf/config.json");
    config.load_compile_config(r"models/Llama-2-70b-hf.json");

    let mut sequences: Vec<usize> = vec![0; (config.max_position_embeddings + 1) * config.batch_size];
    build_graph(&config, sequences.as_mut_ptr(), cache.clone(), operator_queue.clone());
    // build_linear(cache.clone(), operator_queue.clone());
    // Borrow the inner Vec<Operator<f32>> from the Rc<RefCell<Vec<Operator<f32>>>>
    start(operator_queue.borrow().clone());
}
