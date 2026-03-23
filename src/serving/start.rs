use core_affinity;
use std::cell::RefCell;
use std::cell::SyncUnsafeCell;
use std::hint::spin_loop;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Instant;

// use hurdles::Barrier;
// use super::barrier::Barrier;
// use serde::{Deserialize, Serialize};
use std::ops::{AddAssign, Neg, Sub};

use super::super::compiler::operator::Operator;
// use crate::kernel::generic::from_f32::FromF32;
use crate::kernel::generic::sigmoid::Sigmoid;
use crate::kernel::generic::sqrt::Sqrt;
use crate::kernel::generic::{exp::Exp, neg_infinity::NegInfinity};
// use super::state::State;

struct Barrier {
    arrived: AtomicUsize,
    generation: AtomicUsize,
    parties: usize,
}

impl Barrier {
    fn new(parties: usize) -> Self {
        Self {
            arrived: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
            parties,
        }
    }

    fn wait(&self) {
        let generation = self.generation.load(Ordering::Acquire);
        let arrived = self.arrived.fetch_add(1, Ordering::AcqRel) + 1;

        if arrived == self.parties {
            self.arrived.store(0, Ordering::Release);
            self.generation.fetch_add(1, Ordering::AcqRel);
            return;
        }

        while self.generation.load(Ordering::Acquire) == generation {
            spin_loop();
        }
    }
}

pub fn start<T>(operator_queue: Vec<Operator<T>>)
where
    T: PartialOrd
        + Copy
        + Default
        + Sub<Output = T>
        + Neg<Output = T>
        + Exp
        + NegInfinity
        + Sigmoid<T>
        + Sqrt
        + AddAssign
        + Send
        + Sync
        + 'static,
{
    println!("start");
    // let prompt_operator_num;
    // let data = SyncUnsafeCell::new(DataReader::new(prompt_data));
    let thread_num = thread::available_parallelism().unwrap().get();
    let sync_operator_queue = Arc::new(operator_queue);

    let barrier = Arc::new(Barrier::new(thread_num));

    let sequence_chunk_size = 64;
    let mut handles = Vec::with_capacity(thread_num);
    let core_ids = core_affinity::get_core_ids().unwrap();
    for (i, core_id) in core_ids.into_iter().enumerate() {
        // println!("thread id {}", i);
        // let _state = &state;
        // let _prompt_begin = &prompt_begin;
        // let _prompt_end = &prompt_end;
        // let _generation_end = &generation_end;
        // let _batch_size = &batch_size;
        let b = Arc::clone(&barrier);
        // let mut b = barrier .clone();
        let queue = Arc::clone(&sync_operator_queue);

        // let start_pos = start_pos; // 显式捕获当前值

        // let decode_start = 40;

        let handle = thread::spawn(move || {
            let thread_id = i;
            core_affinity::set_for_current(core_id);
            println!("{} start", thread_id);
            // let mut counter = 0;

            // 预先创建子切片，避免在热循环中重复操作
            // let prompt_queue_slice = &queue[..decode_start.min(queue.len())];
            // let decode_queue_slice = &queue[decode_start.min(queue.len())..];

            let sequence_length = 128;

            let s = Instant::now();
            let batch_size = 1;
            for p in 0..sequence_length {
                println!("thread {} position {}", thread_id, p);
                for operator in queue.iter() {
                    operator.run(0, 1, batch_size, thread_num, thread_id);
                    b.wait();
                }
            }
            let t = s.elapsed();
            println!("thread {} decode time {:?}", thread_id, t);
        });

        // std::mem::forget(handle);
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;

    use super::*;
    use crate::memory::allocator::allocate_init;
    use crate::memory::cache::Cache;
    use crate::ptensor::tensor::Tensor;
    use crate::qwen3_moe::config::Config;
    use crate::qwen3_moe::model::Model;
    // use crate::qwen3_moe::sparse_moe_block::SparseMoeBlock;

    // use crate::memory::allocator::allocate_init;

    /*
    #[test]
    fn test_start() {
        let position_window_size = 4;
        let batch_size = 24;
        // let head_size = 128;

        let hidden_size = 256;
        let intermediate_size = 4 * hidden_size;
        let num_experts = 128;
        let top_k = 8;
        let norm_topk_prob = true;

        let cache = Rc::new(RefCell::new(Cache::<f32>::new(
            std::collections::HashMap::new(),
        )));
        let operator_queue = Rc::new(RefCell::new(Vec::new()));

        let sparse_moe = SparseMoeBlock::<f32>::new(
            // position_window_size,
            hidden_size,
            intermediate_size,
            num_experts,
            top_k,
            norm_topk_prob,
            "model.layers.0",
            cache.clone(),
            operator_queue.clone(),
        );

        let shape = vec![position_window_size, batch_size, hidden_size];
        let input = Tensor::from_cache(
            shape.clone(),
            String::from("model.layers.0.input_tensor"),
            cache.clone(),
            operator_queue.clone(),
        );

        let residual = Tensor::from_cache(
            shape.clone(),
            String::from("model.layers.0.residual_tensor"),
            cache.clone(),
            operator_queue.clone(),
        );

        for i in 0..input.shape.iter().product() {
            unsafe {
                input.data.add(i).write(1.0);
            }
        }

        for i in 0..residual.shape.iter().product() {
            unsafe {
                residual.data.add(i).write(1.0);
            }
        }

        let output_tensor = sparse_moe.forward(
            &input,
            &residual,
            String::from("model.layers.0.output_tensor"),
        );

        /*
        let thread_num: usize = num_cpus::get();
        for (index, operator) in output_tensor.operator_queue.borrow().iter().enumerate() {
            println!("operator {} in queue", index);
            for i in 0..thread_num {
                operator.run(0, 1, batch_size, thread_num, i);
            }
        }*/

        // output_tensor.operator_queue.borrow().to_vec()
        start(output_tensor.operator_queue.take());
    } */

    #[test]
    fn test_model_start() {
        let sequence_length = 128;
        let sequence_chunk_size = 1;
        let batch_size = 6;
        let topk_size = 8;

        let config =
            Config::load_from_file(r"models/Qwen3-Coder-30B-A3B-Instruct/config.json").unwrap();

        let mut model =
            Model::<f32>::new(&config, sequence_length, sequence_chunk_size, batch_size, topk_size);

        let mut sequences =
            allocate_init::<usize>((sequence_length + 1) * batch_size, 0);
        let _ = unsafe { model.forward(sequences) };

        start(model.operator_queue.take());
    }
}
