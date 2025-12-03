use core_affinity;
use std::cell::SyncUnsafeCell;
use std::sync::Arc;
use std::sync::Barrier;
use std::thread;
use std::time::{Duration, Instant};

use super::super::compiler::operator::Operator;
use crate::init::record::{Phase, TokenRecord, UserRecord};
use crate::init::send_sync_ptr::MutPtr;

pub fn start(
    operator_queue: Vec<Operator<f32>>,
    token_ptr: MutPtr<TokenRecord>,
    user_ptr: MutPtr<UserRecord>,
    max_batch_size: usize,
    max_token_size: usize,
) {
    println!("start");
    let thread_num = thread::available_parallelism().unwrap().get();
    let sync_operator_queue = Arc::new(operator_queue);
    let barrier = Arc::new(Barrier::new(thread_num));
    let shared_sizes = Arc::new(SyncUnsafeCell::new((0usize, 0usize)));
    let mut handles = Vec::with_capacity(thread_num);
    let core_ids = core_affinity::get_core_ids().unwrap();

    for (i, core_id) in core_ids.into_iter().enumerate() {
        // println!("thread id {}", i);
        let b = Arc::clone(&barrier);
        let queue = Arc::clone(&sync_operator_queue);
        let shared_sizes: Arc<SyncUnsafeCell<(usize, usize)>> = Arc::clone(&shared_sizes);

        let user_ptr_addr = SyncUnsafeCell::new(user_ptr);

        // let max_batch_size = 80;
        // let max_token_size = max_token_size;

        let handle = thread::spawn(move || {
            let thread_id = i;
            core_affinity::set_for_current(core_id);
            println!("{} start", thread_id);
            let s = Instant::now();
            let sizes_ptr = shared_sizes.get();
            loop {
                if thread_id == 0 {
                    // let mut token_record = unsafe { &mut *token_ptr.add(p * batch_size) };
                    let mut flag = false;
                    unsafe {
                        let user_raw_ptr = (*user_ptr_addr.get()).ptr;
                        loop {
                            for i in 0..max_batch_size {
                                let user_record = &mut *user_raw_ptr.add(i);
                                match user_record.phase {
                                    Phase::Prefill_begin => {
                                        if user_record.sequence_index > user_record.kv_index + 1 {
                                            flag = true;
                                            break;
                                        }
                                    }
                                    Phase::Prefill_end => {
                                        if user_record.sequence_index > user_record.kv_index {
                                            flag = true;
                                            break;
                                        }
                                    }
                                    Phase::Decode => {
                                        flag = true;
                                        break;
                                    }
                                    Phase::Eos => {}
                                }
                            }
                            if flag {
                                break;
                            } else {
                                thread::sleep(Duration::from_micros(1));
                            }
                        }
                        let mut token_size = 0;
                        let mut decode_size = 0;
                        for i in 0..max_batch_size {
                            if token_size >= max_token_size {
                                break;
                            }
                            let user_record = &mut *user_raw_ptr.add(i);
                            match user_record.phase {
                                Phase::Prefill_begin => {
                                    let sequence_index = user_record.sequence_index;
                                    user_record.snapshot_sequence_index = sequence_index - 1;
                                    let len = sequence_index - user_record.kv_index - 1;
                                    if len > 0 {
                                        let take = std::cmp::min(len, max_token_size - token_size);
                                        user_record.kv_index = user_record.kv_index + take;
                                        token_size += take;
                                    }
                                }

                                Phase::Prefill_end => {
                                    let len = user_record.sequence_index - user_record.kv_index;
                                    if len > 0 {
                                        let take = std::cmp::min(len, max_token_size - token_size);
                                        user_record.kv_index = user_record.kv_index + take;
                                        token_size += take;
                                    }
                                }
                                Phase::Decode => {
                                    user_record.kv_index += 1;
                                    user_record.sequence_index += 1;
                                    token_size += 1;
                                    decode_size += 1;
                                }
                                Phase::Eos => {}
                            }
                        }
                        *sizes_ptr = (token_size, decode_size);
                    }
                }
                b.wait();

                let (token_size, decode_size) = unsafe { *sizes_ptr };

                for operator in queue.iter() {
                    operator.run(token_size, decode_size, thread_num, thread_id);
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
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;
    use crate::memory::cache::Cache;
    use crate::ptensor::tensor::Tensor;
    use crate::qwen3_moe::sparse_moe_block::SparseMoeBlock;

    // use crate::memory::allocator::allocate_init;

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

        let max_token_size = 100;
        let mut token_records = (0..max_token_size)
            .map(|_| TokenRecord {
                token_id: 0,
                batch_index: 0,
                position_index: 0,
            })
            .collect::<Vec<_>>();

        let mut user_records = (0..max_token_size)
            .map(|_| UserRecord {
                sequence_index: 0,
                kv_index: 0,
                phase: Phase::Prefill,
            })
            .collect::<Vec<_>>();

        let token_ptr = MutPtr {
            ptr: token_records.as_mut_ptr(),
        };
        let user_ptr = MutPtr {
            ptr: user_records.as_mut_ptr(),
        };

        start(
            output_tensor.operator_queue.take(),
            token_ptr,
            user_ptr,
            max_token_size,
        );
    }
}
