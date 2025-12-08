use core_affinity;
use std::cell::SyncUnsafeCell;
use std::sync::Arc;
use std::sync::Barrier;
use std::thread;
use std::time::{Duration, Instant};

use super::super::compiler::operator::Operator;
use crate::init::record::{
    LastPrefillList, LastPrefillRecord, Phase, TokenList, TokenRecord, UserList,
};
use crate::init::send_sync_ptr::MutPtr;

/// Helper function to fill a range of token records.
/// Returns the number of tokens added.
#[inline(always)]
fn fill_token_batch(
    token_raw_ptr: &mut [TokenRecord],
    start_index: usize,
    batch_index: usize,
    start_pos: usize,
    count: usize,
) {
    let end_index = start_index + count;
    for (p, token_record) in token_raw_ptr[start_index..end_index].iter_mut().enumerate() {
        token_record.batch_index = batch_index;
        token_record.position_index = start_pos + p;
    }
}

fn schedule_batch(
    user_list: &mut UserList,
    token_list: &mut TokenList,
    last_prefill_list: &mut LastPrefillList,
) -> (usize, usize, usize) {
    let max_token_size = token_list.records.len();
    let current_batch_size = user_list.current_size;
    let user_raw_ptr = &mut user_list.records;
    let token_raw_ptr = &mut token_list.records;

    // Capture max size before mutable borrow of records to avoid borrow checker error
    let max_prefill_size = last_prefill_list.records.len();
    let last_prefill_raw_ptr = &mut last_prefill_list.records;

    loop {
        let mut token_index = 0;
        // Reset the size for the current batch generation
        last_prefill_list.current_size = 0;

        // Pass 1: Handle Decode Phase (Priority)
        for (i, user_record) in user_raw_ptr.iter_mut().take(current_batch_size).enumerate() {
            if token_index >= max_token_size {
                break;
            }
            if let Phase::Decode = user_record.phase {
                user_record.kv_index += 1;

                // Optimization: Inline single token write for Decode phase to avoid function call overhead
                // fill_token_batch(token_raw_ptr, token_index, i, user_record.kv_index, 1);
                let token_record = &mut token_raw_ptr[token_index];
                token_record.batch_index = i;
                token_record.position_index = user_record.kv_index;

                token_index += 1;
            }
        }

        let decode_index = token_index;

        // Pass 2: Handle Prefill Phase
        let mut lift_index = decode_index;
        for (i, user_record) in user_raw_ptr.iter_mut().take(current_batch_size).enumerate() {
            if token_index >= max_token_size {
                break;
            }

            // Calculate remaining length safely
            let len = match user_record.phase {
                Phase::Prefill_begin | Phase::Prefill_end => user_record
                    .sequence_index
                    .saturating_sub(user_record.kv_index)
                    .saturating_sub(1),
                _ => 0,
            };

            if len > 0 {
                let take = std::cmp::min(len, max_token_size - token_index);

                fill_token_batch(token_raw_ptr, token_index, i, user_record.kv_index, take);

                user_record.kv_index += take;

                if let Phase::Prefill_end = user_record.phase {
                    // Optimization: Direct indexing into pre-allocated buffer instead of push
                    if last_prefill_list.current_size < max_prefill_size {
                        last_prefill_raw_ptr[last_prefill_list.current_size] = LastPrefillRecord {
                            prefill_index: token_index,
                            lift_index: lift_index,
                        };
                        last_prefill_list.current_size += 1;
                    }
                    lift_index += 1;
                }

                token_index += take;
            }
        }

        if token_index > 0 {
            return (token_index, decode_index, lift_index);
        } else {
            thread::sleep(Duration::from_micros(1));
        }
    }
}

/// Starts the inference serving loop.
///
/// This function initializes a thread pool where Thread 0 schedules tasks by monitoring
/// user request phases (Prefill/Decode) and populating the token list. All threads
/// then synchronize to execute the operators in the queue for the current batch.
pub fn start(
    user_list_ptr: MutPtr<UserList>,
    token_list_ptr: MutPtr<TokenList>,
    last_prefill_list_ptr: MutPtr<LastPrefillList>,
    operator_queue: Vec<Operator<f32>>,
) {
    println!("start");
    let thread_num = thread::available_parallelism().unwrap().get();

    // Optimization: Convert Vec to Arc<[T]> to reduce one level of indirection compared to Arc<Vec<T>>
    let sync_operator_queue: Arc<[Operator<f32>]> = operator_queue.into();

    let barrier = Arc::new(Barrier::new(thread_num));
    let shared_sizes = Arc::new(SyncUnsafeCell::new((0usize, 0usize, 0usize)));
    let mut handles = Vec::with_capacity(thread_num);
    let core_ids = core_affinity::get_core_ids().unwrap();

    for (i, core_id) in core_ids.into_iter().enumerate() {
        // println!("thread id {}", i);
        let b = Arc::clone(&barrier);
        let queue = Arc::clone(&sync_operator_queue);
        let shared_sizes: Arc<SyncUnsafeCell<(usize, usize, usize)>> = Arc::clone(&shared_sizes);

        // Wrap pointers in SyncUnsafeCell to ensure safe transport across threads.
        // This creates a new cell for each thread containing a copy of the pointer.
        let user_ptr_addr = SyncUnsafeCell::new(user_list_ptr);
        let token_list_ptr_addr = SyncUnsafeCell::new(token_list_ptr);
        let last_prefill_list_ptr_addr = SyncUnsafeCell::new(last_prefill_list_ptr);

        let handle = thread::spawn(move || {
            let thread_id = i;
            core_affinity::set_for_current(core_id);
            println!("{} start", thread_id);
            let s = Instant::now();
            let sizes_ptr = shared_sizes.get();

            // Main inference loop: continuously processes batches of tokens
            loop {
                // Thread 0 acts as the scheduler: monitors user states and prepares the token batch
                if thread_id == 0 {
                    unsafe {
                        let user_list = &mut *(*user_ptr_addr.get()).ptr;
                        let token_list = &mut *(*token_list_ptr_addr.get()).ptr;
                        let last_prefill_list = &mut *(*last_prefill_list_ptr_addr.get()).ptr;
                        *sizes_ptr = schedule_batch(user_list, token_list, last_prefill_list);
                    }
                }

                // Synchronization barrier: Wait for Thread 0 to finish scheduling
                b.wait();

                let (token_size, decode_size, lift_size) = unsafe { *sizes_ptr };

                // Execute the operator queue in parallel
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
    use crate::init::record::{LastPrefillList, LastPrefillRecord, TokenRecord, UserRecord};
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
            false,
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
        let token_records = (0..max_token_size)
            .map(|_| TokenRecord {
                token_id: 0,
                batch_index: 0,
                position_index: 0,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let user_records = (0..batch_size)
            .map(|_| UserRecord {
                sequence_index: 0,
                snapshot_sequence_index: 0,
                kv_index: 0,
                phase: Phase::Prefill_begin,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let mut token_list = TokenList {
            records: token_records,
            current_size: 0,
        };

        let mut user_list = UserList {
            records: user_records,
            current_size: batch_size,
        };

        // Pre-allocate LastPrefillList buffer
        let last_prefill_records = vec![
            LastPrefillRecord {
                prefill_index: 0,
                lift_index: 0
            };
            batch_size
        ]
        .into_boxed_slice();

        let mut last_prefill_list = LastPrefillList {
            records: last_prefill_records,
            current_size: 0,
        };

        let token_ptr = MutPtr {
            ptr: &mut token_list as *mut TokenList,
        };
        let user_ptr = MutPtr {
            ptr: &mut user_list as *mut UserList,
        };
        let last_prefill_ptr = MutPtr {
            ptr: &mut last_prefill_list as *mut LastPrefillList,
        };

        start(
            user_ptr,
            token_ptr,
            last_prefill_ptr,
            output_tensor.operator_queue.take(),
        );
    }
}
