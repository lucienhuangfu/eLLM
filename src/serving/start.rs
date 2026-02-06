use core_affinity;
use std::cell::SyncUnsafeCell;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::sync::Arc;
use std::sync::Barrier;
use std::thread;
use std::time::{Duration, Instant};

use super::super::compiler::operator::Operator;

use super::super::kernel::generic::{exp::Exp, neg_infinity::NegInfinity, sqrt::Sqrt};
use crate::init::record::{BatchList, Phase, SequenceSlice, TaskList, ThreadTask};
use crate::init::send_sync_ptr::MutPtr;

fn schedule_batch(
    batch_list: &mut BatchList,
    prefill_list: &mut TaskList,
    decode_list: &mut TaskList,
) -> (usize, usize) {
    let prefill_task_count = prefill_list.tasks.len();
    let decode_task_count = decode_list.tasks.len();

    loop {
        let mut has_decode = false;
        let mut has_prefill = false;
        let mut has_prefill_end = false;

        for record in batch_list.records[..batch_list.current_size].iter() {
            if record.phase == Phase::Decode {
                has_decode = true;
                break;
            }
        }

        for record in batch_list.records[..batch_list.current_size].iter() {
            if record.phase == Phase::PrefillBegin || record.phase == Phase::PrefillEnd {
                has_prefill = true;
                if record.phase == Phase::PrefillEnd {
                    has_prefill_end = true;
                }
                break;
            }
        }

        if !has_decode && !has_prefill {
            thread::sleep(Duration::from_millis(1));
            continue;
        }

        for task in prefill_list.tasks.iter_mut() {
            task.current_size = 0;
        }
        for task in decode_list.tasks.iter_mut() {
            task.current_size = 0;
        }

        let mut token_count = 0usize;
        let mut decode_count = 0usize;

        if has_decode {
            let max_token_size = decode_list.max_token_size;
            if decode_task_count != 0 && max_token_size != 0 {
                let mut total_tokens = 0usize;
                for record in batch_list.records[..batch_list.current_size].iter() {
                    if record.phase == Phase::Decode {
                        total_tokens += 1;
                    }
                }

                if total_tokens != 0 {
                    total_tokens = total_tokens.min(max_token_size);
                    let mut task_index = 0usize;
                    let mut task_remaining =
                        (total_tokens + decode_task_count - 1) / decode_task_count;
                    let mut scheduled_tokens = 0usize;

                    for (batch_index, record) in batch_list.records[..batch_list.current_size]
                        .iter_mut()
                        .enumerate()
                    {
                        if scheduled_tokens >= total_tokens {
                            break;
                        }
                        if record.phase != Phase::Decode {
                            continue;
                        }

                        while task_index < decode_task_count && task_remaining == 0 {
                            task_index += 1;
                            if task_index < decode_task_count {
                                task_remaining =
                                    (total_tokens - scheduled_tokens
                                        + (decode_task_count - task_index)
                                        - 1)
                                        / (decode_task_count - task_index);
                            }
                        }
                        if task_index >= decode_task_count {
                            break;
                        }

                        let task = &mut decode_list.tasks[task_index];
                        if task.current_size >= task.slices.len() {
                            break;
                        }

                        let slice = SequenceSlice {
                            batch_index,
                            sequence_index: record.sequence_index,
                            token_start_index: scheduled_tokens,
                            length: 1,
                        };
                        let idx = task.current_size;
                        task.slices[idx] = slice;
                        task.current_size += 1;

                        record.kv_index = record.kv_index.saturating_add(1);
                        token_count += 1;
                        decode_count += 1;
                        scheduled_tokens += 1;
                        if task_remaining > 0 {
                            task_remaining -= 1;
                        }
                    }
                }
            }
        }

        if has_prefill && !has_decode {
            let max_token_size = prefill_list.max_token_size;
            if prefill_task_count != 0 && max_token_size != 0 {
                let mut total_tokens = 0usize;
                for record in batch_list.records[..batch_list.current_size].iter() {
                    if record.phase != Phase::PrefillBegin && record.phase != Phase::PrefillEnd {
                        continue;
                    }
                    let remaining = record
                        .sequence_index
                        .saturating_sub(record.kv_index);
                    total_tokens += remaining;
                }

                if total_tokens != 0 {
                    total_tokens = total_tokens.min(max_token_size);
                    let mut task_index = 0usize;
                    let mut task_remaining =
                        (total_tokens + prefill_task_count - 1) / prefill_task_count;
                    let mut scheduled_tokens = 0usize;

                    for (batch_index, record) in batch_list.records[..batch_list.current_size]
                        .iter_mut()
                        .enumerate()
                    {
                        if scheduled_tokens >= total_tokens {
                            break;
                        }
                        if record.phase != Phase::PrefillBegin && record.phase != Phase::PrefillEnd {
                            continue;
                        }

                        if record.snapshot_sequence_index != record.sequence_index {
                            record.snapshot_sequence_index = record.sequence_index;
                        }

                        let mut remaining = record
                            .snapshot_sequence_index
                            .saturating_sub(record.kv_index);

                        while remaining > 0 && scheduled_tokens < total_tokens {
                            while task_index < prefill_task_count && task_remaining == 0 {
                                task_index += 1;
                                if task_index < prefill_task_count {
                                    task_remaining =
                                        (total_tokens - scheduled_tokens
                                            + (prefill_task_count - task_index)
                                            - 1)
                                            / (prefill_task_count - task_index);
                                }
                            }
                            if task_index >= prefill_task_count {
                                break;
                            }

                            let task = &mut prefill_list.tasks[task_index];
                            if task.current_size >= task.slices.len() {
                                break;
                            }

                            let available = total_tokens - scheduled_tokens;
                            let take = remaining.min(task_remaining).min(available);
                            if take == 0 {
                                break;
                            }

                            let slice = SequenceSlice {
                                batch_index,
                                sequence_index: record.sequence_index,
                                token_start_index: scheduled_tokens,
                                length: take,
                            };

                            let idx = task.current_size;
                            task.slices[idx] = slice;
                            task.current_size += 1;

                            record.kv_index = record.kv_index.saturating_add(take);
                            token_count += take;
                            scheduled_tokens += take;
                            remaining -= take;
                            if task_remaining > 0 {
                                task_remaining = task_remaining.saturating_sub(take);
                            }
                        }
                    }
                }
            }
        }

        if has_prefill_end && !has_decode {
            let max_token_size = decode_list.max_token_size;
            if decode_task_count != 0 && max_token_size != 0 {
                let mut decode_tokens_used = 0usize;
                let mut task_token_counts = vec![0usize; decode_task_count];
                for (task_index, task) in decode_list.tasks.iter().enumerate() {
                    let mut task_tokens = 0usize;
                    for i in 0..task.current_size {
                        task_tokens += task.slices[i].length;
                    }
                    task_token_counts[task_index] = task_tokens;
                    decode_tokens_used += task_tokens;
                }

                let remaining_budget = max_token_size.saturating_sub(decode_tokens_used);
                if remaining_budget > 0 {
                    let mut total_tokens = 0usize;
                    for record in batch_list.records[..batch_list.current_size].iter() {
                        if record.phase == Phase::PrefillEnd {
                            total_tokens += 1;
                        }
                    }

                    if total_tokens > 0 {
                        total_tokens = total_tokens.min(remaining_budget);
                        let mut task_index = 0usize;
                        let mut task_remaining =
                            (total_tokens + decode_task_count - 1) / decode_task_count;
                        let mut scheduled_tokens = 0usize;

                        for (batch_index, record) in batch_list.records[..batch_list.current_size]
                            .iter()
                            .enumerate()
                        {
                            if scheduled_tokens >= total_tokens {
                                break;
                            }
                            if record.phase != Phase::PrefillEnd {
                                continue;
                            }

                            while task_index < decode_task_count && task_remaining == 0 {
                                task_index += 1;
                                if task_index < decode_task_count {
                                    task_remaining =
                                        (total_tokens - scheduled_tokens
                                            + (decode_task_count - task_index)
                                            - 1)
                                            / (decode_task_count - task_index);
                                }
                            }
                            if task_index >= decode_task_count {
                                break;
                            }

                            let task = &mut decode_list.tasks[task_index];
                            if task.current_size >= task.slices.len() {
                                break;
                            }

                            let slice = SequenceSlice {
                                batch_index,
                                sequence_index: record.sequence_index.saturating_sub(1),
                                token_start_index: decode_tokens_used + scheduled_tokens,
                                length: 1,
                            };
                            let idx = task.current_size;
                            task.slices[idx] = slice;
                            task.current_size += 1;

                            token_count += 1;
                            decode_count += 1;
                            scheduled_tokens += 1;
                            if task_remaining > 0 {
                                task_remaining -= 1;
                            }
                        }
                    }
                }
            }
        }

        if token_count == 0 && decode_count == 0 {
            thread::sleep(Duration::from_millis(1));
            continue;
        }

        return (token_count, decode_count);
    }
}

/// Starts the inference serving loop.
/// This function initializes a thread pool where Thread 0 schedules tasks by monitoring
/// user request phases (Prefill/Decode) and populating the token list. All threads
/// then synchronize to execute the operators in the queue for the current batch.
pub fn start<T>(
    batch_list_ptr: MutPtr<BatchList>,
    operator_queue: Vec<Operator<T>>,
) where
    T: Send
        + Sync
        + 'static
        + Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Neg<Output = T>
        + AddAssign
        + Exp
        + Sqrt
        + NegInfinity,
{
    println!("start");
    // let thread_num = thread::available_parallelism().unwrap().get();
    let core_ids = core_affinity::get_core_ids().unwrap();
    let thread_num = core_ids.len();

    // Optimization: Convert Vec to Arc<[T]> to reduce one level of indirection compared to Arc<Vec<T>>
    let sync_operator_queue: Arc<[Operator<T>]> = operator_queue.into();

    let barrier = Arc::new(Barrier::new(thread_num));
    let shared_sizes = Arc::new(SyncUnsafeCell::new((0usize, 0usize)));
    let mut handles = Vec::with_capacity(thread_num);
    // let core_ids = core_affinity::get_core_ids().unwrap();

    for (i, core_id) in core_ids.into_iter().enumerate() {
        // println!("thread id {}", i);
        let b = Arc::clone(&barrier);
        let queue = Arc::clone(&sync_operator_queue);
        let shared_sizes: Arc<SyncUnsafeCell<(usize, usize)>> = Arc::clone(&shared_sizes);

        // Wrap pointers in SyncUnsafeCell to ensure safe transport across threads.
        // This creates a new cell for each thread containing a copy of the pointer.
        let batch_list_ptr_addr = SyncUnsafeCell::new(batch_list_ptr);
        // let last_prefill_list_ptr_addr = SyncUnsafeCell::new(last_prefill_list_ptr);

        let handle = thread::spawn(move || {
            let thread_id = i;
            core_affinity::set_for_current(core_id);
            println!("{} start", thread_id);
            let s = Instant::now();
            let sizes_ptr = shared_sizes.get();
            let mut task_list = if thread_id == 0 {
                let batch_size = unsafe { (&*(*batch_list_ptr_addr.get()).ptr).current_size };
                let max_token_size = batch_size.max(1);
                let task_count = thread_num.max(1);
                let slices_per_task = (max_token_size + task_count - 1) / task_count;
                let tasks = (0..task_count)
                    .map(|_| ThreadTask {
                        slices: vec![
                            SequenceSlice {
                                batch_index: 0,
                                sequence_index: 0,
                                length: 0,
                            };
                            slices_per_task
                        ]
                        .into_boxed_slice(),
                        current_size: 0,
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice();

                Some(TaskList {
                    tasks,
                    current_size: task_count,
                    max_token_size,
                })
            } else {
                None
            };

            // Main inference loop: continuously processes batches of tokens
            loop {
                // Thread 0 acts as the scheduler: monitors user states and prepares the token batch
                if thread_id == 0 {
                    unsafe {
                        let batch_list = &mut *(*batch_list_ptr_addr.get()).ptr;
                        let task_list = task_list.as_mut().unwrap();
                        *sizes_ptr = schedule_batch(batch_list, task_list);
                    }
                }

                // Synchronization barrier: Wait for Thread 0 to finish scheduling
                b.wait();

                let (token_size, decode_size) = unsafe { *sizes_ptr };

                // Execute the operator queue in parallel
                for operator in queue.iter() {
                    operator.run(token_size, decode_size, thread_num, thread_id);
                    b.wait();
                }
            }

            // let t = s.elapsed();
            // println!("thread {} decode time {:?}", thread_id, t);
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
    use crate::init::record::BatchRecord;
    use crate::memory::cache::Cache;
    use crate::ptensor::tensor::Tensor;
    use crate::qwen3_moe::sparse_moe_block::SparseMoeBlock;

    // use crate::memory::allocator::allocate_init;

    
    #[test]
    fn test_schedule_batch_decode_only() {
        let batch_records = vec![
            BatchRecord {
                sequence_index: 100,
                snapshot_sequence_index: 0,
                kv_index: 10,
                phase: Phase::Decode,
                prompt_length: 100,
                notify: Arc::new(tokio::sync::Notify::new()),
            },
            BatchRecord {
                sequence_index: 20,
                snapshot_sequence_index: 0,
                kv_index: 0,
                phase: Phase::PrefillBegin,
                prompt_length: 20,
                notify: Arc::new(tokio::sync::Notify::new()),
            },
            BatchRecord {
                sequence_index: 100,
                snapshot_sequence_index: 0,
                kv_index: 50,
                phase: Phase::Decode,
                prompt_length: 100,
                notify: Arc::new(tokio::sync::Notify::new()),
            },
        ]
        .into_boxed_slice();

        let mut batch_list = BatchList {
            records: batch_records,
            current_size: 3,
        };

        let mut task_list = TaskList {
            tasks: vec![ThreadTask {
                slices: vec![
                    SequenceSlice {
                        batch_index: 0,
                        sequence_index: 0,
                        length: 0,
                    };
                    4
                ]
                .into_boxed_slice(),
                current_size: 0,
            }]
            .into_boxed_slice(),
            current_size: 1,
            max_token_size: 4,
        };

        let (token_count, decode_count) = schedule_batch(&mut batch_list, &mut task_list);

        assert_eq!(decode_count, 2);
        assert_eq!(token_count, 2);
        assert_eq!(batch_list.records[0].kv_index, 11);
        assert_eq!(batch_list.records[1].kv_index, 0);
        assert_eq!(batch_list.records[2].kv_index, 51);
    }

    #[test]
    fn test_schedule_batch_prefill_only() {
        let batch_records = vec![BatchRecord {
            sequence_index: 10,
            snapshot_sequence_index: 0,
            kv_index: 5,
            phase: Phase::PrefillEnd,
            prompt_length: 10,
            notify: Arc::new(tokio::sync::Notify::new()),
        }]
        .into_boxed_slice();

        let mut batch_list = BatchList {
            records: batch_records,
            current_size: 1,
        };

        let mut task_list = TaskList {
            tasks: vec![ThreadTask {
                slices: vec![
                    SequenceSlice {
                        batch_index: 0,
                        sequence_index: 0,
                        length: 0,
                    };
                    8
                ]
                .into_boxed_slice(),
                current_size: 0,
            }]
            .into_boxed_slice(),
            current_size: 1,
            max_token_size: 8,
        };

        let (token_count, decode_count) = schedule_batch(&mut batch_list, &mut task_list);

        assert_eq!(decode_count, 0);
        assert_eq!(token_count, 5);
        assert_eq!(batch_list.records[0].snapshot_sequence_index, 10);
        assert_eq!(batch_list.records[0].kv_index, 10);
    }

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

        let batch_records = (0..batch_size)
            .map(|_| BatchRecord {
                sequence_index: 50,
                snapshot_sequence_index: 0,
                kv_index: 0,
                phase: Phase::PrefillBegin,
                prompt_length: 50,
                notify: Arc::new(tokio::sync::Notify::new()),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let mut batch_list = BatchList {
            records: batch_records,
            current_size: batch_size,
        };

        let batch_ptr = MutPtr {
            ptr: &mut batch_list as *mut BatchList,
        };

        start(batch_ptr, output_tensor.operator_queue.take());
    }
}
