use core_affinity;
use std::cell::SyncUnsafeCell;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::sync::Arc;
use std::sync::Barrier;
use std::thread;
use std::time::Instant;

use super::super::compiler::operator::Operator;

use super::super::kernel::generic::{exp::Exp, neg_infinity::NegInfinity, sqrt::Sqrt};
use crate::init::record::{BatchRecord, Phase};
use crate::init::send_sync_ptr::MutPtr;
use crate::serving::schedule::BatchScheduler;

/// Runs the inference serving loop.
///
/// This initializes a thread pool where Thread 0 schedules tasks by monitoring
/// user request phases (Prefill/Decode) and populating the token list. All threads
/// then synchronize to execute the operators in the queue for the current batch.
pub struct ServingRunner<T> {
    batch_list_ptr: MutPtr<Vec<BatchRecord>>,
    operator_queue: Vec<Operator<T>>,
}

impl<T> ServingRunner<T>
where
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
    pub fn new(batch_list_ptr: MutPtr<Vec<BatchRecord>>, operator_queue: Vec<Operator<T>>) -> Self {
        Self {
            batch_list_ptr,
            operator_queue,
        }
    }

    pub fn start(self) {
        println!("start");
        // let thread_num = thread::available_parallelism().unwrap().get();
        let core_ids = core_affinity::get_core_ids().unwrap();
        let thread_num = core_ids.len();

        // Optimization: Convert Vec to Arc<[T]> to reduce one level of indirection compared to Arc<Vec<T>>
        let sync_operator_queue: Arc<[Operator<T>]> = self.operator_queue.into();

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
            let batch_list_ptr_addr = SyncUnsafeCell::new(self.batch_list_ptr);
            // let last_prefill_list_ptr_addr = SyncUnsafeCell::new(last_prefill_list_ptr);

            let handle = thread::spawn(move || {
                let thread_id = i;
                core_affinity::set_for_current(core_id);
                println!("{} start", thread_id);
                let s = Instant::now();
                let sizes_ptr = shared_sizes.get();
                let mut scheduler = if thread_id == 0 {
                    let batch_size = unsafe { (&*(*batch_list_ptr_addr.get()).ptr).len() };
                    Some(BatchScheduler::new(thread_num, batch_size))
                } else {
                    None
                };

                // Main inference loop: continuously processes batches of tokens
                loop {
                    // Thread 0 acts as the scheduler: monitors user states and prepares the token batch
                    if thread_id == 0 {
                        unsafe {
                            let batch_list = &mut *(*batch_list_ptr_addr.get()).ptr;
                            *sizes_ptr = scheduler.as_mut().unwrap().schedule_batch(batch_list);
                        }
                    }

                    // Synchronization barrier: Wait for Thread 0 to finish scheduling
                    b.wait();

                    let (prefill_size, decode_size) = unsafe { *sizes_ptr };

                    // Execute the operator queue in parallel
                    for operator in queue.iter() {
                        operator.run(prefill_size, decode_size, thread_num, thread_id);
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

        let mut batch_list = batch_records.into_vec();

        let batch_ptr = MutPtr {
            ptr: &mut batch_list as *mut Vec<BatchRecord>,
        };

        ServingRunner::new(batch_ptr, output_tensor.operator_queue.take()).start();
    }
}
