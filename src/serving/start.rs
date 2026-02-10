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
use crate::serving::schedule::BatchScheduler;

/// Runs the inference serving loop.
///
/// This initializes a thread pool where Thread 0 schedules tasks by monitoring
/// user request phases (Prefill/Decode) and populating the token list. All threads
/// then synchronize to execute the operators in the queue for the current batch.
pub struct ServingRunner<T> {
    operator_queue: Vec<Operator<T>>,
    batch_scheduler: BatchScheduler,
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
    pub fn new(operator_queue: Vec<Operator<T>>, batch_scheduler: BatchScheduler) -> Self {
        Self {
            operator_queue,
            batch_scheduler,
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
        let shared_scheduler = Arc::new(SyncUnsafeCell::new(self.batch_scheduler));
        let mut handles = Vec::with_capacity(thread_num);
        // let core_ids = core_affinity::get_core_ids().unwrap();

        for (i, core_id) in core_ids.into_iter().enumerate() {
            // println!("thread id {}", i);
            let b = Arc::clone(&barrier);
            let queue = Arc::clone(&sync_operator_queue);
            let shared_sizes: Arc<SyncUnsafeCell<(usize, usize)>> = Arc::clone(&shared_sizes);
            let shared_scheduler: Arc<SyncUnsafeCell<BatchScheduler>> =
                Arc::clone(&shared_scheduler);

            let handle = thread::spawn(move || {
                let thread_id = i;
                core_affinity::set_for_current(core_id);
                println!("{} start", thread_id);
                let s = Instant::now();
                let sizes_ptr = shared_sizes.get();
                let scheduler_ptr = shared_scheduler.get();

                // Main inference loop: continuously processes batches of tokens
                loop {
                    // Thread 0 acts as the scheduler: monitors user states and prepares the token batch
                    if thread_id == 0 {
                        unsafe {
                            let scheduler = &mut *scheduler_ptr;
                            *sizes_ptr = scheduler.schedule_batch();
                        }
                    }

                    // Synchronization barrier: Wait for Thread 0 to finish scheduling
                    b.wait();

                    let (prefill_size, decode_size) = unsafe { *sizes_ptr };

                    let (prefill_list, decode_list, batch_list) = unsafe {
                        let scheduler = &mut *scheduler_ptr;
                        (
                            &scheduler.prefill_list,
                            &scheduler.decode_list,
                            &mut scheduler.batch_list,
                        )
                    };

                    // Execute the operator queue in parallel
                    for operator in queue.iter() {
                        operator.run(
                            prefill_size,
                            decode_size,
                            thread_num,
                            thread_id,
                            prefill_list,
                            decode_list,
                            batch_list,
                        );
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

        let thread_num = core_affinity::get_core_ids().unwrap().len();
        let mut batch_scheduler = BatchScheduler::new(position_window_size, batch_size, thread_num);
        batch_scheduler.batch_list = (0..batch_size)
            .map(|_| BatchRecord {
                sequence_index: 50,
                snapshot_sequence_index: 0,
                kv_index: 0,
                phase: Phase::PrefillBegin,
                prompt_length: 50,
                notify: tokio::sync::Notify::new(),
            })
            .collect();

        ServingRunner::new(output_tensor.operator_queue.take(), batch_scheduler).start();
    }
}
