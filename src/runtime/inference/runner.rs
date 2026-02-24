use core_affinity;
use std::cell::SyncUnsafeCell;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::sync::Arc;
use std::sync::Barrier;
use std::thread;

use crate::runtime::operator::Operator;

use crate::common::num_traits::{
    exp::Exp, neg_infinity::NegInfinity, sigmoid::Sigmoid, sqrt::Sqrt,
};
use crate::runtime::inference::state::{Phase, SequenceState};
use crate::runtime::inference::scheduler::BatchScheduler;

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
    T: Copy
        + Default
        + Sub<Output = T>
        + Neg<Output = T>
        + AddAssign
        + Exp
        + Sqrt
        + NegInfinity
        + Sigmoid
        + PartialOrd
        + 'static,
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

        struct SharedState {
            sizes: (usize, usize),
            scheduler: BatchScheduler,
        }

        let barrier = Arc::new(Barrier::new(thread_num));
        let shared_state = Arc::new(SyncUnsafeCell::new(SharedState {
            sizes: (0, 0),
            scheduler: self.batch_scheduler,
        }));
        let mut handles = Vec::with_capacity(thread_num);
        // let core_ids = core_affinity::get_core_ids().unwrap();

        for (i, core_id) in core_ids.into_iter().enumerate() {
            // println!("thread id {}", i);
            let b = Arc::clone(&barrier);
            let queue = Arc::clone(&sync_operator_queue);
            let shared_state: Arc<SyncUnsafeCell<SharedState>> = Arc::clone(&shared_state);

            let handle = thread::spawn(move || {
                let thread_id = i;
                core_affinity::set_for_current(core_id);
                println!("{} start", thread_id);
                let state_ptr = shared_state.get();

                // Main inference loop: continuously processes batches of tokens
                loop {
                    // Thread 0 acts as the scheduler: monitors user states and prepares the token batch
                    if thread_id == 0 {
                        unsafe {
                            let state = &mut *state_ptr;
                            state.sizes = state.scheduler.schedule_batch();
                        }
                    }

                    // Synchronization barrier: Wait for Thread 0 to finish scheduling
                    b.wait();

                    let (prefill_size, decode_size, prefill_list, decode_list, batch_list_guard) = unsafe {
                        let state = &mut *state_ptr;
                        let (prefill_size, decode_size) = state.sizes;
                        let scheduler = &mut state.scheduler;
                        (
                            prefill_size,
                            decode_size,
                            &scheduler.prefill_list,
                            &scheduler.decode_list,
                            &mut *scheduler.batch_list.get(),
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
                            batch_list_guard,
                        );
                        b.wait();
                    }
                }

                // let t = s.elapsed();
                // println!("thread {} decode time {:?}", thread_id, t);
            });

            // std::mem_mgr::forget(handle);
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
    use crate::runtime::inference::state::SequenceState;
    use crate::mem_mgr::cache::Cache;
    use crate::qwen3_moe::sparse_moe_block::SparseMoeBlock;
    use crate::runtime::tensor::{Tensor, TensorCtx};

    // use crate::mem_mgr::allocator::allocate_init;

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
        let ctx = Rc::new(TensorCtx::new(cache, operator_queue));

        let sparse_moe = SparseMoeBlock::<f32>::new(
            // position_window_size,
            hidden_size,
            intermediate_size,
            num_experts,
            top_k,
            norm_topk_prob,
            "model.layers.0",
            ctx.clone(),
        );

        let shape = vec![position_window_size, batch_size, hidden_size];
        let input = ctx.tensor(shape.clone(), String::from("model.layers.0.input_tensor"));

        let residual = ctx.tensor(
            shape.clone(),
            String::from("model.layers.0.residual_tensor"),
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
        {
            let mut batch_list = unsafe { &mut *batch_scheduler.batch_list.get() };
            batch_list.extend((0..batch_size).map(|_| SequenceState {
                sequence_index: 50,
                kv_index: 0,
                phase: Phase::Prefill,
                // prompt_length: 50,
                notify: std::sync::Arc::new(tokio::sync::Notify::new()),
            }));
        }

        ServingRunner::new(output_tensor.operator_queue.take(), batch_scheduler).start();
    }
}

