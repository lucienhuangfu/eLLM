use core_affinity;
use std::cell::SyncUnsafeCell;
use std::ops::{AddAssign, Neg, Sub};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::Barrier;
use std::thread;

use crate::operators::operator::Operator;

use crate::common::num_traits::{
    exp::Exp, neg_infinity::NegInfinity, sigmoid::Sigmoid, sqrt::Sqrt,
};
use crate::runtime::BatchScheduler;

/// Runs the inference serving loop.
///
/// This initializes a thread pool where Thread 0 schedules tasks by monitoring
/// user request phases (Prefill/Decode) and populating the token list. All threads
/// then synchronize to execute the operators in the queue for the current batch.
pub struct ServingRunner<T> {
    operator_queue: Vec<Operator<T>>,
    batch_scheduler: BatchScheduler,
    stop_flag: Arc<AtomicBool>,
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
        + Send
        + Sync
        + 'static,
{
    pub fn new(operator_queue: Vec<Operator<T>>, batch_scheduler: BatchScheduler) -> Self {
        Self {
            operator_queue,
            batch_scheduler,
            stop_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn start(self) {
        let core_ids = core_affinity::get_core_ids().unwrap_or_default();
        let thread_num = core_ids.len().max(1).min(self.batch_scheduler.thread_num());

        let operator_queue: Arc<[Operator<T>]> = self.operator_queue.into();

        let barrier = Arc::new(Barrier::new(thread_num));
        let shared_sizes = Arc::new(SyncUnsafeCell::new((0usize, 0usize)));
        let shared_scheduler = Arc::new(SyncUnsafeCell::new(self.batch_scheduler));
        let stop_flag = self.stop_flag;

        let mut handles = Vec::with_capacity(thread_num);

        for thread_id in 0..thread_num {
            let barrier = Arc::clone(&barrier);
            let queue = Arc::clone(&operator_queue);
            let shared_sizes = Arc::clone(&shared_sizes);
            let shared_scheduler = Arc::clone(&shared_scheduler);
            let stop_flag = Arc::clone(&stop_flag);
            let core_id = core_ids.get(thread_id).copied();

            let handle = thread::spawn(move || {
                if let Some(core_id) = core_id {
                    core_affinity::set_for_current(core_id);
                }

                let sizes_ptr = shared_sizes.get();
                let scheduler_ptr = shared_scheduler.get();

                loop {
                    if stop_flag.load(Ordering::Relaxed) {
                        break;
                    }

                    if thread_id == 0 {
                        unsafe {
                            let scheduler = &mut *scheduler_ptr;
                            *sizes_ptr = scheduler.schedule_batch();
                        }
                    }

                    barrier.wait();

                    let (prefill_size, decode_size) = unsafe { *sizes_ptr };
                    let (prefill_list, decode_list, batch_list) = unsafe {
                        let scheduler = &mut *scheduler_ptr;
                        (
                            &scheduler.prefill_list,
                            &scheduler.decode_list,
                            &mut *scheduler.batch_list.get(),
                        )
                    };

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
                        barrier.wait();
                    }

                    if thread_id == 0 {
                        let all_eos = batch_list
                            .iter()
                            .all(|s| matches!(s.phase, crate::runtime::Phase::Eos));
                        if all_eos && !batch_list.is_empty() {
                            stop_flag.store(true, Ordering::Relaxed);
                        }
                    }
                    barrier.wait();
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ServingRunner;
    use crate::runtime::BatchScheduler;

    #[test]
    fn new_preserves_operator_queue_and_scheduler_layout() {
        let operator_queue = Vec::<crate::operators::operator::Operator<f32>>::new();
        let batch_scheduler = BatchScheduler::new(16, 4, 16, 3);

        let runner = ServingRunner::new(operator_queue, batch_scheduler);

        assert_eq!(runner.operator_queue.len(), 0);
        assert_eq!(runner.batch_scheduler.prefill_list.len(), 3);
        assert_eq!(runner.batch_scheduler.prefill_list[0].capacity(), 4);
        assert_eq!(runner.batch_scheduler.decode_list.len(), 0);
    }
}
