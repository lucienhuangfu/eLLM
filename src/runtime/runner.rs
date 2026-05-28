use core_affinity;
use std::cell::SyncUnsafeCell;
use std::ops::{AddAssign, Neg, Sub};
use std::sync::Arc;
use std::sync::Barrier;
use std::thread;

use crate::operators::operator::Operator;

use crate::num_traits::{exp::Exp, neg_infinity::NegInfinity, sigmoid::Sigmoid, sqrt::Sqrt};
use crate::runtime::BatchScheduler;

/// Runs the inference loop.
///
/// This initializes a thread pool where Thread 0 schedules tasks by monitoring
/// user request phases (Prefill/Decode) and populating the token list. All threads
/// then synchronize to execute the operators in the queue for the current batch.
pub struct Runner<T> {
    operator_queue: Vec<Operator<T>>,
    batch_scheduler: BatchScheduler,
}

impl<T> Runner<T>
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
        }
    }

    pub fn start(self) {
        let core_ids = core_affinity::get_core_ids().unwrap_or_default();
        let thread_num = core_ids.len().max(1).min(self.batch_scheduler.thread_num());

        let operator_queue: Arc<[Operator<T>]> = self.operator_queue.into();

        let barrier = Arc::new(Barrier::new(thread_num));
        let shared_round = Arc::new(SyncUnsafeCell::new(
            crate::runtime::BatchScheduleStep::default(),
        ));
        let shared_scheduler = Arc::new(SyncUnsafeCell::new(self.batch_scheduler));

        let mut handles = Vec::with_capacity(thread_num);

        for thread_id in 0..thread_num {
            let barrier = Arc::clone(&barrier);
            let queue = Arc::clone(&operator_queue);
            let shared_round = Arc::clone(&shared_round);
            let shared_scheduler = Arc::clone(&shared_scheduler);
            let core_id = core_ids.get(thread_id).copied();

            let handle = thread::spawn(move || {
                if let Some(core_id) = core_id {
                    core_affinity::set_for_current(core_id);
                }

                let round_ptr = shared_round.get();
                let scheduler_ptr = shared_scheduler.get();

                loop {
                    if thread_id == 0 {
                        unsafe {
                            let scheduler = &mut *scheduler_ptr;
                            *round_ptr = scheduler.schedule_batch_step();
                        }
                    }

                    barrier.wait();

                    let round = unsafe { *round_ptr };
                    if round.stop_requested {
                        break;
                    }

                    let prefill_size = round.prefill_tokens;
                    let decode_size = round.decode_tokens;
                    let continuous_service = unsafe { (&*scheduler_ptr).is_continuous_service() };
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
                        let all_done = batch_list
                            .iter()
                            .all(|s| matches!(s.phase, crate::runtime::Phase::Eos));
                        if !continuous_service && all_done && !batch_list.is_empty() {
                            unsafe {
                                (*round_ptr).stop_requested = true;
                            }
                        }
                    }
                    barrier.wait();

                    if unsafe { (*round_ptr).stop_requested } {
                        break;
                    }
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
    use super::Runner;
    use crate::runtime::BatchScheduler;

    #[test]
    fn new_preserves_operator_queue_and_scheduler_layout() {
        let operator_queue = Vec::<crate::operators::operator::Operator<f32>>::new();
        let batch_scheduler = BatchScheduler::new(16, 4, 16, 3);

        let runner = Runner::new(operator_queue, batch_scheduler);

        assert_eq!(runner.operator_queue.len(), 0);
        assert_eq!(runner.batch_scheduler.prefill_list.len(), 3);
        assert_eq!(runner.batch_scheduler.prefill_list[0].capacity(), 4);
        assert_eq!(runner.batch_scheduler.decode_list.len(), 0);
    }
}
