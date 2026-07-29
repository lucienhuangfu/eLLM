use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::num_traits::{exp::Exp, neg_infinity::NegInfinity, sigmoid::Sigmoid, sqrt::Sqrt};
use crate::operators::operator::Operator;
use crate::runtime::scheduler::{ScheduleTask, Scheduler};

use super::sync::{AdaptiveWait, SpinBarrier};

pub struct ExecutorPool<T> {
    scheduler: Arc<Scheduler>,
    operator_queue: Arc<[Operator<T>]>,
    thread_num: usize,
    shutdown: Arc<AtomicBool>,
}

impl<T> ExecutorPool<T>
where
    T: Copy
        + Default
        + std::ops::Sub<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::AddAssign
        + Exp
        + Sqrt
        + NegInfinity
        + Sigmoid
        + PartialOrd
        + Send
        + Sync
        + 'static,
{
    pub fn new(
        operator_queue: Vec<Operator<T>>,
        scheduler: Arc<Scheduler>,
        thread_num: usize,
    ) -> Self {
        Self {
            scheduler,
            operator_queue: operator_queue.into(),
            thread_num: thread_num.max(1),
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn start(self) {
        let barrier = Arc::new(SpinBarrier::new(self.thread_num));

        for thread_id in 0..self.thread_num {
            let scheduler = Arc::clone(&self.scheduler);
            let operator_queue = Arc::clone(&self.operator_queue);
            let barrier = Arc::clone(&barrier);
            let thread_num = self.thread_num;
            let shutdown = Arc::clone(&self.shutdown);

            std::thread::Builder::new()
                .name(format!("worker-{thread_id}"))
                .spawn(move || {
                    Self::run_worker(
                        scheduler,
                        operator_queue.as_ref(),
                        &barrier,
                        thread_num,
                        thread_id,
                        shutdown,
                    );
                })
                .expect("failed to spawn worker thread");
        }
    }

    fn run_worker(
        scheduler: Arc<Scheduler>,
        operator_queue: &[Operator<T>],
        barrier: &SpinBarrier,
        thread_num: usize,
        thread_id: usize,
        shutdown: Arc<AtomicBool>,
    ) {
        let mut wait = AdaptiveWait::default();

        while !shutdown.load(Ordering::Acquire) {
            if thread_id == 0 {
                if !scheduler.schedule_batch() {
                    wait.wait(|| shutdown.load(Ordering::Acquire) || scheduler.schedule_batch());
                    continue;
                }
            } else {
                wait.wait(|| shutdown.load(Ordering::Acquire) || scheduler.has_work());
            }

            if shutdown.load(Ordering::Acquire) {
                break;
            }

            barrier.wait();
            scheduler.with_task(|task| {
                Self::execute_operators(
                    &scheduler,
                    operator_queue,
                    barrier,
                    thread_num,
                    thread_id,
                    task,
                );
            });
            barrier.wait();

            if thread_id == 0 {
                scheduler.with_task_mut(|task| {
                    task.reset();
                });
            }
        }
    }

    #[inline]
    fn execute_operators(
        scheduler: &Scheduler,
        operator_queue: &[Operator<T>],
        barrier: &SpinBarrier,
        thread_num: usize,
        thread_id: usize,
        task: &ScheduleTask,
    ) {
        let prefill_size = task.prefill_size;
        let decode_size = task.decode_size;
        for operator in operator_queue.iter() {
            let slot_list_ptr = scheduler.slot_list().get();
            unsafe {
                let slot_list = &mut *slot_list_ptr;
                operator.run(
                    prefill_size,
                    decode_size,
                    thread_num,
                    thread_id,
                    &task.slices,
                    slot_list,
                );
            }

            barrier.wait();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::send_sync_ptr::SharedMut;
    use crate::runtime::session::{Phase, SlotState};

    fn decode_state(next_sequence_index: usize) -> SlotState {
        let mut s = SlotState::idle();
        s.start_decode(next_sequence_index, next_sequence_index);
        s
    }

    fn prefill_state(next_sequence_index: usize, filling_length: usize) -> SlotState {
        let mut s = SlotState::idle();
        s.start_prefill(next_sequence_index, filling_length);
        s
    }

    #[test]
    fn schedule_batch_returns_none_when_no_work_exists() {
        let batch_list = Arc::new(SharedMut::new(Vec::<SlotState>::new()));
        let scheduler = Arc::new(Scheduler::new(0, 0, 1, batch_list));

        let has_work = scheduler.schedule_batch();

        assert!(!has_work);
    }

    #[test]
    fn schedule_batch_builds_decode_plan_from_active_slots() {
        let batch_list = Arc::new(SharedMut::new(vec![
            decode_state(0),
            SlotState::idle(),
            decode_state(2),
        ]));
        let scheduler = Arc::new(Scheduler::new(32, 1024, 1, batch_list));

        let has_work = scheduler.schedule_batch();

        assert!(has_work);

        scheduler.with_task(|task| {
            assert_eq!(task.decode_size, 2);
            assert_eq!(task.slices.len(), 2);
            assert_eq!(task.slices[0].batch_index, 0);
            assert_eq!(task.slices[1].batch_index, 2);
        });
    }

    #[test]
    fn scheduler_work_tracking() {
        let batch_list = Arc::new(SharedMut::new(vec![SlotState::idle()]));
        let scheduler = Scheduler::new(8, 8, 1, batch_list);

        assert!(!scheduler.has_work());

        scheduler.with_task_mut(|task| {
            task.prefill_size = 1;
        });
        assert!(scheduler.has_work());

        scheduler.with_task(|task| {
            assert_eq!(task.prefill_size, 1);
            assert_eq!(task.decode_size, 0);
        });

        scheduler.with_task_mut(|task| {
            task.reset();
        });

        assert!(!scheduler.has_work());
    }

    #[test]
    fn worker_pool_new_clamps_thread_count_to_at_least_one() {
        let batch_list = Arc::new(SharedMut::new(vec![SlotState::idle()]));
        let scheduler = Arc::new(Scheduler::new(8, 1, 1, batch_list));
        let executor = ExecutorPool::<f32>::new(Vec::new(), scheduler, 0);

        assert_eq!(executor.thread_num, 1);
    }

    #[test]
    fn worker_pool_uses_larger_thread_num_and_chunk_size_in_default_scheduler() {
        let batch_list = Arc::new(SharedMut::new(vec![
            decode_state(0),
            decode_state(1),
            decode_state(2),
            decode_state(3),
            prefill_state(10, 500),
            prefill_state(20, 500),
        ]));
        let scheduler = Arc::new(Scheduler::new(8, 2048, 8, batch_list));

        let executor = ExecutorPool::<f32>::new(Vec::new(), Arc::clone(&scheduler), 8);

        assert_eq!(executor.thread_num, 8);

        let scheduled = executor.scheduler.schedule_batch();
        assert!(scheduled);

        executor.scheduler.with_task(|task| {
            assert_eq!(task.decode_size, 4);
            assert!(task.prefill_size <= 2048);
            assert_eq!(task.slices.len(), 6);
        });
    }
}
