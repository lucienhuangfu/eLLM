use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::num_traits::{exp::Exp, neg_infinity::NegInfinity, sigmoid::Sigmoid, sqrt::Sqrt};
use crate::operators::operator::Operator;
use crate::runtime::scheduler::{ScheduleTask, Scheduler};

use super::sync::{AdaptiveWait, SpinBarrier};

pub struct ExecutorPool<T> {
    pub scheduler: Arc<Scheduler>,
    operator_queue: Arc<[Operator<T>]>,
    pub thread_num: usize,
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
        _chunk_size: usize,
        _timeout: Duration,
    ) -> Self {
        Self {
            scheduler,
            operator_queue: operator_queue.into(),
            thread_num: thread_num.max(1),
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn with_thread_count(mut self, thread_num: usize) -> Self {
        self.thread_num = thread_num.max(1);
        self
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
                .name(format!("executor-worker-{thread_id}"))
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
                .expect("failed to spawn executor worker thread");
        }
    }

    pub fn scheduler(&self) -> Arc<Scheduler> {
        Arc::clone(&self.scheduler)
    }

    pub fn shutdown_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.shutdown)
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
                    wait.wait(|| shutdown.load(Ordering::Acquire) || scheduler.has_work());
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
        let prefill_list = &task.prefilling_chunked_slices;
        let decode_list = &task.slices;

        for operator in operator_queue.iter() {
            let batch_list_ptr = scheduler.batch_list().get();
            unsafe {
                let batch_list = &mut *batch_list_ptr;
                operator.run(
                    prefill_size,
                    decode_size,
                    thread_num,
                    thread_id,
                    prefill_list,
                    decode_list,
                    batch_list,
                );
            }

            barrier.wait();
        }
    }

    #[cfg(test)]
    fn execute_batch(
        scheduler: &Scheduler,
        operator_queue: &[Operator<T>],
        barrier: &SpinBarrier,
        thread_num: usize,
        thread_id: usize,
        task: &ScheduleTask,
    ) {
        Self::execute_operators(
            scheduler,
            operator_queue,
            barrier,
            thread_num,
            thread_id,
            task,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::fake_echo::FakeEcho;
    use crate::operators::operator::Operator;
    use crate::operators::send_sync_ptr::SharedMut;
    use crate::runtime::batch::SequenceSlice;
    use crate::runtime::session::SlotState;

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
            SlotState::new_decode_state(0, 0),
            SlotState::new_start_state(),
            SlotState::new_decode_state(2, 2),
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
    fn execute_batch_runs_fake_echo_across_threads() {
        let thread_num = 2usize;
        let sequence_stride = 16usize;
        let eos_id = 999usize;

        let mut sequences = vec![0usize; 2 * sequence_stride];
        sequences[0] = 11;
        sequences[sequence_stride] = 21;

        let fake_echo = FakeEcho::new(sequences.as_mut_ptr(), sequence_stride, eos_id);
        let operator_queue = vec![Operator::<f32>::FakeEcho(fake_echo)];

        let batch_list = Arc::new(SharedMut::new(vec![
            SlotState::new_decode_state(0, 0),
            SlotState::new_decode_state(0, 0),
        ]));
        let scheduler = Arc::new(Scheduler::new(8, 8, thread_num, Arc::clone(&batch_list)));

        let mut decode_list = Vec::with_capacity(2);
        decode_list.push(SequenceSlice {
            batch_index: 0,
            sequence_index: 0,
            token_start_index: 0,
            length: 1,
            last_token_flag: true,
        });
        decode_list.push(SequenceSlice {
            batch_index: 1,
            sequence_index: 0,
            token_start_index: 1,
            length: 1,
            last_token_flag: true,
        });

        scheduler.with_task_mut(|task| {
            task.decode_size = 2;
            task.prefill_size = 0;
            task.prefilling_chunked_slices.resize_with(thread_num, || Vec::new());
            task.slices = decode_list;
        });

        let barrier = Arc::new(SpinBarrier::new(thread_num));
        let mut handles = Vec::with_capacity(thread_num);

        for thread_id in 0..thread_num {
            let scheduler = Arc::clone(&scheduler);
            let operator_queue = operator_queue.clone();
            let barrier = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                scheduler.with_task(|task| {
                    ExecutorPool::<f32>::execute_batch(
                        &scheduler,
                        &operator_queue,
                        &barrier,
                        thread_num,
                        thread_id,
                        task,
                    );
                });
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(sequences[1], 11);
        assert_eq!(sequences[sequence_stride + 1], 21);

        scheduler.batch_list().with(|batch_list| {
            assert_eq!(batch_list[0].phase, crate::runtime::session::Phase::Decode);
            assert_eq!(batch_list[1].phase, crate::runtime::session::Phase::Decode);
        });
    }

    #[test]
    fn execute_batch_fake_echo_can_reach_eos() {
        let thread_num = 1usize;
        let sequence_stride = 128usize;
        let eos_id = 151643usize;

        let mut sequences = vec![0usize; sequence_stride];
        sequences[98] = 42;

        let fake_echo = FakeEcho::new(sequences.as_mut_ptr(), sequence_stride, eos_id);
        let operator_queue = vec![Operator::<f32>::FakeEcho(fake_echo)];

        let batch_list = Arc::new(SharedMut::new(vec![SlotState::new_decode_state(98, 98)]));
        let scheduler = Scheduler::new(8, 8, thread_num, Arc::clone(&batch_list));

        let mut decode_list = Vec::with_capacity(1);
        decode_list.push(SequenceSlice {
            batch_index: 0,
            sequence_index: 98,
            token_start_index: 0,
            length: 1,
            last_token_flag: true,
        });

        scheduler.with_task_mut(|task| {
            task.decode_size = 1;
            task.prefill_size = 0;
            task.prefilling_chunked_slices.resize_with(thread_num, || Vec::new());
            task.slices = decode_list;
        });

        let barrier = SpinBarrier::new(thread_num);
        scheduler.with_task(|task| {
            ExecutorPool::<f32>::execute_batch(
                &scheduler,
                &operator_queue,
                &barrier,
                thread_num,
                0,
                task,
            );
        });

        assert_eq!(sequences[99], eos_id);
        scheduler.batch_list().with(|batch_list| {
            assert_eq!(batch_list[0].phase, crate::runtime::session::Phase::Eos);
            assert_eq!(batch_list[0].sequence_index, 100);
            assert_eq!(batch_list[0].filling_length, 0);
        });
    }

    #[test]
    fn scheduler_work_tracking() {
        let batch_list = Arc::new(SharedMut::new(vec![SlotState::new_start_state()]));
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
    fn executor_pool_new_clamps_thread_count_to_at_least_one() {
        let batch_list = Arc::new(SharedMut::new(vec![SlotState::new_start_state()]));
        let scheduler = Arc::new(Scheduler::new(8, 1, 1, batch_list));
        let executor =
            ExecutorPool::<f32>::new(Vec::new(), scheduler, 0, 1, Duration::from_millis(1));

        assert_eq!(executor.thread_num, 1);
    }

    #[test]
    fn scheduler_accessor_returns_same_arc() {
        let batch_list = Arc::new(SharedMut::new(vec![SlotState::new_start_state()]));
        let scheduler = Arc::new(Scheduler::new(8, 1, 1, batch_list));
        let executor = ExecutorPool::<f32>::new(
            Vec::new(),
            Arc::clone(&scheduler),
            1,
            1,
            Duration::from_millis(1),
        );

        let returned = executor.scheduler();
        assert!(Arc::ptr_eq(&scheduler, &returned));
    }

    #[test]
    fn executor_pool_uses_larger_thread_num_and_chunk_size_in_default_scheduler() {
        let batch_list = Arc::new(SharedMut::new(vec![
            SlotState::new_decode_state(0, 0),
            SlotState::new_decode_state(1, 1),
            SlotState::new_decode_state(2, 2),
            SlotState::new_decode_state(3, 3),
            SlotState::new_prefill_state(10, 500),
            SlotState::new_prefill_state(20, 500),
        ]));
        let scheduler = Arc::new(Scheduler::new(8, 2048, 8, batch_list));

        let executor = ExecutorPool::<f32>::new(
            Vec::new(),
            Arc::clone(&scheduler),
            8,
            2048,
            Duration::from_millis(1),
        );

        assert_eq!(executor.thread_num, 8);

        let scheduled = executor.scheduler.schedule_batch();
        assert!(scheduled);

        executor.scheduler.with_task(|task| {
            assert_eq!(task.decode_size, 4);
            assert_eq!(task.prefilling_chunked_slices.len(), 8);
            assert!(task.prefill_size <= 2048);
            assert_eq!(task.slices.len(), 6);
        });
    }

    #[test]
    fn executor_pool_with_thread_count_supports_large_values() {
        let batch_list = Arc::new(SharedMut::new(
            (0..2).map(|_| SlotState::new_start_state()).collect(),
        ));
        let scheduler = Arc::new(Scheduler::new(8, 64, 2, batch_list));

        let executor =
            ExecutorPool::<f32>::new(Vec::new(), scheduler, 2, 64, Duration::from_millis(1))
                .with_thread_count(16);

        assert_eq!(executor.thread_num, 16);
    }

    #[tokio::test]
    async fn executor_pool_start_runs_end_to_end_once_and_can_shutdown() {
        let sequence_stride = 32usize;
        let eos_id = 1000usize;

        let mut sequences = vec![0usize; sequence_stride];
        sequences[0] = 7;

        let fake_echo = FakeEcho::new(sequences.as_mut_ptr(), sequence_stride, eos_id);
        let operator_queue = vec![Operator::<f32>::FakeEcho(fake_echo)];

        let batch_list = Arc::new(SharedMut::new(vec![SlotState::new_decode_state(0, 0)]));
        let scheduler = Arc::new(Scheduler::new(8, 64, 2, batch_list));

        let executor =
            ExecutorPool::<f32>::new(operator_queue, scheduler, 2, 64, Duration::from_millis(1));

        let shutdown = executor.shutdown_handle();
        executor.start();

        for _ in 0..100 {
            if sequences[1] == 7 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        assert_eq!(sequences[1], 7);
        shutdown.store(true, Ordering::Release);
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}
