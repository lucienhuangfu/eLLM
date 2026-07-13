use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use crate::num_traits::{exp::Exp, neg_infinity::NegInfinity, sigmoid::Sigmoid, sqrt::Sqrt};
use crate::operators::operator::Operator;
use crate::runtime::executor::sync::{AdaptiveWait, SpinBarrier};
use crate::runtime::scheduler::{DefaultSchedulerStrategy, ScheduleTask, SchedulerStrategy};
use crate::runtime::state::shared::SharedState;

pub struct ExecutorPool<T> {
    shared_state: Arc<SharedState>,
    operator_queue: Arc<[Operator<T>]>,
    thread_num: usize,
    strategy: Arc<dyn SchedulerStrategy>,
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
        shared_state: Arc<SharedState>,
        thread_num: usize,
        chunk_size: usize,
        _timeout: Duration,
    ) -> Self {
        let batch_size = shared_state.batch_list.with(|list| list.len());
        let strategy = Arc::new(DefaultSchedulerStrategy::new(
            batch_size, chunk_size, thread_num,
        ));
        Self {
            shared_state,
            operator_queue: operator_queue.into(),
            thread_num: thread_num.max(1),
            strategy,
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn with_strategy(mut self, strategy: Arc<dyn SchedulerStrategy>) -> Self {
        self.strategy = strategy;
        self
    }

    pub fn with_thread_count(mut self, thread_num: usize) -> Self {
        self.thread_num = thread_num.max(1);
        self
    }

    pub fn start(self) {
        let barrier = Arc::new(SpinBarrier::new(self.thread_num));

        for thread_id in 0..self.thread_num {
            let shared_state = Arc::clone(&self.shared_state);
            let operator_queue = Arc::clone(&self.operator_queue);
            let barrier = Arc::clone(&barrier);
            let thread_num = self.thread_num;
            let strategy = Arc::clone(&self.strategy);
            let shutdown = Arc::clone(&self.shutdown);

            std::thread::Builder::new()
                .name(format!("executor-worker-{thread_id}"))
                .spawn(move || {
                    Self::run_worker(
                        shared_state,
                        operator_queue.as_ref(),
                        &barrier,
                        thread_num,
                        thread_id,
                        strategy,
                        shutdown,
                    );
                })
                .expect("failed to spawn executor worker thread");
        }
    }

    pub fn shared_state(&self) -> Arc<SharedState> {
        Arc::clone(&self.shared_state)
    }

    pub fn shutdown_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.shutdown)
    }

    fn run_worker(
        shared_state: Arc<SharedState>,
        operator_queue: &[Operator<T>],
        barrier: &SpinBarrier,
        thread_num: usize,
        thread_id: usize,
        strategy: Arc<dyn SchedulerStrategy>,
        shutdown: Arc<AtomicBool>,
    ) {
        let mut wait = AdaptiveWait::default();

        while !shutdown.load(Ordering::Acquire) {
            if thread_id == 0 {
                if !Self::try_schedule_next_round(&shared_state, strategy.as_ref()) {
                    wait.wait(|| shutdown.load(Ordering::Acquire) || shared_state.has_work());
                    continue;
                }
            } else {
                wait.wait(|| shutdown.load(Ordering::Acquire) || shared_state.has_work());
            }

            if shutdown.load(Ordering::Acquire) {
                break;
            }

            let task = shared_state
                .current_task()
                .expect("executor work became unavailable before execution");
            barrier.wait();
            Self::execute_operators(
                &shared_state,
                operator_queue,
                barrier,
                thread_num,
                thread_id,
                &task,
            );
            barrier.wait();

            if thread_id == 0 {
                shared_state.clear_work();
            }
        }
    }

    #[inline]
    fn try_schedule_next_round(
        shared_state: &SharedState,
        strategy: &dyn SchedulerStrategy,
    ) -> bool {
        let plan = shared_state
            .batch_list
            .with(|batch_list| strategy.plan_next_round(batch_list));
        if plan.is_empty() {
            return false;
        }

        shared_state.set_task(ScheduleTask::new(
            plan.prefill_size,
            plan.decode_size,
            plan.prefill_list,
            plan.decode_list,
            plan.task_id,
        ));
        true
    }

    #[inline]
    fn execute_operators(
        shared_state: &SharedState,
        operator_queue: &[Operator<T>],
        barrier: &SpinBarrier,
        thread_num: usize,
        thread_id: usize,
        task: &ScheduleTask,
    ) {
        let prefill_size = task.prefill_size;
        let decode_size = task.decode_size;
        let prefill_list = &task.prefill_list;
        let decode_list = &task.decode_list;

        for operator in operator_queue.iter() {
            let batch_list_ptr = shared_state.batch_list.get();
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
        shared_state: &SharedState,
        operator_queue: &[Operator<T>],
        barrier: &SpinBarrier,
        thread_num: usize,
        thread_id: usize,
        task: &ScheduleTask,
    ) {
        Self::execute_operators(
            shared_state,
            operator_queue,
            barrier,
            thread_num,
            thread_id,
            task,
        );
    }

    #[cfg(test)]
    fn schedule_batch(
        strategy: &dyn SchedulerStrategy,
        batch_list: &Arc<
            crate::operators::send_sync_ptr::SharedMut<Vec<crate::runtime::state::core::SlotState>>,
        >,
    ) -> Option<crate::runtime::scheduler::BatchPlan> {
        batch_list.with(|batch_list| {
            let plan = strategy.plan_next_round(batch_list);
            if plan.is_empty() {
                None
            } else {
                Some(plan)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use crate::operators::fake_echo::FakeEcho;
    use crate::operators::operator::Operator;
    use crate::operators::send_sync_ptr::SharedMut;
    use crate::runtime::scheduler::SchedulerStrategy;
    use crate::runtime::scheduler::{BatchMode, BatchPlan};
    use crate::runtime::state::sequence::{DecodeList, SequenceSlice};
    use crate::runtime::state::SlotState;

    struct EmptyStrategy;

    impl SchedulerStrategy for EmptyStrategy {
        fn plan_next_round(&self, _batch_list: &[SlotState]) -> BatchPlan {
            BatchPlan::new(0)
        }
    }

    struct SingleDecodeStrategy;

    impl SchedulerStrategy for SingleDecodeStrategy {
        fn plan_next_round(&self, batch_list: &[SlotState]) -> BatchPlan {
            let mut plan = BatchPlan::new(7);
            let mut decode_list = DecodeList::with_capacity(batch_list.len());

            for (batch_index, state) in batch_list.iter().enumerate() {
                if matches!(state.phase, crate::runtime::Phase::Decode) {
                    decode_list.push(SequenceSlice {
                        batch_index,
                        sequence_index: state.sequence_index,
                        token_start_index: decode_list.len(),
                        length: 1,
                        last_token_flag: true,
                    });
                }
            }

            plan.mode = BatchMode::Decode;
            plan.decode_size = decode_list.len();
            plan.decode_list = decode_list;
            plan
        }
    }

    #[test]
    fn schedule_batch_returns_none_when_no_work_exists() {
        let batch_list = Arc::new(SharedMut::new(Vec::<SlotState>::new()));
        let shared_state = SharedState::new(batch_list);
        let strategy = EmptyStrategy;

        let plan = ExecutorPool::<f32>::schedule_batch(&strategy, &shared_state.batch_list);

        assert!(plan.is_none());
    }

    #[test]
    fn schedule_batch_builds_decode_plan_from_active_slots() {
        let batch_list = Arc::new(SharedMut::new(vec![
            SlotState::new_decode_state(0, 0),
            SlotState::new_start_state(),
            SlotState::new_decode_state(2, 2),
        ]));
        let shared_state = SharedState::new(batch_list);
        let strategy = SingleDecodeStrategy;

        let plan = ExecutorPool::<f32>::schedule_batch(&strategy, &shared_state.batch_list)
            .expect("expected a non-empty plan");

        assert_eq!(plan.mode, BatchMode::Decode);
        assert_eq!(plan.decode_size, 2);
        assert_eq!(plan.decode_list.len(), 2);
        assert_eq!(plan.decode_list[0].batch_index, 0);
        assert_eq!(plan.decode_list[1].batch_index, 2);
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
        let shared_state = Arc::new(SharedState::new(Arc::clone(&batch_list)));

        let mut decode_list = DecodeList::with_capacity(2);
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

        let task = ScheduleTask::new(0, 2, vec![Vec::new(); thread_num], decode_list, 99);
        shared_state.set_task(task);

        let barrier = Arc::new(SpinBarrier::new(thread_num));
        let mut handles = Vec::with_capacity(thread_num);

        for thread_id in 0..thread_num {
            let shared_state = Arc::clone(&shared_state);
            let operator_queue = operator_queue.clone();
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                ExecutorPool::<f32>::execute_batch(
                    &shared_state,
                    &operator_queue,
                    &barrier,
                    thread_num,
                    thread_id,
                    &shared_state
                        .current_task()
                        .expect("expected scheduled task"),
                );
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(sequences[1], 11);
        assert_eq!(sequences[sequence_stride + 1], 21);

        shared_state.batch_list.with(|batch_list| {
            assert_eq!(batch_list[0].phase, crate::runtime::Phase::Decode);
            assert_eq!(batch_list[1].phase, crate::runtime::Phase::Decode);
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
        let shared_state = SharedState::new(Arc::clone(&batch_list));

        let mut decode_list = DecodeList::with_capacity(1);
        decode_list.push(SequenceSlice {
            batch_index: 0,
            sequence_index: 98,
            token_start_index: 0,
            length: 1,
            last_token_flag: true,
        });

        let task = ScheduleTask::new(0, 1, vec![Vec::new(); thread_num], decode_list, 100);
        shared_state.set_task(task);

        let barrier = SpinBarrier::new(thread_num);
        ExecutorPool::<f32>::execute_batch(
            &shared_state,
            &operator_queue,
            &barrier,
            thread_num,
            0,
            &shared_state
                .current_task()
                .expect("expected scheduled task"),
        );

        assert_eq!(sequences[99], eos_id);
        shared_state.batch_list.with(|batch_list| {
            assert_eq!(batch_list[0].phase, crate::runtime::Phase::Eos);
            assert_eq!(batch_list[0].sequence_index, 100);
            assert_eq!(batch_list[0].filling_length, 0);
        });
    }

    #[test]
    fn shared_state_set_task_sets_work_state_and_clear_work_resets_it() {
        let batch_list = Arc::new(SharedMut::new(vec![SlotState::new_start_state()]));
        let shared_state = SharedState::new(batch_list);

        assert!(!shared_state.has_work());
        assert!(matches!(
            shared_state.current_work(),
            crate::runtime::state::shared::ExecutorWork::Idle
        ));

        let task = ScheduleTask::new(1, 2, Vec::new(), DecodeList::with_capacity(0), 11);
        shared_state.set_task(task);
        assert!(shared_state.has_work());
        assert!(matches!(
            shared_state.current_work(),
            crate::runtime::state::shared::ExecutorWork::Scheduled(_)
        ));

        let task_ref = shared_state
            .current_task()
            .expect("expected scheduled task");
        assert_eq!(task_ref.prefill_size, 1);
        assert_eq!(task_ref.decode_size, 2);
        assert_eq!(task_ref.task_id, 11);

        shared_state.clear_work();
        assert!(!shared_state.has_work());
        assert!(matches!(
            shared_state.current_work(),
            crate::runtime::state::shared::ExecutorWork::Idle
        ));
    }

    #[test]
    fn executor_pool_new_clamps_thread_count_to_at_least_one() {
        let batch_list = Arc::new(SharedMut::new(vec![SlotState::new_start_state()]));
        let shared_state = Arc::new(SharedState::new(batch_list));
        let executor =
            ExecutorPool::<f32>::new(Vec::new(), shared_state, 0, 1, Duration::from_millis(1));

        assert_eq!(executor.thread_num, 1);
    }

    #[test]
    fn shared_state_accessor_returns_same_arc() {
        let batch_list = Arc::new(SharedMut::new(vec![SlotState::new_start_state()]));
        let shared_state = Arc::new(SharedState::new(batch_list));
        let executor = ExecutorPool::<f32>::new(
            Vec::new(),
            Arc::clone(&shared_state),
            1,
            1,
            Duration::from_millis(1),
        );

        let returned = executor.shared_state();
        assert!(Arc::ptr_eq(&shared_state, &returned));
    }

    #[test]
    fn executor_pool_uses_larger_thread_num_and_chunk_size_in_default_strategy() {
        let batch_list = Arc::new(SharedMut::new(vec![
            SlotState::new_decode_state(0, 0),
            SlotState::new_decode_state(1, 1),
            SlotState::new_decode_state(2, 2),
            SlotState::new_decode_state(3, 3),
            SlotState::new_prefill_state(10, 500),
            SlotState::new_prefill_state(20, 500),
        ]));
        let shared_state = Arc::new(SharedState::new(batch_list));

        let executor =
            ExecutorPool::<f32>::new(Vec::new(), shared_state, 8, 2048, Duration::from_millis(1));

        assert_eq!(executor.thread_num, 8);

        let plan = executor.strategy.plan_next_round(&[
            SlotState::new_decode_state(0, 0),
            SlotState::new_decode_state(1, 1),
            SlotState::new_decode_state(2, 2),
            SlotState::new_decode_state(3, 3),
            SlotState::new_prefill_state(10, 500),
            SlotState::new_prefill_state(20, 500),
        ]);

        assert_eq!(plan.mode, BatchMode::Mixed);
        assert_eq!(plan.decode_size, 6);
        assert_eq!(plan.prefill_list.len(), 8);
        assert!(plan.prefill_size <= 2048);
        assert_eq!(plan.decode_list.len(), 6);
    }

    #[test]
    fn executor_pool_with_thread_count_supports_large_values() {
        let batch_list = Arc::new(SharedMut::new(vec![SlotState::new_start_state(); 2]));
        let shared_state = Arc::new(SharedState::new(batch_list));

        let executor =
            ExecutorPool::<f32>::new(Vec::new(), shared_state, 2, 64, Duration::from_millis(1))
                .with_thread_count(16);

        assert_eq!(executor.thread_num, 16);
    }

    struct OnceStrategy {
        fired: AtomicBool,
    }

    impl OnceStrategy {
        fn new() -> Self {
            Self {
                fired: AtomicBool::new(false),
            }
        }
    }

    impl SchedulerStrategy for OnceStrategy {
        fn plan_next_round(&self, batch_list: &[SlotState]) -> BatchPlan {
            if self
                .fired
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                return BatchPlan::new(0);
            }

            let mut plan = BatchPlan::new(123);
            let mut decode_list = DecodeList::with_capacity(batch_list.len());

            for (batch_index, state) in batch_list.iter().enumerate() {
                if matches!(state.phase, crate::runtime::Phase::Decode) {
                    decode_list.push(SequenceSlice {
                        batch_index,
                        sequence_index: state.sequence_index,
                        token_start_index: decode_list.len(),
                        length: 1,
                        last_token_flag: true,
                    });
                }
            }

            plan.mode = BatchMode::Decode;
            plan.decode_size = decode_list.len();
            plan.decode_list = decode_list;
            plan
        }
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
        let shared_state = Arc::new(SharedState::new(Arc::clone(&batch_list)));

        let executor = ExecutorPool::<f32>::new(
            operator_queue,
            shared_state,
            2,
            64,
            Duration::from_millis(1),
        )
        .with_strategy(Arc::new(OnceStrategy::new()));

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
