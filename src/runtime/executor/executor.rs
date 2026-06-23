use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use tokio::task::JoinHandle;

use crate::operators::operator::Operator;
use crate::runtime::executor::sync::{AdaptiveWait, SpinBarrier};
use crate::runtime::plan::BatchPlan;
use crate::runtime::scheduler::{DefaultSchedulerStrategy, ScheduleTask, SchedulerStrategy};
use crate::runtime::session::SlotManager;
use crate::runtime::state::machine::SlotStateMachine;
use crate::runtime::state::shared::SharedState;

use crate::num_traits::{exp::Exp, neg_infinity::NegInfinity, sigmoid::Sigmoid, sqrt::Sqrt};

pub struct ExecutorPool<T> {
    shared_state: Arc<SharedState>,
    operator_queue: Arc<[Operator<T>]>,
    thread_num: usize,
    handles: Vec<JoinHandle<()>>,
    strategy: Arc<dyn SchedulerStrategy>,
    slot_manager: Arc<SlotManager<f16>>,
    timeout: Duration,
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
        slot_manager: Arc<SlotManager<f16>>,
        timeout: Duration,
    ) -> Self {
        let batch_size = shared_state.batch_list.with(|list| list.len());
        let strategy = Arc::new(DefaultSchedulerStrategy::new(
            batch_size,
            batch_size * 1024,
            thread_num,
        ));
        Self {
            shared_state,
            operator_queue: operator_queue.into(),
            thread_num: thread_num.max(1),
            handles: Vec::with_capacity(thread_num.max(1)),
            strategy,
            slot_manager,
            timeout,
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

    pub fn execute_task(&self, task: &ScheduleTask) {
        let barrier = SpinBarrier::new(self.thread_num);
        execute_batch(
            &self.shared_state,
            self.operator_queue.as_ref(),
            &barrier,
            self.thread_num,
            0,
            task,
        );
        update_states(&self.shared_state, task);
    }

    pub fn start(mut self) {
        let barrier = Arc::new(SpinBarrier::new(self.thread_num));

        for thread_id in 0..self.thread_num {
            let shared_state = Arc::clone(&self.shared_state);
            let operator_queue = Arc::clone(&self.operator_queue);
            let barrier = Arc::clone(&barrier);
            let thread_num = self.thread_num;
            let strategy = Arc::clone(&self.strategy);
            let slot_manager = Arc::clone(&self.slot_manager);
            let timeout = self.timeout;

            let handle = tokio::task::spawn_blocking(move || {
                Self::run_worker(
                    shared_state,
                    operator_queue.as_ref(),
                    &barrier,
                    thread_num,
                    thread_id,
                    strategy,
                    slot_manager,
                    timeout,
                );
            });

            self.handles.push(handle);
        }
    }

    pub fn shared_state(&self) -> Arc<SharedState> {
        Arc::clone(&self.shared_state)
    }

    fn schedule_batch(
        strategy: &dyn SchedulerStrategy,
        batch_list: &Arc<
            crate::operators::send_sync_ptr::SharedMut<Vec<crate::runtime::state::core::SlotState>>,
        >,
    ) -> Option<BatchPlan> {
        batch_list.with(|batch_list| {
            let plan = strategy.plan_next_round(batch_list, 0, 0);
            if plan.is_empty() {
                None
            } else {
                Some(plan)
            }
        })
    }

    fn run_worker(
        shared_state: Arc<SharedState>,
        operator_queue: &[Operator<T>],
        barrier: &SpinBarrier,
        thread_num: usize,
        thread_id: usize,
        strategy: Arc<dyn SchedulerStrategy>,
        slot_manager: Arc<SlotManager<f16>>,
        timeout: Duration,
    ) {
        let mut wait = AdaptiveWait::new();
        println!("[Executor] Worker {} 启动", thread_id);

        loop {
            if thread_id == 0 {
                if shared_state
                    .task_in_flight
                    .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                    .is_err()
                {
                    wait.wait(|| shared_state.task_in_flight.load(Ordering::Acquire) == false);
                    continue;
                }

                let plan = match Self::schedule_batch(&*strategy, &shared_state.batch_list) {
                    Some(p) => p,
                    None => {
                        shared_state.task_in_flight.store(false, Ordering::Release);

                        let has_work = slot_manager.has_work_blocking();
                        if !has_work {
                            std::thread::sleep(timeout);
                        }
                        continue;
                    }
                };

                let task = ScheduleTask::new(
                    plan.prefill_size,
                    plan.decode_size,
                    plan.prefill_list,
                    plan.decode_list,
                    plan.task_id,
                );

                println!(
                    "[Executor] Worker {} 调度任务: task_id={}, prefill_size={}, decode_size={}",
                    thread_id, task.task_id, task.prefill_size, task.decode_size
                );

                *shared_state.last_task.lock().unwrap() = Some(task.clone());

                execute_batch(
                    &shared_state,
                    operator_queue,
                    barrier,
                    thread_num,
                    thread_id,
                    &task,
                );

                update_states(&shared_state, &task);

                shared_state.task_in_flight.store(false, Ordering::Release);

                wait.reset();
            } else {
                wait.wait(|| shared_state.task_in_flight.load(Ordering::Acquire) == true);

                if shared_state.task_in_flight.load(Ordering::Acquire) {
                    if let Some(ref task) = *shared_state.last_task.lock().unwrap() {
                        execute_batch(
                            &shared_state,
                            operator_queue,
                            barrier,
                            thread_num,
                            thread_id,
                            task,
                        );
                    }
                    wait.reset();
                }
            }
        }
    }
}

fn execute_batch<T>(
    shared_state: &SharedState,
    operator_queue: &[Operator<T>],
    barrier: &SpinBarrier,
    thread_num: usize,
    thread_id: usize,
    task: &ScheduleTask,
) where
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
    let prefill_size = task.prefill_size;
    let decode_size = task.decode_size;
    let prefill_list = &task.prefill_list;
    let decode_list = &task.decode_list;

    barrier.wait();

    for operator in operator_queue.iter() {
        barrier.wait();

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

    barrier.wait();
}

fn update_states(shared_state: &SharedState, task: &ScheduleTask) {
    let batch_list_ptr = shared_state.batch_list.get();
    unsafe {
        let batch_list = &mut *batch_list_ptr;

        for slice in task.decode_list.iter() {
            if let Some(record) = batch_list.get_mut(slice.batch_index) {
                SlotStateMachine::advance_sequence(record, slice.length);
            }
        }

        for prefill_list in task.prefill_list.iter() {
            for slice in prefill_list.iter() {
                if let Some(record) = batch_list.get_mut(slice.batch_index) {
                    SlotStateMachine::advance_sequence(record, slice.length);
                }
            }
        }
    }
}
