use std::sync::atomic::Ordering;
use std::sync::Arc;

use tokio::task::JoinHandle;

use crate::operators::operator::Operator;
use crate::runtime::executor::sync::{AdaptiveWait, SpinBarrier};
use crate::runtime::plan::BatchPlan;
use crate::runtime::state::shared::SharedState;
use crate::runtime::state::types::Phase;
use crate::runtime::ScheduleTask;

use crate::num_traits::{exp::Exp, neg_infinity::NegInfinity, sigmoid::Sigmoid, sqrt::Sqrt};

pub struct ExecutorPool<T> {
    shared_state: Arc<SharedState>,
    operator_queue: Arc<[Operator<T>]>,
    thread_num: usize,
    handles: Vec<JoinHandle<()>>,
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
    pub fn new(operator_queue: Vec<Operator<T>>, shared_state: Arc<SharedState>) -> Self {
        let thread_num = num_cpus::get();
        Self {
            shared_state,
            operator_queue: operator_queue.into(),
            thread_num: thread_num.max(1),
            handles: Vec::with_capacity(thread_num.max(1)),
        }
    }

    pub fn execute_single_thread_batch(&self, task: &ScheduleTask) {
        let prefill_size = task.prefill_size;
        let decode_size = task.decode_size;
        let prefill_list = &task.prefill_list;
        let decode_list = &task.decode_list;

        for operator in self.operator_queue.iter() {
            let batch_list_ptr = self.shared_state.batch_list.get();
            unsafe {
                let batch_list = &mut *batch_list_ptr;
                operator.run(
                    prefill_size,
                    decode_size,
                    1,
                    0,
                    prefill_list,
                    decode_list,
                    batch_list,
                );
            }
        }

        let batch_list_ptr = self.shared_state.batch_list.get();
        unsafe {
            let batch_list = &mut *batch_list_ptr;
            for slice in decode_list.iter() {
                if let Some(record) = batch_list.get_mut(slice.batch_index) {
                    record.sequence_index += slice.length;
                    if record.phase == Phase::Prefill {
                        record.filling_length = record.filling_length.saturating_sub(slice.length);
                        if record.filling_length == 0 {
                            record.transition_to_decode();
                        }
                    }
                }
            }
        }
    }

    pub fn with_thread_count(mut self, thread_num: usize) -> Self {
        self.thread_num = thread_num.max(1);
        self
    }

    pub fn start(mut self) {
        let barrier = Arc::new(SpinBarrier::new(self.thread_num));

        for thread_id in 0..self.thread_num {
            let shared_state = Arc::clone(&self.shared_state);
            let operator_queue = Arc::clone(&self.operator_queue);
            let barrier = Arc::clone(&barrier);
            let thread_num = self.thread_num;
            let is_leader = thread_id == 0;

            let handle = tokio::task::spawn_blocking(move || {
                Self::run_worker(
                    shared_state,
                    operator_queue.as_ref(),
                    &barrier,
                    thread_num,
                    thread_id,
                    is_leader,
                );
            });

            self.handles.push(handle);
        }
    }

    pub fn shared_state(&self) -> Arc<SharedState> {
        Arc::clone(&self.shared_state)
    }

    fn run_worker(
        shared_state: Arc<SharedState>,
        operator_queue: &[Operator<T>],
        barrier: &SpinBarrier,
        thread_num: usize,
        thread_id: usize,
        is_leader: bool,
    ) {
        let mut wait = AdaptiveWait::new();

        loop {
            if is_leader {
                wait.wait(|| shared_state.request_count.load(Ordering::Acquire) > 0);

                let request_count = shared_state.take_requests();
                if request_count == 0 {
                    continue;
                }

                shared_state
                    .set_scheduler_state(crate::runtime::state::shared::SchedulerState::Scheduling);

                let plan = build_plan(&shared_state);
                if plan.is_empty() {
                    shared_state
                        .set_scheduler_state(crate::runtime::state::shared::SchedulerState::Idle);
                    continue;
                }

                shared_state.batch_tracker.reset(plan.sequence_count());
                shared_state.publish_batch(Box::new(plan));

                shared_state
                    .set_scheduler_state(crate::runtime::state::shared::SchedulerState::Executing);
            } else {
                wait.wait(|| shared_state.batch_ready.load(Ordering::Acquire));
            }

            let plan = match shared_state.take_batch() {
                Some(p) => p,
                None => {
                    if is_leader {
                        shared_state.set_scheduler_state(
                            crate::runtime::state::shared::SchedulerState::Idle,
                        );
                    }
                    continue;
                }
            };

            execute_batch(
                &shared_state,
                operator_queue,
                barrier,
                thread_num,
                thread_id,
                &plan,
            );

            if is_leader {
                shared_state
                    .set_scheduler_state(crate::runtime::state::shared::SchedulerState::Completing);

                update_states(&shared_state, &plan);

                shared_state.clear_batch();
                shared_state
                    .set_scheduler_state(crate::runtime::state::shared::SchedulerState::Idle);
            } else {
                while shared_state.get_scheduler_state()
                    == crate::runtime::state::shared::SchedulerState::Executing
                {
                    std::hint::spin_loop();
                }
            }
        }
    }
}

fn build_plan(shared_state: &SharedState) -> BatchPlan {
    let batch_list_ptr = shared_state.batch_list.get();
    unsafe {
        let batch_list = &*batch_list_ptr;
        shared_state.plan_builder.build_plan(batch_list)
    }
}

fn execute_batch<T>(
    shared_state: &SharedState,
    operator_queue: &[Operator<T>],
    barrier: &SpinBarrier,
    thread_num: usize,
    thread_id: usize,
    plan: &BatchPlan,
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
    let prefill_size = plan.prefill_size;
    let decode_size = plan.decode_size;
    let prefill_list = &plan.prefill_list;
    let decode_list = &plan.decode_list;

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

fn update_states(shared_state: &SharedState, plan: &BatchPlan) {
    let batch_list_ptr = shared_state.batch_list.get();
    unsafe {
        let batch_list = &mut *batch_list_ptr;

        for slice in plan.decode_list.iter() {
            if let Some(record) = batch_list.get_mut(slice.batch_index) {
                record.sequence_index += slice.length;

                if record.phase == Phase::Prefill {
                    record.filling_length = record.filling_length.saturating_sub(slice.length);
                    if record.filling_length == 0 {
                        record.transition_to_decode();
                    }
                }
            }
        }
    }

    let sequence_count = plan.sequence_count();
    for _ in 0..sequence_count {
        shared_state.batch_tracker.complete_slot();
    }
}
