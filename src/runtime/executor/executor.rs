use std::sync::atomic::Ordering;
use std::sync::Arc;

use tokio::sync::broadcast;
use tokio::task::JoinHandle;

use crate::operators::operator::Operator;
use crate::runtime::executor::sync::{AdaptiveWait, SpinBarrier};
use crate::runtime::state::machine::SlotStateMachine;
use crate::runtime::state::shared::SharedState;
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
    pub fn new(
        operator_queue: Vec<Operator<T>>,
        shared_state: Arc<SharedState>,
        thread_num: usize,
    ) -> Self {
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
                    SlotStateMachine::advance_sequence(record, slice.length);
                }
            }
        }
    }

    pub fn with_thread_count(mut self, thread_num: usize) -> Self {
        self.thread_num = thread_num.max(1);
        self
    }

    pub fn start(mut self, schedule_rx: broadcast::Receiver<ScheduleTask>) {
        let barrier = Arc::new(SpinBarrier::new(self.thread_num));

        for thread_id in 0..self.thread_num {
            let shared_state = Arc::clone(&self.shared_state);
            let operator_queue = Arc::clone(&self.operator_queue);
            let barrier = Arc::clone(&barrier);
            let thread_num = self.thread_num;
            let schedule_rx = schedule_rx.resubscribe();

            let handle = tokio::task::spawn_blocking(move || {
                Self::run_worker(
                    shared_state,
                    operator_queue.as_ref(),
                    &barrier,
                    thread_num,
                    thread_id,
                    schedule_rx,
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
        mut schedule_rx: broadcast::Receiver<ScheduleTask>,
    ) {
        let mut wait = AdaptiveWait::new();

        loop {
            match schedule_rx.try_recv() {
                Ok(task) => {
                    let sequence_count: usize = task.decode_size
                        + if task.prefill_size > 0 {
                            1usize
                        } else {
                            0usize
                        };

                    shared_state.batch_tracker.reset(sequence_count);

                    execute_batch(
                        &shared_state,
                        operator_queue,
                        barrier,
                        thread_num,
                        thread_id,
                        &task,
                    );

                    update_states(&shared_state, &task);
                }
                Err(_) => {
                    wait.wait(|| {
                        shared_state.get_scheduler_state()
                            != crate::runtime::state::shared::SchedulerState::Idle
                    });
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
    }

    let sequence_count: usize = task.decode_size
        + if task.prefill_size > 0 {
            1usize
        } else {
            0usize
        };
    for _ in 0..sequence_count {
        shared_state.batch_tracker.complete_slot();
    }
}
