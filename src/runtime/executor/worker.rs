use std::sync::Arc;

use super::barrier::SpinBarrier;
use super::plan::BatchPlan;
use crate::operators::operator::Operator;
use crate::runtime::state::shared::SharedState;
use crate::runtime::state::types::Phase;

pub struct Worker<T> {
    shared_state: Arc<SharedState>,
    operator_queue: Arc<[Operator<T>]>,
    barrier: Arc<SpinBarrier>,
    thread_id: usize,
    thread_num: usize,
}

impl<T> Worker<T>
where
    T: Copy
        + Default
        + std::ops::Sub<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::AddAssign
        + crate::num_traits::exp::Exp
        + crate::num_traits::sqrt::Sqrt
        + crate::num_traits::neg_infinity::NegInfinity
        + crate::num_traits::sigmoid::Sigmoid
        + PartialOrd
        + Send
        + Sync
        + 'static,
{
    pub fn new(
        shared_state: Arc<SharedState>,
        operator_queue: Arc<[Operator<T>]>,
        barrier: Arc<SpinBarrier>,
        thread_id: usize,
        thread_num: usize,
    ) -> Self {
        Self {
            shared_state,
            operator_queue,
            barrier,
            thread_id,
            thread_num,
        }
    }

    pub fn run(self) {
        if self.thread_id == 0 {
            self.run_leader();
        } else {
            self.run_follower();
        }
    }

    fn run_leader(self) {
        loop {
            self.wait_for_request();

            let request_count = self.shared_state.take_requests();
            if request_count == 0 {
                continue;
            }

            self.shared_state.set_scheduler_state(
                crate::runtime::state::shared::SchedulerState::Scheduling
            );

            let plan = self.build_plan();
            if plan.is_empty() {
                self.shared_state.set_scheduler_state(
                    crate::runtime::state::shared::SchedulerState::Idle
                );
                continue;
            }

            self.shared_state.batch_tracker.reset(plan.sequence_count());

            self.shared_state.publish_batch(Box::new(plan));

            self.shared_state.set_scheduler_state(
                crate::runtime::state::shared::SchedulerState::Executing
            );

            let plan = match self.shared_state.take_batch() {
                Some(p) => p,
                None => {
                    self.shared_state.set_scheduler_state(
                        crate::runtime::state::shared::SchedulerState::Idle
                    );
                    continue;
                }
            };

            self.execute_batch(&plan);

            self.shared_state.set_scheduler_state(
                crate::runtime::state::shared::SchedulerState::Completing
            );

            self.update_states(&plan);

            self.shared_state.clear_batch();

            self.shared_state.set_scheduler_state(
                crate::runtime::state::shared::SchedulerState::Idle
            );
        }
    }

    fn run_follower(self) {
        loop {
            self.wait_for_batch();

            let plan = match self.shared_state.take_batch() {
                Some(p) => p,
                None => continue,
            };

            self.execute_batch(&plan);

            self.update_states(&plan);

            while self.shared_state.get_scheduler_state()
                == crate::runtime::state::shared::SchedulerState::Executing
            {
                std::hint::spin_loop();
            }
        }
    }

    fn wait_for_request(&self) {
        for _ in 0..10000 {
            if self.shared_state.request_count.load(std::sync::atomic::Ordering::Acquire) > 0 {
                return;
            }
            std::hint::spin_loop();
        }

        while self.shared_state.request_count.load(std::sync::atomic::Ordering::Acquire) == 0 {
            std::thread::yield_now();
        }
    }

    fn wait_for_batch(&self) {
        for _ in 0..1000 {
            if self.shared_state.batch_ready.load(std::sync::atomic::Ordering::Acquire) {
                return;
            }
            std::hint::spin_loop();
        }

        while !self.shared_state.batch_ready.load(std::sync::atomic::Ordering::Acquire) {
            std::hint::spin_loop();
        }
    }

    fn build_plan(&self) -> BatchPlan {
        let batch_list_ptr = self.shared_state.batch_list.get();
        unsafe {
            let batch_list = &*batch_list_ptr;
            self.shared_state.plan_builder.build_plan(batch_list)
        }
    }

    fn execute_batch(&self, plan: &BatchPlan) {
        let prefill_size = plan.prefill_size;
        let decode_size = plan.decode_size;
        let prefill_list = &plan.prefill_list;
        let decode_list = &plan.decode_list;

        self.barrier.wait();

        for operator in self.operator_queue.iter() {
            self.barrier.wait();

            let batch_list_ptr = self.shared_state.batch_list.get();
            unsafe {
                let batch_list = &mut *batch_list_ptr;
                operator.run(
                    prefill_size,
                    decode_size,
                    self.thread_num,
                    self.thread_id,
                    prefill_list,
                    decode_list,
                    batch_list,
                );
            }

            self.barrier.wait();
        }

        let _ = self.barrier.wait();
    }

    fn update_states(&self, plan: &BatchPlan) {
        let batch_list_ptr = self.shared_state.batch_list.get();
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

        if self.thread_id == 0 {
            for _ in 0..plan.sequence_count() {
                self.shared_state.batch_tracker.complete_slot();
            }
        }
    }
}