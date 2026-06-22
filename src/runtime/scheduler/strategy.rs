use crate::runtime::plan::BatchPlan;
use crate::runtime::state::core::SlotState;

pub trait SchedulerStrategy: Send + Sync + 'static {
    fn plan_next_round(
        &self,
        batch_list: &[SlotState],
        max_decode_size: usize,
        max_prefill_size: usize,
    ) -> BatchPlan;
}

pub struct DefaultSchedulerStrategy {
    max_decode_size: usize,
    max_prefill_size: usize,
    thread_num: usize,
}

impl DefaultSchedulerStrategy {
    pub fn new(max_decode_size: usize, max_prefill_size: usize, thread_num: usize) -> Self {
        Self {
            max_decode_size,
            max_prefill_size,
            thread_num,
        }
    }
}

impl SchedulerStrategy for DefaultSchedulerStrategy {
    fn plan_next_round(
        &self,
        batch_list: &[SlotState],
        _max_decode_size: usize,
        _max_prefill_size: usize,
    ) -> BatchPlan {
        let builder = crate::runtime::plan::PlanBuilder::new(
            self.max_decode_size,
            self.max_prefill_size,
            self.thread_num,
        );
        builder.build_plan(batch_list)
    }
}