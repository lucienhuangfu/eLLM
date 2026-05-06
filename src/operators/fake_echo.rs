use crate::runtime::{Phase, SequenceState};

#[derive(Clone, Default)]
pub struct FakeEcho;

impl FakeEcho {
    pub fn run(&self, batch_list: &mut Vec<SequenceState>, thread_id: usize) {
        // ServingRunner 的 thread_id 从 0 开始，0 号线程负责推进 fake 完成。
        if thread_id != 0 {
            return;
        }

        for record in batch_list.iter_mut() {
            if matches!(record.phase, Phase::Prefill) {
                record.sequence_index = 0;
                record.kv_index = record.filling_length;
                record.phase = Phase::Eos;
                record.notify.notify_one();
            }
        }
    }
}
