//! Runtime 模块测试模块

pub(super) use crate::operators::send_sync_ptr::SharedMut;
pub(super) use crate::runtime::error::{SlotError, SlotResult};
pub(super) use crate::runtime::plan::{BatchMode, BatchPlan, PlanBuilder};
pub(super) use crate::runtime::scheduler::ScheduleTask;
pub(super) use crate::runtime::session::{SessionHandle, SessionMode, SlotManager};
pub(super) use crate::runtime::state::batch::BatchSequence;
pub(super) use crate::runtime::state::core::SlotState;
pub(super) use crate::runtime::state::machine::{SlotStateMachine, TransitionError};
pub(super) use crate::runtime::state::sequence::{DecodeList, DecodeLookupResult, SequenceSlice};
pub(super) use crate::runtime::state::shared::SharedState;
pub(super) use crate::runtime::state::types::Phase;
pub(super) use crate::runtime::ExecutorPool;
pub(super) use std::sync::Arc;

#[cfg(test)]
mod integration_tests;
#[cfg(test)]
mod workflow_tests;

/// 保留原有的 runtime reexports 构造测试
#[test]
fn runtime_reexports_are_constructible() {
    let prefill_state = SlotState::new_prefill_state(8, 4);
    let decode_state = SlotState::new_decode_state(16, 16);

    assert_eq!(prefill_state.sequence_index, 8);
    assert_eq!(prefill_state.kv_index, 8);
    assert_eq!(prefill_state.filling_length, 4);
    assert_eq!(prefill_state.phase, Phase::Prefill);
    assert_eq!(Arc::strong_count(&prefill_state.notify), 1);

    assert_eq!(decode_state.sequence_index, 16);
    assert_eq!(decode_state.kv_index, 16);
    assert_eq!(decode_state.filling_length, 0);
    assert_eq!(decode_state.phase, Phase::Decode);
    assert_eq!(Arc::strong_count(&decode_state.notify), 1);
}
