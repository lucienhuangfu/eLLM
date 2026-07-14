use std::cell::UnsafeCell;
use std::sync::Arc;

use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::scheduler::task::{BatchMode, ScheduleTask};
use crate::runtime::state::core::SlotState;
use crate::runtime::state::sequence::SequenceSlice;
use crate::runtime::state::shared::SharedState;
use crate::runtime::state::types::Phase;

/// Lightweight prefill metadata kept during scheduling.
#[derive(Debug, Clone, Copy)]
struct PrefillSlot {
    batch_index: usize,
    sequence_index: usize,
    filling_length: usize,
}

pub struct Scheduler {
    max_decode_size: usize,
    max_prefill_size: usize,
    thread_num: usize,
    /// Mutable scheduling scratch space, accessed only inside `schedule_batch`
    /// which is guaranteed to be called sequentially.
    prefill_slots: UnsafeCell<Vec<PrefillSlot>>,
    shared_state: Arc<SharedState>,
}

// SAFETY: prefill_slots is only accessed through &self in schedule_batch
// which is guaranteed to be called sequentially by the runtime.
unsafe impl Send for Scheduler {}
unsafe impl Sync for Scheduler {}

impl Scheduler {
    pub fn new(
        max_decode_size: usize,
        max_prefill_size: usize,
        thread_num: usize,
        shared_state: Arc<SharedState>,
    ) -> Self {
        Self {
            max_decode_size,
            max_prefill_size,
            thread_num: thread_num.max(1),
            prefill_slots: UnsafeCell::new(Vec::with_capacity(max_decode_size)),
            shared_state,
        }
    }

    #[inline]
    pub fn thread_num(&self) -> usize {
        self.thread_num
    }

    pub fn set_thread_num(&mut self, thread_num: usize) {
        self.thread_num = thread_num.max(1);
    }

    #[inline]
    pub fn batch_list(&self) -> Arc<SharedMut<Vec<SlotState>>> {
        Arc::clone(&self.shared_state.batch_list)
    }

    #[inline]
    pub fn shared_state(&self) -> Arc<SharedState> {
        Arc::clone(&self.shared_state)
    }

    /// Schedule a batch of slots, building the task for the executor.
    ///
    /// Returns `true` if work was scheduled, `false` otherwise.
    #[inline]
    pub fn schedule_batch(&self) -> bool {
        // SAFETY: schedule_batch is called sequentially by the runtime.
        let prefill_slots = unsafe { &mut *self.prefill_slots.get() };

        self.shared_state.batch_list.with(|batch_list| {
            self.shared_state.task().with_mut(|task| {
                Self::build_task(
                    batch_list,
                    task,
                    prefill_slots,
                    self.max_decode_size,
                    self.max_prefill_size,
                    self.thread_num,
                );
            });
            !self.shared_state.task().with(|task| task.is_empty())
        })
    }

    /// Core scheduling logic: builds a [`ScheduleTask`] from the current batch list.
    ///
    /// Collects prefill and decode candidates from the batch list,
    /// then constructs the task so that `decode_list` contains prefill entries
    /// first (with correct `token_start_index`), followed by pure decode entries.
    fn build_task(
        batch_list: &[SlotState],
        task: &mut ScheduleTask,
        prefill_slots: &mut Vec<PrefillSlot>,
        max_decode_size: usize,
        max_prefill_size: usize,
        thread_num: usize,
    ) {
        task.reset();
        prefill_slots.clear();

        // ── Phase 1: Collect ────────────────────────────────────────
        let mut decode_count = 0usize;

        for (batch_index, slot) in batch_list.iter().enumerate() {
            match slot.phase {
                Phase::Prefill if prefill_slots.len() < max_decode_size => {
                    prefill_slots.push(PrefillSlot {
                        batch_index,
                        sequence_index: slot.sequence_index,
                        filling_length: slot.filling_length,
                    });
                }
                Phase::Decode if decode_count < max_decode_size => {
                    decode_count += 1;
                }
                _ => {}
            }
        }

        let has_prefill = !prefill_slots.is_empty();
        let has_decode = decode_count > 0;

        task.mode = match (has_prefill, has_decode) {
            (true, true) => BatchMode::Mixed,
            (true, false) => BatchMode::Prefill,
            (false, true) => BatchMode::Decode,
            (false, false) => return,
        };

        task.decode_size = decode_count;

        // ── Phase 2: Build prefill ──────────────────────────────────
        if has_prefill {
            Self::build_prefill(task, prefill_slots, max_prefill_size, thread_num);
        }

        // ── Phase 3: Build decode ───────────────────────────────────
        if has_decode {
            Self::build_decode(task, batch_list, decode_count);
        }
    }

    /// Writes prefill entries into `decode_list` and distributes tokens
    /// across threads in `prefill_list`.
    fn build_prefill(
        task: &mut ScheduleTask,
        prefill_slots: &[PrefillSlot],
        max_prefill_size: usize,
        thread_num: usize,
    ) {
        let raw_total: usize = prefill_slots.iter().map(|s| s.filling_length).sum();
        let total_tokens = raw_total.min(max_prefill_size);

        if total_tokens == 0 {
            return;
        }

        task.resize_prefill_list(thread_num);

        // Pre-reserve per-thread capacity
        let per_thread = total_tokens / thread_num;
        for list in task.prefill_list.iter_mut() {
            list.reserve(per_thread + 1);
        }

        // Distribute tokens evenly across threads
        let base_quota = total_tokens / thread_num;
        let extra = total_tokens % thread_num;

        let mut scheduled = 0usize;
        let mut prefill_count = 0usize;
        let mut thread_idx = 0usize;
        let mut thread_remaining = base_quota + if extra > 0 { 1 } else { 0 };

        for slot in prefill_slots {
            if scheduled >= total_tokens {
                break;
            }

            let mut cursor = slot.sequence_index;
            let mut remaining = slot.filling_length;
            let attention_len = remaining.min(total_tokens - scheduled);

            // Write the prefill entry to decode_list
            task.decode_list.push(SequenceSlice {
                batch_index: slot.batch_index,
                sequence_index: slot.sequence_index,
                token_start_index: prefill_count,
                length: attention_len,
                last_token_flag: attention_len == slot.filling_length,
            });
            prefill_count += attention_len;

            // Distribute this sequence's tokens across threads
            while remaining > 0 && scheduled < total_tokens {
                // Advance to next thread with remaining quota
                while thread_idx < thread_num && thread_remaining == 0 {
                    thread_idx += 1;
                    if thread_idx < thread_num {
                        thread_remaining = base_quota + if thread_idx < extra { 1 } else { 0 };
                    }
                }
                if thread_idx >= thread_num {
                    break;
                }

                let chunk = thread_remaining
                    .min(remaining)
                    .min(total_tokens - scheduled);
                task.prefill_list[thread_idx].push(SequenceSlice {
                    batch_index: slot.batch_index,
                    sequence_index: cursor,
                    token_start_index: scheduled,
                    length: chunk,
                    last_token_flag: false,
                });

                scheduled += chunk;
                thread_remaining -= chunk;
                remaining -= chunk;
                cursor += chunk;
            }
        }

        task.prefill_size = prefill_count;
    }

    /// Scans `batch_list` for decode slots and appends them to `decode_list`.
    fn build_decode(task: &mut ScheduleTask, batch_list: &[SlotState], max_count: usize) {
        let mut count = 0usize;
        for (batch_index, slot) in batch_list.iter().enumerate() {
            if count >= max_count {
                break;
            }
            if slot.phase == Phase::Decode {
                task.decode_list.push(SequenceSlice {
                    batch_index,
                    sequence_index: slot.sequence_index,
                    token_start_index: count,
                    length: 1,
                    last_token_flag: true,
                });
                count += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_shared_state(batch_list: Vec<SlotState>) -> Arc<SharedState> {
        Arc::new(SharedState::new(Arc::new(SharedMut::new(batch_list))))
    }

    fn decode_state(sequence_index: usize, kv_index: usize) -> SlotState {
        SlotState::new_decode_state(sequence_index, kv_index)
    }

    #[test]
    fn schedule_batch_returns_false_for_empty_batch() {
        let shared_state = make_shared_state(Vec::new());
        let scheduler = Scheduler::new(16, 4, 3, shared_state);

        assert!(!scheduler.schedule_batch());
    }

    #[test]
    fn schedule_batch_fills_task_for_decode() {
        let shared_state = make_shared_state(Vec::new());
        let scheduler = Scheduler::new(16, 4, 3, Arc::clone(&shared_state));
        shared_state.batch_list.with_mut(|batch_list| {
            batch_list.push(decode_state(100, 128));
        });

        assert!(scheduler.schedule_batch());

        shared_state.task().with(|task| {
            assert_eq!(task.prefill_size, 0);
            assert_eq!(task.decode_size, 1);
        });
    }

    #[test]
    fn set_thread_num_updates_thread_count() {
        let shared_state = make_shared_state(Vec::new());
        let mut scheduler = Scheduler::new(16, 4, 6, shared_state);

        scheduler.set_thread_num(3);
        assert_eq!(scheduler.thread_num(), 3);

        scheduler.set_thread_num(5);
        assert_eq!(scheduler.thread_num(), 5);
    }
}
