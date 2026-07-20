use std::cell::UnsafeCell;
use std::sync::Arc;

use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::batch::SequenceSlice;
use crate::runtime::session::{Phase, SlotState};

use super::task::ScheduleTask;

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
    prefill_slots: UnsafeCell<Vec<PrefillSlot>>,
    batch_list: Arc<SharedMut<Vec<SlotState>>>,
    task: SharedMut<ScheduleTask>,
}

unsafe impl Send for Scheduler {}
unsafe impl Sync for Scheduler {}

impl Scheduler {
    pub fn new(
        max_decode_size: usize,
        max_prefill_size: usize,
        thread_num: usize,
        batch_list: Arc<SharedMut<Vec<SlotState>>>,
    ) -> Self {
        let thread_num = thread_num.max(1);
        Self {
            max_decode_size,
            max_prefill_size,
            thread_num,
            prefill_slots: UnsafeCell::new(Vec::with_capacity(max_decode_size)),
            batch_list,
            task: SharedMut::new(ScheduleTask::new(thread_num, max_decode_size)),
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
        Arc::clone(&self.batch_list)
    }

    #[inline]
    pub fn with_task<R>(&self, f: impl FnOnce(&ScheduleTask) -> R) -> R {
        self.task.with(f)
    }

    #[inline]
    pub fn with_task_mut<R>(&self, f: impl FnOnce(&mut ScheduleTask) -> R) -> R {
        self.task.with_mut(f)
    }

    #[inline]
    pub fn has_work(&self) -> bool {
        self.task.with(|task| !task.is_empty())
    }

    #[inline]
    pub fn schedule_batch(&self) -> bool {
        let prefill_slots = unsafe { &mut *self.prefill_slots.get() };

        self.batch_list.with(|batch_list| {
            self.task.with_mut(|task| {
                Self::build_task(
                    batch_list,
                    task,
                    prefill_slots,
                    self.max_decode_size,
                    self.max_prefill_size,
                    self.thread_num,
                );
            });
            !self.task.with(|task| task.is_empty())
        })
    }

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

        if !has_prefill && !has_decode {
            return;
        }

        task.decode_size = decode_count;

        if has_prefill {
            Self::build_prefill(task, prefill_slots, max_prefill_size, thread_num);
        }

        if has_decode {
            Self::build_decode(task, batch_list, decode_count);
        }

        task.total_token_num = task.prefill_size + task.decode_size;
    }

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
            let prefill_len = remaining.min(total_tokens - scheduled);

            task.slices.push(SequenceSlice {
                batch_index: slot.batch_index,
                sequence_index: slot.sequence_index,
                token_start_index: prefill_count,
                length: prefill_len,
                last_token_flag: prefill_len == slot.filling_length,
            });
            prefill_count += prefill_len;

            while remaining > 0 && scheduled < total_tokens {
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
                task.prefilling_chunked_slices[thread_idx].push(SequenceSlice {
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

    fn build_decode(task: &mut ScheduleTask, batch_list: &[SlotState], max_count: usize) {
        let mut count = 0usize;
        let token_offset = task.prefill_size;
        for (batch_index, slot) in batch_list.iter().enumerate() {
            if count >= max_count {
                break;
            }
            if slot.phase == Phase::Decode {
                task.slices.push(SequenceSlice {
                    batch_index,
                    sequence_index: slot.sequence_index,
                    token_start_index: token_offset + count,
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

    fn make_batch_list(slots: Vec<SlotState>) -> Arc<SharedMut<Vec<SlotState>>> {
        Arc::new(SharedMut::new(slots))
    }

    fn advance_slot(slot: &mut SlotState, steps: usize) -> Option<Phase> {
        if slot.phase == Phase::Eos {
            return None;
        }
        slot.sequence_index += steps;
        if slot.phase == Phase::Prefill {
            slot.filling_length = slot.filling_length.saturating_sub(steps);
            if slot.filling_length == 0 {
                slot.phase = Phase::Decode;
                return Some(Phase::Decode);
            }
        }
        None
    }

    #[test]
    fn schedule_batch_returns_false_for_empty_batch() {
        let batch_list = make_batch_list(Vec::new());
        let scheduler = Scheduler::new(16, 4, 3, batch_list);
        assert!(!scheduler.schedule_batch());
    }

    #[test]
    fn schedule_batch_fills_task_for_decode() {
        let batch_list = make_batch_list(Vec::new());
        let scheduler = Scheduler::new(16, 4, 3, Arc::clone(&batch_list));
        batch_list.with_mut(|batch_list| {
            batch_list.push(SlotState::new_decode_state(100, 128));
        });

        assert!(scheduler.schedule_batch());

        scheduler.with_task(|task| {
            assert_eq!(task.prefill_size, 0);
            assert_eq!(task.decode_size, 1);
        });
    }

    #[test]
    fn set_thread_num_updates_thread_count() {
        let batch_list = make_batch_list(Vec::new());
        let mut scheduler = Scheduler::new(16, 4, 6, batch_list);

        scheduler.set_thread_num(3);
        assert_eq!(scheduler.thread_num(), 3);

        scheduler.set_thread_num(5);
        assert_eq!(scheduler.thread_num(), 5);
    }

    #[test]
    fn test_realistic_batch_sequence_workflow() {
        const MAX_DECODE_SIZE: usize = 8;
        const MAX_PREFILL_SIZE: usize = 512;
        const THREAD_NUM: usize = 4;

        let batch_list = make_batch_list(Vec::new());
        let scheduler = Scheduler::new(
            MAX_DECODE_SIZE,
            MAX_PREFILL_SIZE,
            THREAD_NUM,
            Arc::clone(&batch_list),
        );

        let total_sequences = 5;
        let prefill_token_counts = [64, 128, 32, 96, 48];
        let max_decode_steps = 20;

        batch_list.with_mut(|batch_list| {
            for i in 0..total_sequences {
                batch_list.push(SlotState::new_prefill_state(
                    i * 200,
                    prefill_token_counts[i],
                ));
            }
        });

        let mut tasks = Vec::new();

        assert!(scheduler.schedule_batch());
        scheduler.with_task(|task| {
            assert!(task.prefill_size > 0);
            assert_eq!(task.decode_size, 0);
        });
        tasks.push(scheduler.with_task(|t| t.clone()));

        batch_list.with_mut(|batch_list| {
            for i in 0..total_sequences {
                let phase_change = advance_slot(&mut batch_list[i], prefill_token_counts[i]);
                assert_eq!(phase_change, Some(Phase::Decode));
            }
        });

        for step in 0..max_decode_steps {
            assert!(
                scheduler.schedule_batch(),
                "step {}: should have work",
                step
            );
            scheduler.with_task(|task| {
                assert_eq!(task.decode_size, total_sequences);
                assert_eq!(task.prefill_size, 0);
            });
            tasks.push(scheduler.with_task(|t| t.clone()));
        }

        batch_list.with_mut(|batch_list| {
            for i in 0..total_sequences {
                batch_list[i].phase = Phase::Eos;
            }
        });

        assert!(!scheduler.schedule_batch());
        assert_eq!(tasks.len(), 1 + max_decode_steps);
    }

    #[test]
    fn test_mixed_prefill_decode_workflow() {
        let batch_list = make_batch_list(Vec::new());
        let scheduler = Scheduler::new(8, 256, 2, Arc::clone(&batch_list));

        batch_list.with_mut(|batch_list| {
            for i in 0..3 {
                let mut state = SlotState::new_decode_state(i, i);
                state.phase = Phase::Decode;
                batch_list.push(state);
            }
            for i in 0..2 {
                batch_list.push(SlotState::new_prefill_state(100 + i * 50, 50));
            }
        });

        assert!(scheduler.schedule_batch());
        scheduler.with_task(|task| {
            assert_eq!(task.decode_size, 3);
            assert!(task.prefill_size > 0);
        });
    }

    #[test]
    fn test_chunked_prefill_workflow() {
        const MAX_PREFILL_SIZE: usize = 100;

        let batch_list = make_batch_list(Vec::new());
        let scheduler = Scheduler::new(8, MAX_PREFILL_SIZE, 2, Arc::clone(&batch_list));

        let total_prefill_tokens = 250;
        batch_list.with_mut(|batch_list| {
            batch_list.push(SlotState::new_prefill_state(0, total_prefill_tokens));
        });

        let mut prefill_rounds = 0;
        let mut total_prefilled = 0;

        loop {
            let has_work = scheduler.schedule_batch();
            if !has_work {
                break;
            }

            let prefill_size = scheduler.with_task(|t| t.prefill_size);
            total_prefilled += prefill_size;
            prefill_rounds += 1;

            batch_list.with_mut(|batch_list| {
                advance_slot(&mut batch_list[0], prefill_size);
            });

            if batch_list.with(|bl| bl[0].phase == Phase::Decode) {
                break;
            }
        }

        assert_eq!(total_prefilled, total_prefill_tokens);
        assert_eq!(prefill_rounds, 3);
    }

    #[test]
    fn test_prefilling_chunked_slices_content_validation() {
        let batch_list = make_batch_list(Vec::new());
        let scheduler = Scheduler::new(8, 200, 2, Arc::clone(&batch_list));

        batch_list.with_mut(|batch_list| {
            batch_list.push(SlotState::new_prefill_state(0, 60));
            batch_list.push(SlotState::new_prefill_state(100, 80));
        });

        assert!(scheduler.schedule_batch());
        let task = scheduler.with_task(|t| t.clone());
        assert_eq!(task.prefill_size, 140);
        assert_eq!(task.prefilling_chunked_slices.len(), 2);

        let total_tokens: usize = task
            .prefilling_chunked_slices
            .iter()
            .flat_map(|v| v.iter())
            .map(|s| s.length)
            .sum();
        assert_eq!(total_tokens, 140);

        let mut token_count = 0;
        for thread_slices in &task.prefilling_chunked_slices {
            for slice in thread_slices {
                assert_eq!(slice.token_start_index, token_count);
                if token_count < 60 {
                    assert_eq!(slice.sequence_index, token_count);
                } else {
                    assert_eq!(slice.sequence_index, 100 + (token_count - 60));
                }
                token_count += slice.length;
            }
        }
        assert_eq!(token_count, 140);
    }

    #[test]
    fn test_slices_content_validation() {
        let batch_list = make_batch_list(Vec::new());
        let scheduler = Scheduler::new(8, 256, 2, Arc::clone(&batch_list));

        batch_list.with_mut(|batch_list| {
            for i in 0..3 {
                batch_list.push(SlotState::new_decode_state(i * 10, i * 10));
            }
        });

        assert!(scheduler.schedule_batch());
        let task = scheduler.with_task(|t| t.clone());

        assert_eq!(task.slices.len(), 3);
        for (idx, slice) in task.slices.iter().enumerate() {
            assert_eq!(slice.token_start_index, idx);
            assert_eq!(slice.length, 1);
            assert!(slice.last_token_flag);
            assert_eq!(slice.sequence_index, idx * 10);
        }
    }

    #[test]
    fn test_realistic_prefill_mixed_decode_full_scenario() {
        const MAX_DECODE_SIZE: usize = 16;
        const MAX_PREFILL_SIZE: usize = 512;
        const THREAD_NUM: usize = 4;

        let batch_list = make_batch_list(Vec::new());
        let scheduler = Scheduler::new(
            MAX_DECODE_SIZE,
            MAX_PREFILL_SIZE,
            THREAD_NUM,
            Arc::clone(&batch_list),
        );

        let prefill_len_a = 64usize;
        let prefill_len_b = 48usize;
        batch_list.with_mut(|batch_list| {
            batch_list.push(SlotState::new_prefill_state(0, prefill_len_a));
            batch_list.push(SlotState::new_prefill_state(200, prefill_len_b));
        });

        assert!(scheduler.schedule_batch());
        let task_p1 = scheduler.with_task(|t| t.clone());

        assert_eq!(task_p1.decode_size, 0);
        assert_eq!(task_p1.prefill_size, prefill_len_a + prefill_len_b);

        let dl_p1 = &task_p1.slices;
        assert_eq!(dl_p1.len(), 2);
        assert_eq!(dl_p1[0].batch_index, 0);
        assert_eq!(dl_p1[0].sequence_index, 0);
        assert_eq!(dl_p1[0].token_start_index, 0);
        assert_eq!(dl_p1[0].length, prefill_len_a);
        assert!(dl_p1[0].last_token_flag);
        assert_eq!(dl_p1[1].batch_index, 1);
        assert_eq!(dl_p1[1].sequence_index, 200);
        assert_eq!(dl_p1[1].token_start_index, prefill_len_a);
        assert_eq!(dl_p1[1].length, prefill_len_b);
        assert!(dl_p1[1].last_token_flag);

        let prefill_token_sum: usize = task_p1
            .prefilling_chunked_slices
            .iter()
            .flat_map(|v| v.iter())
            .map(|s| s.length)
            .sum();
        assert_eq!(prefill_token_sum, prefill_len_a + prefill_len_b);

        batch_list.with_mut(|batch_list| {
            let phase_a = advance_slot(&mut batch_list[0], prefill_len_a);
            assert_eq!(phase_a, Some(Phase::Decode));
            let phase_b = advance_slot(&mut batch_list[1], prefill_len_b);
            assert_eq!(phase_b, Some(Phase::Decode));
        });

        let prefill_len_c = 32usize;
        let prefill_len_d = 80usize;
        batch_list.with_mut(|batch_list| {
            batch_list.push(SlotState::new_prefill_state(400, prefill_len_c));
            batch_list.push(SlotState::new_prefill_state(600, prefill_len_d));
        });

        assert!(scheduler.schedule_batch());
        let task_p2 = scheduler.with_task(|t| t.clone());

        assert_eq!(task_p2.decode_size, 2);
        assert_eq!(task_p2.prefill_size, prefill_len_c + prefill_len_d);

        let dl_p2 = &task_p2.slices;
        assert_eq!(dl_p2.len(), 4);

        assert_eq!(dl_p2[0].batch_index, 2);
        assert_eq!(dl_p2[0].sequence_index, 400);
        assert_eq!(dl_p2[0].token_start_index, 0);
        assert_eq!(dl_p2[0].length, prefill_len_c);
        assert!(dl_p2[0].last_token_flag);
        assert_eq!(dl_p2[1].batch_index, 3);
        assert_eq!(dl_p2[1].sequence_index, 600);
        assert_eq!(dl_p2[1].token_start_index, prefill_len_c);
        assert_eq!(dl_p2[1].length, prefill_len_d);
        assert!(dl_p2[1].last_token_flag);

        let expected_decode_offset = prefill_len_c + prefill_len_d;
        assert_eq!(dl_p2[2].batch_index, 0);
        assert_eq!(dl_p2[2].token_start_index, expected_decode_offset);
        assert_eq!(dl_p2[2].length, 1);
        assert!(dl_p2[2].last_token_flag);
        assert_eq!(dl_p2[3].batch_index, 1);
        assert_eq!(dl_p2[3].token_start_index, expected_decode_offset + 1);
        assert_eq!(dl_p2[3].length, 1);
        assert!(dl_p2[3].last_token_flag);

        for i in 1..dl_p2.len() {
            assert!(
                dl_p2[i].token_start_index > dl_p2[i - 1].token_start_index,
                "slices token_start_index should be strictly increasing"
            );
        }

        batch_list.with_mut(|batch_list| {
            advance_slot(&mut batch_list[0], 1);
            advance_slot(&mut batch_list[1], 1);
            let phase_c = advance_slot(&mut batch_list[2], prefill_len_c);
            assert_eq!(phase_c, Some(Phase::Decode));
            let phase_d = advance_slot(&mut batch_list[3], prefill_len_d);
            assert_eq!(phase_d, Some(Phase::Decode));
        });

        assert!(scheduler.schedule_batch());
        let task_p3 = scheduler.with_task(|t| t.clone());

        assert_eq!(task_p3.decode_size, 4);
        assert_eq!(task_p3.prefill_size, 0);

        let dl_p3 = &task_p3.slices;
        assert_eq!(dl_p3.len(), 4);
        for (idx, slice) in dl_p3.iter().enumerate() {
            assert_eq!(slice.token_start_index, idx);
            assert_eq!(slice.length, 1);
            assert!(slice.last_token_flag);
            assert_eq!(slice.batch_index, idx);
        }

        for _ in 0..5 {
            batch_list.with_mut(|batch_list| {
                for s in batch_list.iter_mut() {
                    if s.phase == Phase::Decode {
                        advance_slot(s, 1);
                    }
                }
            });
            assert!(scheduler.schedule_batch());
            scheduler.with_task(|task| {
                assert_eq!(task.decode_size, 4);
            });
        }

        batch_list.with_mut(|batch_list| {
            batch_list[0].phase = Phase::Eos;
            batch_list[1].phase = Phase::Eos;
        });

        assert!(scheduler.schedule_batch());
        scheduler.with_task(|task| {
            assert_eq!(task.decode_size, 2);
        });

        batch_list.with_mut(|batch_list| {
            batch_list[2].phase = Phase::Eos;
            batch_list[3].phase = Phase::Eos;
        });

        assert!(!scheduler.schedule_batch());
    }

    #[test]
    fn test_mixed_mode_slices_token_layout() {
        let batch_list = make_batch_list(Vec::new());
        let scheduler = Scheduler::new(16, 1024, 2, Arc::clone(&batch_list));

        batch_list.with_mut(|batch_list| {
            for i in 0..3 {
                batch_list.push(SlotState::new_decode_state(i * 100, i * 100));
            }
            batch_list.push(SlotState::new_prefill_state(500, 50));
        });

        assert!(scheduler.schedule_batch());
        let task = scheduler.with_task(|t| t.clone());
        assert_eq!(task.prefill_size, 50);
        assert_eq!(task.decode_size, 3);

        let dl = &task.slices;
        assert_eq!(dl.len(), 4);

        assert_eq!(dl[0].batch_index, 3);
        assert_eq!(dl[0].token_start_index, 0);
        assert_eq!(dl[0].length, 50);

        assert_eq!(dl[1].batch_index, 0);
        assert_eq!(dl[1].token_start_index, 50);
        assert_eq!(dl[1].length, 1);

        assert_eq!(dl[2].batch_index, 1);
        assert_eq!(dl[2].token_start_index, 51);
        assert_eq!(dl[2].length, 1);

        assert_eq!(dl[3].batch_index, 2);
        assert_eq!(dl[3].token_start_index, 52);
        assert_eq!(dl[3].length, 1);

        let total: usize = dl.iter().map(|s| s.length).sum();
        assert_eq!(total, 50 + 3);
    }

    #[test]
    fn test_chunked_prefill_with_decode_token_layout() {
        const MAX_PREFILL_SIZE: usize = 100;

        let batch_list = make_batch_list(Vec::new());
        let scheduler = Scheduler::new(8, MAX_PREFILL_SIZE, 2, Arc::clone(&batch_list));

        batch_list.with_mut(|batch_list| {
            batch_list.push(SlotState::new_decode_state(0, 0));
            batch_list.push(SlotState::new_decode_state(50, 50));
            batch_list.push(SlotState::new_prefill_state(200, 250));
        });

        assert!(scheduler.schedule_batch());
        let task = scheduler.with_task(|t| t.clone());
        assert_eq!(task.prefill_size, MAX_PREFILL_SIZE);
        assert_eq!(task.decode_size, 2);

        let dl = &task.slices;
        assert_eq!(dl.len(), 3);

        assert_eq!(dl[0].batch_index, 2);
        assert_eq!(dl[0].token_start_index, 0);
        assert_eq!(dl[0].length, MAX_PREFILL_SIZE);
        assert!(!dl[0].last_token_flag);

        assert_eq!(dl[1].batch_index, 0);
        assert_eq!(dl[1].token_start_index, MAX_PREFILL_SIZE);
        assert_eq!(dl[1].length, 1);
        assert!(dl[1].last_token_flag);

        assert_eq!(dl[2].batch_index, 1);
        assert_eq!(dl[2].token_start_index, MAX_PREFILL_SIZE + 1);
        assert_eq!(dl[2].length, 1);
        assert!(dl[2].last_token_flag);
    }

    #[test]
    fn test_mixed_slices_lookup() {
        let batch_list = make_batch_list(Vec::new());
        let scheduler = Scheduler::new(16, 1024, 2, Arc::clone(&batch_list));

        batch_list.with_mut(|batch_list| {
            batch_list.push(SlotState::new_decode_state(0, 0));
            batch_list.push(SlotState::new_decode_state(100, 100));
            batch_list.push(SlotState::new_prefill_state(300, 40));
        });

        assert!(scheduler.schedule_batch());
        let task = scheduler.with_task(|t| t.clone());

        let dl = &task.slices;

        let r0 = crate::runtime::batch::lookup_global_index(dl, 0).unwrap();
        assert_eq!(r0.batch_index, 2);
        assert_eq!(r0.sequence_index, 300);

        let r39 = crate::runtime::batch::lookup_global_index(dl, 39).unwrap();
        assert_eq!(r39.batch_index, 2);
        assert_eq!(r39.sequence_index, 339);

        let r40 = crate::runtime::batch::lookup_global_index(dl, 40).unwrap();
        assert_eq!(r40.batch_index, 0);
        assert_eq!(r40.sequence_index, 0);

        let r41 = crate::runtime::batch::lookup_global_index(dl, 41).unwrap();
        assert_eq!(r41.batch_index, 1);
        assert_eq!(r41.sequence_index, 100);

        assert!(crate::runtime::batch::lookup_global_index(dl, 42).is_none());
    }
}
