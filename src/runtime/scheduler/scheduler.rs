use std::sync::Arc;

use super::task::{ScheduleTask, SequenceSlice};
use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::session::{Phase, SlotState};

pub struct Scheduler {
    max_batch_size: usize,
    max_chunk_size: usize,
    thread_num: usize,
    slot_list: Arc<SharedMut<Vec<SlotState>>>,
    task: SharedMut<ScheduleTask>,
}

unsafe impl Send for Scheduler {}
unsafe impl Sync for Scheduler {}

impl Scheduler {
    pub fn new(
        max_batch_size: usize,
        max_chunk_size: usize,
        thread_num: usize,
        slot_list: Arc<SharedMut<Vec<SlotState>>>,
    ) -> Self {
        let thread_num = thread_num.max(1);
        Self {
            max_batch_size,
            max_chunk_size,
            thread_num,
            slot_list,
            task: SharedMut::new(ScheduleTask::new(thread_num, max_batch_size)),
        }
    }

    #[inline]
    pub fn slot_list(&self) -> Arc<SharedMut<Vec<SlotState>>> {
        Arc::clone(&self.slot_list)
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

    /// 两遍遍历 + decode 优先，零后处理、零中间结构。
    ///
    /// Pass 1（极轻量）：decode 优先占用 chunk 预算，剩余给 prefill，
    ///         仅做整数比较/减法，无写入、无分配。
    /// Pass 2：直接构建 Vec<SequenceSlice>，token_start_index 当场正确，
    ///         输出布局 [Prefill 0..prefill_total][Decode prefill_total..total]。
    #[inline]
    pub fn schedule_batch(&self) -> bool {
        let max_batch_size = self.max_batch_size;
        let max_chunk_size = self.max_chunk_size;

        self.slot_list.with(|slot_list| {
            self.task.with_mut(|task| {
                task.reset();

                // ===== Pass 1: decode 优先，确定预算分配 =====
                let mut decode_count = 0usize;
                for slot in slot_list.iter() {
                    if slot.phase == Phase::Decode && decode_count < max_batch_size {
                        decode_count += 1;
                    }
                }
                // decode 占 chunk 的一小部分，剩余给 prefill
                let prefill_budget = max_chunk_size.saturating_sub(decode_count);

                // 模拟 prefill 消耗（chunked prefill 可能跨多轮）
                let mut budget = prefill_budget;
                let mut prefill_total = 0usize;
                for slot in slot_list.iter() {
                    if budget == 0 {
                        break;
                    }
                    if slot.phase == Phase::Prefill {
                        let remaining = slot.prompt_length.saturating_sub(slot.next_sequence_index);
                        let len = remaining.min(budget);
                        prefill_total += len;
                        budget -= len;
                    }
                }

                if prefill_total == 0 && decode_count == 0 {
                    return;
                }

                // ===== Pass 2: 直接构建 slices，零后处理 =====
                // 布局约定：[Prefill 0..prefill_total][Decode prefill_total..total]
                let mut prefill_acc = 0usize;
                let mut remaining_budget = prefill_budget;

                // 先推 Prefill 切片（保证在前）
                for (batch_index, slot) in slot_list.iter().enumerate() {
                    if remaining_budget == 0 {
                        break;
                    }
                    if slot.phase == Phase::Prefill {
                        let remaining = slot.prompt_length.saturating_sub(slot.next_sequence_index);
                        let prefill_length = remaining.min(remaining_budget);

                        task.slices.push(SequenceSlice {
                            batch_index,
                            next_sequence_index: slot.next_sequence_index,
                            token_start_index: prefill_acc,
                            length: prefill_length,
                            last_token_flag: prefill_length == remaining,
                            lift_index: 0,
                        });
                        prefill_acc += prefill_length;
                        remaining_budget -= prefill_length;
                    }
                }

                // 再推 Decode 切片（保证在后）
                let mut decode_acc = 0usize;
                for (batch_index, slot) in slot_list.iter().enumerate() {
                    if decode_acc >= decode_count {
                        break;
                    }
                    if slot.phase == Phase::Decode {
                        task.slices.push(SequenceSlice {
                            batch_index,
                            next_sequence_index: slot.next_sequence_index,
                            token_start_index: prefill_total + decode_acc,
                            length: 1,
                            last_token_flag: true,
                            lift_index: 0,
                        });
                        decode_acc += 1;
                    }
                }

                task.prefill_size = prefill_acc;
                task.decode_size = decode_acc;
                task.total_size = prefill_acc + decode_acc;
            });
            !self.task.with(|task| task.is_empty())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prefill_state(next_sequence_index: usize, filling_length: usize) -> SlotState {
        let mut s = SlotState::idle();
        s.start_prefill(next_sequence_index, filling_length);
        s
    }

    fn decode_state(next_sequence_index: usize) -> SlotState {
        let mut s = SlotState::idle();
        s.start_decode(next_sequence_index, next_sequence_index);
        s
    }

    fn make_slot_list(slots: Vec<SlotState>) -> Arc<SharedMut<Vec<SlotState>>> {
        Arc::new(SharedMut::new(slots))
    }

    fn advance_slot(slot: &mut SlotState, steps: usize) -> Option<Phase> {
        if slot.phase == Phase::Eos {
            return None;
        }
        slot.next_sequence_index += steps;
        if slot.phase == Phase::Prefill {
            if slot.filling_length() == 0 {
                slot.phase = Phase::Decode;
                return Some(Phase::Decode);
            }
        } else {
            slot.sequence_length += steps;
        }
        None
    }

    #[test]
    fn schedule_batch_returns_false_for_empty_batch() {
        let slot_list = make_slot_list(Vec::new());
        let scheduler = Scheduler::new(16, 4, 3, slot_list);
        assert!(!scheduler.schedule_batch());
    }

    #[test]
    fn schedule_batch_fills_task_for_decode() {
        let slot_list = make_slot_list(Vec::new());
        let scheduler = Scheduler::new(16, 4, 3, Arc::clone(&slot_list));
        slot_list.with_mut(|slot_list| {
            slot_list.push(decode_state(100));
        });

        assert!(scheduler.schedule_batch());

        scheduler.with_task(|task| {
            assert_eq!(task.prefill_size, 0);
            assert_eq!(task.decode_size, 1);
        });
    }

    #[test]
    fn test_realistic_batch_sequence_workflow() {
        const MAX_BATCH_SIZE: usize = 8;
        const MAX_CHUNK_SIZE: usize = 512;
        const THREAD_NUM: usize = 4;

        let slot_list = make_slot_list(Vec::new());
        let scheduler = Scheduler::new(
            MAX_BATCH_SIZE,
            MAX_CHUNK_SIZE,
            THREAD_NUM,
            Arc::clone(&slot_list),
        );

        let total_sequences = 5;
        let prefill_sequence_lengths = [64, 128, 32, 96, 48];
        let max_decode_steps = 20;

        slot_list.with_mut(|slot_list| {
            for i in 0..total_sequences {
                slot_list.push(prefill_state(i * 200, prefill_sequence_lengths[i]));
            }
        });

        let mut tasks = Vec::new();

        assert!(scheduler.schedule_batch());
        scheduler.with_task(|task| {
            assert!(task.prefill_size > 0);
            assert_eq!(task.decode_size, 0);
        });
        tasks.push(scheduler.with_task(|t| t.clone()));

        slot_list.with_mut(|slot_list| {
            for i in 0..total_sequences {
                let phase_change = advance_slot(&mut slot_list[i], prefill_sequence_lengths[i]);
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

        slot_list.with_mut(|slot_list| {
            for i in 0..total_sequences {
                slot_list[i].phase = Phase::Eos;
            }
        });

        assert!(!scheduler.schedule_batch());
        assert_eq!(tasks.len(), 1 + max_decode_steps);
    }

    #[test]
    fn test_mixed_prefill_decode_workflow() {
        let slot_list = make_slot_list(Vec::new());
        let scheduler = Scheduler::new(8, 256, 2, Arc::clone(&slot_list));

        slot_list.with_mut(|slot_list| {
            for i in 0..3 {
                let mut state = decode_state(i);
                state.phase = Phase::Decode;
                slot_list.push(state);
            }
            for i in 0..2 {
                slot_list.push(prefill_state(100 + i * 50, 50));
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
        const MAX_CHUNK_SIZE: usize = 100;

        let slot_list = make_slot_list(Vec::new());
        let scheduler = Scheduler::new(8, MAX_CHUNK_SIZE, 2, Arc::clone(&slot_list));

        let total_prefill_tokens = 250;
        slot_list.with_mut(|slot_list| {
            slot_list.push(prefill_state(0, total_prefill_tokens));
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

            slot_list.with_mut(|slot_list| {
                advance_slot(&mut slot_list[0], prefill_size);
            });

            if slot_list.with(|bl| bl[0].phase == Phase::Decode) {
                break;
            }
        }

        assert_eq!(total_prefilled, total_prefill_tokens);
        assert_eq!(prefill_rounds, 3);
    }

    #[test]
    fn test_prefill_slices_content_validation() {
        let slot_list = make_slot_list(Vec::new());
        let scheduler = Scheduler::new(8, 200, 2, Arc::clone(&slot_list));

        slot_list.with_mut(|slot_list| {
            slot_list.push(prefill_state(0, 60));
            slot_list.push(prefill_state(100, 80));
        });

        assert!(scheduler.schedule_batch());
        let task = scheduler.with_task(|t| t.clone());
        assert_eq!(task.prefill_size, 140);

        let total_tokens: usize = task
            .slices
            .iter()
            .take_while(|s| s.token_start_index < task.prefill_size)
            .map(|s| {
                s.length
                    .min(task.prefill_size.saturating_sub(s.token_start_index))
            })
            .sum();
        assert_eq!(total_tokens, 140);

        let mut sequence_length = 0;
        for slice in &task.slices {
            if slice.token_start_index >= task.prefill_size {
                break;
            }
            let effective_len = slice
                .length
                .min(task.prefill_size.saturating_sub(slice.token_start_index));
            for t in 0..effective_len {
                let global_tok = slice.token_start_index + t;
                assert_eq!(global_tok, sequence_length);
                if sequence_length < 60 {
                    assert_eq!(slice.next_sequence_index + t, sequence_length);
                } else {
                    assert_eq!(slice.next_sequence_index + t, 100 + (sequence_length - 60));
                }
                sequence_length += 1;
            }
        }
        assert_eq!(sequence_length, 140);
    }

    #[test]
    fn test_slices_content_validation() {
        let slot_list = make_slot_list(Vec::new());
        let scheduler = Scheduler::new(8, 256, 2, Arc::clone(&slot_list));

        slot_list.with_mut(|slot_list| {
            for i in 0..3 {
                slot_list.push(decode_state(i * 10));
            }
        });

        assert!(scheduler.schedule_batch());
        let task = scheduler.with_task(|t| t.clone());

        assert_eq!(task.slices.len(), 3);
        for (idx, slice) in task.slices.iter().enumerate() {
            assert_eq!(slice.token_start_index, idx);
            assert_eq!(slice.length, 1);
            assert!(slice.last_token_flag);
            assert_eq!(slice.next_sequence_index, idx * 10);
        }
    }

    #[test]
    fn test_realistic_prefill_mixed_decode_full_scenario() {
        const MAX_BATCH_SIZE: usize = 16;
        const MAX_CHUNK_SIZE: usize = 512;
        const THREAD_NUM: usize = 4;

        let slot_list = make_slot_list(Vec::new());
        let scheduler = Scheduler::new(
            MAX_BATCH_SIZE,
            MAX_CHUNK_SIZE,
            THREAD_NUM,
            Arc::clone(&slot_list),
        );

        let prefill_length_a = 64usize;
        let prefill_length_b = 48usize;
        slot_list.with_mut(|slot_list| {
            slot_list.push(prefill_state(0, prefill_length_a));
            slot_list.push(prefill_state(200, prefill_length_b));
        });

        assert!(scheduler.schedule_batch());
        let task_p1 = scheduler.with_task(|t| t.clone());

        assert_eq!(task_p1.decode_size, 0);
        assert_eq!(task_p1.prefill_size, prefill_length_a + prefill_length_b);

        let dl_p1 = &task_p1.slices;
        assert_eq!(dl_p1.len(), 2);
        assert_eq!(dl_p1[0].batch_index, 0);
        assert_eq!(dl_p1[0].next_sequence_index, 0);
        assert_eq!(dl_p1[0].token_start_index, 0);
        assert_eq!(dl_p1[0].length, prefill_length_a);
        assert!(dl_p1[0].last_token_flag);
        assert_eq!(dl_p1[1].batch_index, 1);
        assert_eq!(dl_p1[1].next_sequence_index, 200);
        assert_eq!(dl_p1[1].token_start_index, prefill_length_a);
        assert_eq!(dl_p1[1].length, prefill_length_b);
        assert!(dl_p1[1].last_token_flag);

        let prefill_token_sum: usize = task_p1
            .slices
            .iter()
            .filter(|s| s.token_start_index < task_p1.prefill_size)
            .map(|s| {
                s.length
                    .min(task_p1.prefill_size.saturating_sub(s.token_start_index))
            })
            .sum();
        assert_eq!(prefill_token_sum, prefill_length_a + prefill_length_b);

        slot_list.with_mut(|slot_list| {
            let phase_a = advance_slot(&mut slot_list[0], prefill_length_a);
            assert_eq!(phase_a, Some(Phase::Decode));
            let phase_b = advance_slot(&mut slot_list[1], prefill_length_b);
            assert_eq!(phase_b, Some(Phase::Decode));
        });

        let prefill_length_c = 32usize;
        let prefill_length_d = 80usize;
        slot_list.with_mut(|slot_list| {
            slot_list.push(prefill_state(400, prefill_length_c));
            slot_list.push(prefill_state(600, prefill_length_d));
        });

        assert!(scheduler.schedule_batch());
        let task_p2 = scheduler.with_task(|t| t.clone());

        assert_eq!(task_p2.decode_size, 2);
        assert_eq!(task_p2.prefill_size, prefill_length_c + prefill_length_d);

        let dl_p2 = &task_p2.slices;
        assert_eq!(dl_p2.len(), 4);

        // Prefill 段 deque 前端 + 顺序保持原 slot_list 顺序（slot2、slot3）
        assert_eq!(dl_p2[0].batch_index, 2);
        assert_eq!(dl_p2[0].next_sequence_index, 400);
        assert_eq!(dl_p2[0].token_start_index, 0);
        assert_eq!(dl_p2[0].length, prefill_length_c);
        assert!(dl_p2[0].last_token_flag);
        assert_eq!(dl_p2[1].batch_index, 3);
        assert_eq!(dl_p2[1].next_sequence_index, 600);
        assert_eq!(dl_p2[1].token_start_index, prefill_length_c);
        assert_eq!(dl_p2[1].length, prefill_length_d);
        assert!(dl_p2[1].last_token_flag);

        let expected_decode_offset = prefill_length_c + prefill_length_d;
        assert_eq!(dl_p2[2].batch_index, 0);
        assert_eq!(dl_p2[2].token_start_index, expected_decode_offset + 0);
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

        slot_list.with_mut(|slot_list| {
            advance_slot(&mut slot_list[0], 1);
            advance_slot(&mut slot_list[1], 1);
            let phase_c = advance_slot(&mut slot_list[2], prefill_length_c);
            assert_eq!(phase_c, Some(Phase::Decode));
            let phase_d = advance_slot(&mut slot_list[3], prefill_length_d);
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
            slot_list.with_mut(|slot_list| {
                for s in slot_list.iter_mut() {
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

        slot_list.with_mut(|slot_list| {
            slot_list[0].phase = Phase::Eos;
            slot_list[1].phase = Phase::Eos;
        });

        assert!(scheduler.schedule_batch());
        scheduler.with_task(|task| {
            assert_eq!(task.decode_size, 2);
        });

        slot_list.with_mut(|slot_list| {
            slot_list[2].phase = Phase::Eos;
            slot_list[3].phase = Phase::Eos;
        });

        assert!(!scheduler.schedule_batch());
    }

    #[test]
    fn test_mixed_mode_slices_token_layout() {
        let slot_list = make_slot_list(Vec::new());
        let scheduler = Scheduler::new(16, 1024, 2, Arc::clone(&slot_list));

        slot_list.with_mut(|slot_list| {
            for i in 0..3 {
                slot_list.push(decode_state(i * 100));
            }
            slot_list.push(prefill_state(500, 50));
        });

        assert!(scheduler.schedule_batch());
        let task = scheduler.with_task(|t| t.clone());
        assert_eq!(task.prefill_size, 50);
        assert_eq!(task.decode_size, 3);

        let dl = &task.slices;
        assert_eq!(dl.len(), 4);

        // Prefill 段 push_front，在前面；Decode 段 push_back，在后面
        assert_eq!(dl[0].batch_index, 3);
        assert_eq!(dl[0].token_start_index, 0);
        assert_eq!(dl[0].length, 50);

        assert_eq!(dl[1].batch_index, 0);
        assert_eq!(dl[1].token_start_index, 50 + 0);
        assert_eq!(dl[1].length, 1);

        assert_eq!(dl[2].batch_index, 1);
        assert_eq!(dl[2].token_start_index, 50 + 1);
        assert_eq!(dl[2].length, 1);

        assert_eq!(dl[3].batch_index, 2);
        assert_eq!(dl[3].token_start_index, 50 + 2);
        assert_eq!(dl[3].length, 1);

        let total: usize = dl.iter().map(|s| s.length).sum();
        assert_eq!(total, 50 + 3);
    }

    #[test]
    fn test_chunked_prefill_with_decode_token_layout() {
        const MAX_CHUNK_SIZE: usize = 100;

        let slot_list = make_slot_list(Vec::new());
        let scheduler = Scheduler::new(8, MAX_CHUNK_SIZE, 2, Arc::clone(&slot_list));

        slot_list.with_mut(|slot_list| {
            slot_list.push(decode_state(0));
            slot_list.push(decode_state(50));
            slot_list.push(prefill_state(200, 250));
        });

        assert!(scheduler.schedule_batch());
        let task = scheduler.with_task(|t| t.clone());
        // decode 优先占用 2 token，prefill 预算 = 100 - 2 = 98
        let expected_prefill = MAX_CHUNK_SIZE - 2;
        assert_eq!(task.prefill_size, expected_prefill);
        assert_eq!(task.decode_size, 2);

        let dl = &task.slices;
        assert_eq!(dl.len(), 3);

        assert_eq!(dl[0].batch_index, 2);
        assert_eq!(dl[0].token_start_index, 0);
        assert_eq!(dl[0].length, expected_prefill);
        assert!(!dl[0].last_token_flag);

        assert_eq!(dl[1].batch_index, 0);
        assert_eq!(dl[1].token_start_index, expected_prefill + 0);
        assert_eq!(dl[1].length, 1);
        assert!(dl[1].last_token_flag);

        assert_eq!(dl[2].batch_index, 1);
        assert_eq!(dl[2].token_start_index, expected_prefill + 1);
        assert_eq!(dl[2].length, 1);
        assert!(dl[2].last_token_flag);
    }
}
