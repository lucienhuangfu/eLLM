use super::*;
use crate::runtime::state::core::SlotState;
use crate::runtime::state::types::Phase;
use std::sync::Arc;

/// 模拟真实情况：输入 batch sequence，进行 prefill，然后连续 decode，最后结束
#[test]
fn test_realistic_batch_sequence_workflow() {
    const MAX_DECODE_SIZE: usize = 8;
    const MAX_PREFILL_SIZE: usize = 512;
    const THREAD_NUM: usize = 4;

    let batch_list = Arc::new(crate::operators::send_sync_ptr::SharedMut::new(Vec::new()));
    let scheduler = Scheduler::new(
        MAX_DECODE_SIZE,
        MAX_PREFILL_SIZE,
        THREAD_NUM,
        Arc::clone(&batch_list),
    );
    let shared_state = scheduler.shared_state();

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
    shared_state.task().with(|task| {
        assert_eq!(task.mode, BatchMode::Prefill);
        assert!(task.prefill_size > 0);
        assert_eq!(task.decode_size, 0);
    });
    tasks.push(shared_state.task().with(|t| t.clone()));

    batch_list.with_mut(|batch_list| {
        for i in 0..total_sequences {
            let phase_change = batch_list[i].advance_sequence(prefill_token_counts[i]);
            assert_eq!(phase_change, Some(Phase::Decode));
            assert_eq!(batch_list[i].phase, Phase::Decode);
            assert_eq!(batch_list[i].filling_length, 0);
        }
    });

    for step in 0..max_decode_steps {
        assert!(scheduler.schedule_batch(), "步骤 {}: 计划应该存在", step);
        shared_state.task().with(|task| {
            assert_eq!(task.mode, BatchMode::Decode);
            assert_eq!(task.decode_size, total_sequences);
            assert_eq!(task.prefill_size, 0);
        });
        tasks.push(shared_state.task().with(|t| t.clone()));
    }

    batch_list.with_mut(|batch_list| {
        for i in 0..total_sequences {
            let result = batch_list[i].transition_to_eos();
            assert!(result.is_ok(), "Sequence {} 转换到 EOS 失败", i);
            assert_eq!(batch_list[i].phase, Phase::Eos);
        }
    });

    assert!(
        !scheduler.schedule_batch(),
        "所有 sequence 结束后，计划应该为空"
    );

    assert_eq!(tasks.len(), 1 + max_decode_steps);
    let mut prev_task_id = 0;
    for (idx, task) in tasks.iter().enumerate() {
        assert!(task.task_id > prev_task_id, "任务 ID 应该递增");
        prev_task_id = task.task_id;

        if idx == 0 {
            assert_eq!(
                task.prefill_size,
                prefill_token_counts.iter().sum::<usize>()
            );
            assert_eq!(task.decode_size, 0);
        } else {
            assert_eq!(task.prefill_size, 0);
            assert_eq!(task.decode_size, total_sequences);
        }
    }
}

/// 模拟更复杂的真实场景：混合 prefill 和 decode
#[test]
fn test_mixed_prefill_decode_workflow() {
    let batch_list = Arc::new(crate::operators::send_sync_ptr::SharedMut::new(Vec::new()));
    let scheduler = Scheduler::new(8, 256, 2, Arc::clone(&batch_list));
    let shared_state = scheduler.shared_state();

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
    shared_state.task().with(|task| {
        assert_eq!(task.mode, BatchMode::Mixed);
        assert_eq!(task.decode_size, 3);
        assert!(task.prefill_size > 0);
    });
    let task = shared_state.task().with(|t| t.clone());
    assert_eq!(task.decode_size, 3);
    assert!(task.prefill_size > 0);

    batch_list.with_mut(|batch_list| {
        for i in 0..3 {
            let _ = batch_list[i].advance_sequence(1);
        }
        for i in 3..5 {
            let remaining = batch_list[i].filling_length;
            let phase_change = batch_list[i].advance_sequence(remaining);
            assert_eq!(phase_change, Some(Phase::Decode));
            assert_eq!(batch_list[i].phase, Phase::Decode);
        }
    });

    let active_count = batch_list.with(|bl| bl.iter().filter(|s| s.phase == Phase::Decode).count());
    assert_eq!(active_count, 5);

    assert!(scheduler.schedule_batch());
    shared_state.task().with(|task| {
        assert_eq!(task.mode, BatchMode::Decode);
        assert_eq!(task.decode_size, 5);
    });
}

/// 模拟真实场景：新请求到达时的调度
#[test]
fn test_new_requests_during_decode() {
    let batch_list = Arc::new(crate::operators::send_sync_ptr::SharedMut::new(Vec::new()));
    let scheduler = Scheduler::new(8, 512, 4, Arc::clone(&batch_list));
    let shared_state = scheduler.shared_state();

    batch_list.with_mut(|batch_list| {
        for i in 0..4 {
            let mut state = SlotState::new_decode_state(i, i);
            state.phase = Phase::Decode;
            batch_list.push(state);
        }
    });

    for step in 0..3 {
        assert!(scheduler.schedule_batch());
        shared_state.task().with(|task| {
            assert_eq!(task.mode, BatchMode::Decode);
            assert_eq!(task.decode_size, 4);
        });

        batch_list.with_mut(|batch_list| {
            for i in 0..batch_list.len() {
                if batch_list[i].phase == Phase::Decode {
                    let _ = batch_list[i].advance_sequence(1);
                }
            }
        });
    }

    batch_list.with_mut(|batch_list| {
        batch_list.push(SlotState::new_prefill_state(100, 64));
        batch_list.push(SlotState::new_prefill_state(200, 32));
    });

    assert!(scheduler.schedule_batch());
    shared_state.task().with(|task| {
        assert_eq!(task.mode, BatchMode::Mixed);
        assert_eq!(task.decode_size, 4);
        assert!(task.prefill_size > 0);
    });

    batch_list.with_mut(|batch_list| {
        for i in 0..batch_list.len() {
            if batch_list[i].phase == Phase::Decode {
                let _ = batch_list[i].advance_sequence(1);
            } else if batch_list[i].phase == Phase::Prefill {
                let remaining = batch_list[i].filling_length;
                let _ = batch_list[i].advance_sequence(remaining);
            }
        }
    });

    assert!(scheduler.schedule_batch());
    shared_state.task().with(|task| {
        assert_eq!(task.mode, BatchMode::Decode);
        assert_eq!(task.decode_size, 6);
    });
}

/// 模拟真实场景：部分序列提前结束
#[test]
fn test_partial_sequence_completion() {
    let batch_list = Arc::new(crate::operators::send_sync_ptr::SharedMut::new(Vec::new()));
    let scheduler = Scheduler::new(8, 512, 4, Arc::clone(&batch_list));
    let shared_state = scheduler.shared_state();

    batch_list.with_mut(|batch_list| {
        for i in 0..5 {
            let mut state = SlotState::new_decode_state(i, i);
            state.phase = Phase::Decode;
            batch_list.push(state);
        }
    });

    for step in 0..10 {
        let has_work = scheduler.schedule_batch();

        let active_count =
            batch_list.with(|bl| bl.iter().filter(|s| s.phase == Phase::Decode).count());

        if active_count > 0 {
            assert!(has_work);
            shared_state.task().with(|task| {
                assert_eq!(task.decode_size, active_count);
            });
        } else {
            assert!(!has_work);
            break;
        }

        if step == 2 {
            batch_list.with_mut(|batch_list| {
                let _ = batch_list[0].transition_to_eos();
            });
        }

        if step == 4 {
            batch_list.with_mut(|batch_list| {
                let _ = batch_list[2].transition_to_eos();
                let _ = batch_list[3].transition_to_eos();
            });
        }

        batch_list.with_mut(|batch_list| {
            for i in 0..batch_list.len() {
                if batch_list[i].phase == Phase::Decode {
                    let _ = batch_list[i].advance_sequence(1);
                }
            }
        });
    }

    let active_count = batch_list.with(|bl| bl.iter().filter(|s| s.phase == Phase::Decode).count());
    assert_eq!(active_count, 2);
}

/// 模拟真实场景：分块预填充（filling_length 超过 max_prefill_size）
#[test]
fn test_chunked_prefill_workflow() {
    const MAX_DECODE_SIZE: usize = 8;
    const MAX_PREFILL_SIZE: usize = 100;
    const THREAD_NUM: usize = 2;

    let batch_list = Arc::new(crate::operators::send_sync_ptr::SharedMut::new(Vec::new()));
    let scheduler = Scheduler::new(
        MAX_DECODE_SIZE,
        MAX_PREFILL_SIZE,
        THREAD_NUM,
        Arc::clone(&batch_list),
    );
    let shared_state = scheduler.shared_state();

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

        shared_state.task().with(|task| {
            assert_eq!(task.mode, BatchMode::Prefill);
            assert_eq!(task.decode_size, 0);
            assert!(task.prefill_size > 0);
        });

        let prefill_size = shared_state.task().with(|t| t.prefill_size);
        total_prefilled += prefill_size;
        prefill_rounds += 1;

        batch_list.with_mut(|batch_list| {
            let phase_change = batch_list[0].advance_sequence(prefill_size);

            if batch_list[0].filling_length == 0 {
                assert_eq!(phase_change, Some(Phase::Decode));
            } else {
                assert_eq!(phase_change, None);
            }
        });

        if batch_list.with(|bl| bl[0].phase == Phase::Decode) {
            break;
        }
    }

    assert_eq!(total_prefilled, total_prefill_tokens);
    assert_eq!(prefill_rounds, 3);

    assert!(scheduler.schedule_batch());
    shared_state.task().with(|task| {
        assert_eq!(task.mode, BatchMode::Decode);
        assert_eq!(task.decode_size, 1);
        assert_eq!(task.prefill_size, 0);
    });
}

/// 模拟真实场景：槽位重用（EOS 后重置并开始新请求）
#[test]
fn test_slot_reuse_workflow() {
    let batch_list = Arc::new(crate::operators::send_sync_ptr::SharedMut::new(Vec::new()));
    let scheduler = Scheduler::new(4, 256, 2, Arc::clone(&batch_list));
    let shared_state = scheduler.shared_state();

    batch_list.with_mut(|batch_list| {
        batch_list.push(SlotState::new_decode_state(0, 0));
    });

    assert!(scheduler.schedule_batch());
    shared_state.task().with(|task| {
        assert_eq!(task.mode, BatchMode::Decode);
        assert_eq!(task.decode_size, 1);
    });

    batch_list.with_mut(|batch_list| {
        let _ = batch_list[0].transition_to_eos();
    });

    assert!(!scheduler.schedule_batch());

    batch_list.with_mut(|batch_list| {
        batch_list[0].reset_to_start();
        let _ = batch_list[0].transition_to_prefill(100, 50);
    });

    assert_eq!(batch_list.with(|bl| bl[0].phase), Phase::Prefill);
    assert_eq!(batch_list.with(|bl| bl[0].sequence_index), 100);
    assert_eq!(batch_list.with(|bl| bl[0].filling_length), 50);

    assert!(scheduler.schedule_batch());
    shared_state.task().with(|task| {
        assert_eq!(task.mode, BatchMode::Prefill);
        assert!(task.prefill_size > 0);
    });
}

/// 测试预填充列表内容的正确性
#[test]
fn test_prefill_list_content_validation() {
    let batch_list = Arc::new(crate::operators::send_sync_ptr::SharedMut::new(Vec::new()));
    let scheduler = Scheduler::new(8, 200, 2, Arc::clone(&batch_list));
    let shared_state = scheduler.shared_state();

    batch_list.with_mut(|batch_list| {
        batch_list.push(SlotState::new_prefill_state(0, 60));
        batch_list.push(SlotState::new_prefill_state(100, 80));
    });

    assert!(scheduler.schedule_batch());
    shared_state.task().with(|task| {
        assert_eq!(task.mode, BatchMode::Prefill);
        assert_eq!(task.prefill_size, 140);
        assert_eq!(task.prefill_list.len(), 2);
    });

    let task = shared_state.task().with(|t| t.clone());
    assert_eq!(task.prefill_list.len(), 2);

    let total_tokens: usize = task
        .prefill_list
        .iter()
        .flat_map(|v| v.iter())
        .map(|s| s.length)
        .sum();
    assert_eq!(total_tokens, 140);

    let mut token_count = 0;

    for thread_slices in &task.prefill_list {
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

/// 测试解码列表内容的正确性
#[test]
fn test_decode_list_content_validation() {
    let batch_list = Arc::new(crate::operators::send_sync_ptr::SharedMut::new(Vec::new()));
    let scheduler = Scheduler::new(8, 256, 2, Arc::clone(&batch_list));
    let shared_state = scheduler.shared_state();

    batch_list.with_mut(|batch_list| {
        for i in 0..3 {
            batch_list.push(SlotState::new_decode_state(i * 10, i * 10));
        }
    });

    assert!(scheduler.schedule_batch());
    shared_state.task().with(|task| {
        assert_eq!(task.mode, BatchMode::Decode);
        assert_eq!(task.decode_size, 3);
    });

    let task = shared_state.task().with(|t| t.clone());

    assert_eq!(task.decode_list.len(), 3);

    for (idx, slice) in task.decode_list.as_slice().iter().enumerate() {
        assert_eq!(slice.token_start_index, idx);
        assert_eq!(slice.length, 1);
        assert!(slice.last_token_flag);
        assert_eq!(slice.sequence_index, idx * 10);
    }
}
