use std::sync::Arc;
use std::time::Instant;

use crate::runtime::state::sequence::{DecodeList, SequenceSlice};

#[derive(Debug, Clone)]
pub struct ScheduleTask {
    pub prefill_size: usize,
    pub decode_size: usize,
    pub prefill_list: Arc<Vec<Vec<SequenceSlice>>>,
    pub decode_list: Arc<DecodeList>,
    pub timestamp: Instant,
    pub task_id: u64,
}

impl ScheduleTask {
    pub fn new(
        prefill_size: usize,
        decode_size: usize,
        prefill_list: Vec<Vec<SequenceSlice>>,
        decode_list: DecodeList,
        task_id: u64,
    ) -> Self {
        Self {
            prefill_size,
            decode_size,
            prefill_list: Arc::new(prefill_list),
            decode_list: Arc::new(decode_list),
            timestamp: Instant::now(),
            task_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试 ScheduleTask::new 创建
    #[test]
    fn test_schedule_task_new() {
        let task = ScheduleTask::new(10, 5, Vec::new(), DecodeList::with_capacity(0), 1);

        assert_eq!(task.prefill_size, 10);
        assert_eq!(task.decode_size, 5);
        assert_eq!(task.task_id, 1);
    }

    /// 测试 ScheduleTask 空任务
    #[test]
    fn test_schedule_task_empty() {
        let task = ScheduleTask::new(0, 0, Vec::new(), DecodeList::with_capacity(0), 0);

        assert_eq!(task.prefill_size, 0);
        assert_eq!(task.decode_size, 0);
        assert!(task.prefill_list.is_empty());
        assert!(task.decode_list.is_empty());
    }

    /// 测试 ScheduleTask 仅 prefill
    #[test]
    fn test_schedule_task_prefill_only() {
        let prefill_list = vec![
            vec![SequenceSlice {
                batch_index: 0,
                sequence_index: 0,
                token_start_index: 0,
                length: 10,
                last_token_flag: false,
            }],
        ];

        let task = ScheduleTask::new(10, 0, prefill_list, DecodeList::with_capacity(0), 1);

        assert_eq!(task.prefill_size, 10);
        assert_eq!(task.decode_size, 0);
        assert_eq!(task.prefill_list.len(), 1);
    }

    /// 测试 ScheduleTask 仅 decode
    #[test]
    fn test_schedule_task_decode_only() {
        let mut decode_list = DecodeList::with_capacity(5);
        for i in 0..5 {
            decode_list.push(SequenceSlice {
                batch_index: i,
                sequence_index: i,
                token_start_index: i,
                length: 1,
                last_token_flag: true,
            });
        }

        let task = ScheduleTask::new(0, 5, Vec::new(), decode_list, 1);

        assert_eq!(task.prefill_size, 0);
        assert_eq!(task.decode_size, 5);
        assert_eq!(task.decode_list.len(), 5);
    }

    /// 测试 ScheduleTask 混合模式
    #[test]
    fn test_schedule_task_mixed() {
        let prefill_list = vec![
            vec![SequenceSlice {
                batch_index: 0,
                sequence_index: 0,
                token_start_index: 0,
                length: 10,
                last_token_flag: false,
            }],
        ];

        let mut decode_list = DecodeList::with_capacity(3);
        for i in 0..3 {
            decode_list.push(SequenceSlice {
                batch_index: i,
                sequence_index: i,
                token_start_index: i,
                length: 1,
                last_token_flag: true,
            });
        }

        let task = ScheduleTask::new(10, 3, prefill_list, decode_list, 1);

        assert_eq!(task.prefill_size, 10);
        assert_eq!(task.decode_size, 3);
    }

    /// 测试 ScheduleTask Clone 特性
    #[test]
    fn test_schedule_task_clone() {
        let task = ScheduleTask::new(10, 5, Vec::new(), DecodeList::with_capacity(0), 1);
        let cloned = task.clone();

        assert_eq!(task.prefill_size, cloned.prefill_size);
        assert_eq!(task.decode_size, cloned.decode_size);
        assert_eq!(task.task_id, cloned.task_id);
    }

    /// 测试 ScheduleTask Debug 实现
    #[test]
    fn test_schedule_task_debug() {
        let task = ScheduleTask::new(10, 5, Vec::new(), DecodeList::with_capacity(0), 1);
        let debug_str = format!("{:?}", task);

        assert!(debug_str.contains("ScheduleTask"));
        assert!(debug_str.contains("prefill_size"));
        assert!(debug_str.contains("decode_size"));
        assert!(debug_str.contains("task_id"));
    }

    /// 测试 ScheduleTask timestamp
    #[test]
    fn test_schedule_task_timestamp() {
        let before = Instant::now();
        let task = ScheduleTask::new(0, 0, Vec::new(), DecodeList::with_capacity(0), 0);
        let after = Instant::now();

        assert!(task.timestamp >= before);
        assert!(task.timestamp <= after);
    }

    /// 测试 ScheduleTask timestamp 不同任务
    #[test]
    fn test_schedule_task_timestamp_different() {
        let task1 = ScheduleTask::new(0, 0, Vec::new(), DecodeList::with_capacity(0), 1);
        std::thread::sleep(std::time::Duration::from_micros(1));
        let task2 = ScheduleTask::new(0, 0, Vec::new(), DecodeList::with_capacity(0), 2);

        assert!(task2.timestamp >= task1.timestamp);
    }

    /// 测试 ScheduleTask task_id 边界值
    #[test]
    fn test_schedule_task_task_id_boundary() {
        let task0 = ScheduleTask::new(0, 0, Vec::new(), DecodeList::with_capacity(0), 0);
        assert_eq!(task0.task_id, 0);

        let task_max = ScheduleTask::new(0, 0, Vec::new(), DecodeList::with_capacity(0), u64::MAX);
        assert_eq!(task_max.task_id, u64::MAX);
    }

    /// 测试 ScheduleTask prefill_list Arc 共享
    #[test]
    fn test_schedule_task_prefill_list_arc() {
        let prefill_list = vec![
            vec![SequenceSlice {
                batch_index: 0,
                sequence_index: 0,
                token_start_index: 0,
                length: 10,
                last_token_flag: false,
            }],
        ];

        let task = ScheduleTask::new(10, 0, prefill_list, DecodeList::with_capacity(0), 1);
        let cloned = task.clone();

        // Arc 应该共享同一个数据
        assert_eq!(Arc::strong_count(&task.prefill_list), 2);
        assert_eq!(task.prefill_list.as_ptr(), cloned.prefill_list.as_ptr());
    }

    /// 测试 ScheduleTask decode_list Arc 共享
    #[test]
    fn test_schedule_task_decode_list_arc() {
        let mut decode_list = DecodeList::with_capacity(1);
        decode_list.push(SequenceSlice::default());

        let task = ScheduleTask::new(0, 1, Vec::new(), decode_list, 1);
        let cloned = task.clone();

        // Arc 应该共享同一个数据
        assert_eq!(Arc::strong_count(&task.decode_list), 2);
    }

    /// 测试 ScheduleTask 多线程 prefill_list
    #[test]
    fn test_schedule_task_multi_thread_prefill() {
        let prefill_list: Vec<Vec<SequenceSlice>> = (0..4)
            .map(|thread_id| {
                vec![SequenceSlice {
                    batch_index: thread_id,
                    sequence_index: thread_id * 100,
                    token_start_index: thread_id * 10,
                    length: 10,
                    last_token_flag: false,
                }]
            })
            .collect();

        let task = ScheduleTask::new(40, 0, prefill_list, DecodeList::with_capacity(0), 1);

        assert_eq!(task.prefill_list.len(), 4);
        for (i, thread_list) in task.prefill_list.iter().enumerate() {
            assert_eq!(thread_list[0].batch_index, i);
        }
    }

    /// 测试 ScheduleTask 大 prefill_size
    #[test]
    fn test_schedule_task_large_prefill_size() {
        let task = ScheduleTask::new(10000, 0, Vec::new(), DecodeList::with_capacity(0), 1);
        assert_eq!(task.prefill_size, 10000);
    }

    /// 测试 ScheduleTask 大 decode_size
    #[test]
    fn test_schedule_task_large_decode_size() {
        let mut decode_list = DecodeList::with_capacity(1000);
        for i in 0..1000 {
            decode_list.push(SequenceSlice {
                batch_index: i,
                sequence_index: i,
                token_start_index: i,
                length: 1,
                last_token_flag: true,
            });
        }

        let task = ScheduleTask::new(0, 1000, Vec::new(), decode_list, 1);
        assert_eq!(task.decode_size, 1000);
        assert_eq!(task.decode_list.len(), 1000);
    }

    /// 测试 ScheduleTask prefill_list 空线程列表
    #[test]
    fn test_schedule_task_empty_thread_prefill() {
        let prefill_list = vec![Vec::new(), Vec::new(), Vec::new()];
        let task = ScheduleTask::new(0, 0, prefill_list, DecodeList::with_capacity(0), 1);

        assert_eq!(task.prefill_list.len(), 3);
        for thread_list in task.prefill_list.iter() {
            assert!(thread_list.is_empty());
        }
    }

    /// 测试 ScheduleTask decode_list 内容访问
    #[test]
    fn test_schedule_task_decode_list_access() {
        let mut decode_list = DecodeList::with_capacity(3);
        decode_list.push(SequenceSlice {
            batch_index: 0,
            sequence_index: 10,
            token_start_index: 0,
            length: 1,
            last_token_flag: true,
        });
        decode_list.push(SequenceSlice {
            batch_index: 1,
            sequence_index: 20,
            token_start_index: 1,
            length: 1,
            last_token_flag: true,
        });

        let task = ScheduleTask::new(0, 2, Vec::new(), decode_list, 1);

        assert_eq!(task.decode_list[0].batch_index, 0);
        assert_eq!(task.decode_list[0].sequence_index, 10);
        assert_eq!(task.decode_list[1].batch_index, 1);
        assert_eq!(task.decode_list[1].sequence_index, 20);
    }

    /// 测试 ScheduleTask prefill_list 内容访问
    #[test]
    fn test_schedule_task_prefill_list_access() {
        let prefill_list = vec![
            vec![
                SequenceSlice {
                    batch_index: 0,
                    sequence_index: 0,
                    token_start_index: 0,
                    length: 5,
                    last_token_flag: false,
                },
                SequenceSlice {
                    batch_index: 0,
                    sequence_index: 5,
                    token_start_index: 5,
                    length: 5,
                    last_token_flag: false,
                },
            ],
        ];

        let task = ScheduleTask::new(10, 0, prefill_list, DecodeList::with_capacity(0), 1);

        assert_eq!(task.prefill_list[0].len(), 2);
        assert_eq!(task.prefill_list[0][0].length, 5);
        assert_eq!(task.prefill_list[0][1].length, 5);
    }

    /// 测试 ScheduleTask 空数据共享
    #[test]
    fn test_schedule_task_empty_data_sharing() {
        let task1 = ScheduleTask::new(0, 0, Vec::new(), DecodeList::with_capacity(0), 1);
        let task2 = task1.clone();

        // 空数据也应该共享
        assert_eq!(task1.prefill_list.len(), task2.prefill_list.len());
        assert_eq!(task1.decode_list.len(), task2.decode_list.len());
    }

    /// 测试 ScheduleTask 时间戳顺序
    #[test]
    fn test_schedule_task_timestamp_order() {
        let mut tasks = Vec::new();
        for i in 0..10 {
            tasks.push(ScheduleTask::new(0, 0, Vec::new(), DecodeList::with_capacity(0), i));
            std::thread::sleep(std::time::Duration::from_micros(10));
        }

        // 验证时间戳递增
        for i in 1..tasks.len() {
            assert!(tasks[i].timestamp >= tasks[i - 1].timestamp);
        }
    }

    /// 测试 ScheduleTask 多次 clone
    #[test]
    fn test_schedule_task_multiple_clones() {
        let mut decode_list = DecodeList::with_capacity(1);
        decode_list.push(SequenceSlice::default());

        let task = ScheduleTask::new(0, 1, Vec::new(), decode_list, 1);

        let clones: Vec<ScheduleTask> = (0..5).map(|_| task.clone()).collect();

        // Arc 引用计数应该增加
        assert_eq!(Arc::strong_count(&task.decode_list), 6);
    }

    /// 测试 ScheduleTask 结构体大小
    #[test]
    fn test_schedule_task_size() {
        let size = std::mem::size_of::<ScheduleTask>();
        // usize: 8 bytes * 2, Arc: 8 bytes * 2, Instant: ~12-16 bytes, u64: 8 bytes
        // 总大小应该在合理范围内
        assert!(size > 0);
        assert!(size < 100);
    }

    /// 测试 ScheduleTask last_token_flag 保留
    #[test]
    fn test_schedule_task_last_token_flag() {
        let mut decode_list = DecodeList::with_capacity(2);
        decode_list.push(SequenceSlice {
            last_token_flag: true,
            ..Default::default()
        });
        decode_list.push(SequenceSlice {
            last_token_flag: false,
            ..Default::default()
        });

        let task = ScheduleTask::new(0, 2, Vec::new(), decode_list, 1);

        assert!(task.decode_list[0].last_token_flag);
        assert!(!task.decode_list[1].last_token_flag);
    }

    /// 测试 ScheduleTask 不同 task_id 唯一性
    #[test]
    fn test_schedule_task_unique_task_id() {
        let tasks: Vec<ScheduleTask> = (0..100)
            .map(|i| ScheduleTask::new(0, 0, Vec::new(), DecodeList::with_capacity(0), i))
            .collect();

        let task_ids: std::collections::HashSet<u64> =
            tasks.iter().map(|t| t.task_id).collect();

        assert_eq!(task_ids.len(), 100);
    }

    /// 测试 ScheduleTask prefill_list 和 decode_list 独立性
    #[test]
    fn test_schedule_task_prefill_decode_independence() {
        let prefill_list = vec![
            vec![SequenceSlice {
                batch_index: 0,
                sequence_index: 0,
                token_start_index: 0,
                length: 10,
                last_token_flag: false,
            }],
        ];

        let mut decode_list = DecodeList::with_capacity(1);
        decode_list.push(SequenceSlice {
            batch_index: 1,
            sequence_index: 100,
            token_start_index: 0,
            length: 1,
            last_token_flag: true,
        });

        let task = ScheduleTask::new(10, 1, prefill_list, decode_list, 1);

        // prefill 和 decode 数据应该独立
        assert_ne!(task.prefill_list[0][0].batch_index, task.decode_list[0].batch_index);
    }

    /// 测试 ScheduleTask Arc 引用释放
    #[test]
    fn test_schedule_task_arc_release() {
        let mut decode_list = DecodeList::with_capacity(1);
        decode_list.push(SequenceSlice::default());

        let task = ScheduleTask::new(0, 1, Vec::new(), decode_list, 1);
        assert_eq!(Arc::strong_count(&task.decode_list), 1);

        let cloned = task.clone();
        assert_eq!(Arc::strong_count(&task.decode_list), 2);

        drop(cloned);
        assert_eq!(Arc::strong_count(&task.decode_list), 1);
    }

    /// 测试 ScheduleTask 跨线程共享
    #[test]
    fn test_schedule_task_cross_thread_sharing() {
        use std::sync::Arc;
        use std::thread;

        let mut decode_list = DecodeList::with_capacity(1);
        decode_list.push(SequenceSlice::default());

        let task = Arc::new(ScheduleTask::new(0, 1, Vec::new(), decode_list, 1));

        let task_clone = Arc::clone(&task);
        let handle = thread::spawn(move || {
            assert_eq!(task_clone.decode_size, 1);
            task_clone
        });

        let returned = handle.join().unwrap();
        assert_eq!(returned.decode_size, 1);
    }
}
