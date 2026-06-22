use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::broadcast;

use super::core::SlotState;
use crate::runtime::executor::sync::BatchTracker;
use crate::runtime::plan::{BatchPlan, PlanBuilder};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum SchedulerState {
    Idle = 0,
    Scheduling = 1,
    Executing = 2,
    Completing = 3,
}

pub struct SharedState {
    pub batch_list: Arc<crate::operators::send_sync_ptr::SharedMut<Vec<SlotState>>>,
    pub request_count: AtomicUsize,
    pub current_batch: AtomicPtr<BatchPlan>,
    pub batch_ready: AtomicBool,
    pub scheduler_state: AtomicUsize,
    pub spin_lock: AtomicBool,
    pub batch_tracker: BatchTracker,
    pub plan_builder: Arc<PlanBuilder>,
    pub schedule_tx: broadcast::Sender<()>,
    pub task_in_flight: AtomicBool,
}

impl SharedState {
    pub fn new(
        batch_list: Arc<crate::operators::send_sync_ptr::SharedMut<Vec<SlotState>>>,
        max_decode: usize,
        max_prefill: usize,
        thread_num: usize,
        schedule_tx: broadcast::Sender<()>,
    ) -> Self {
        Self {
            batch_list,
            request_count: AtomicUsize::new(0),
            current_batch: AtomicPtr::new(std::ptr::null_mut()),
            batch_ready: AtomicBool::new(false),
            scheduler_state: AtomicUsize::new(SchedulerState::Idle as usize),
            spin_lock: AtomicBool::new(false),
            batch_tracker: BatchTracker::new(),
            plan_builder: Arc::new(PlanBuilder::new(max_decode, max_prefill, thread_num)),
            schedule_tx,
            task_in_flight: AtomicBool::new(false),
        }
    }

    pub fn push_request(&self) {
        while self
            .spin_lock
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Acquire)
            .is_err()
        {
            std::hint::spin_loop();
        }
        self.request_count.fetch_add(1, Ordering::Release);
        self.spin_lock.store(false, Ordering::Release);

        // 触发调度
        let _ = self.schedule_tx.send(());
    }

    pub fn take_requests(&self) -> usize {
        while self
            .spin_lock
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Acquire)
            .is_err()
        {
            std::hint::spin_loop();
        }
        let count = self.request_count.swap(0, Ordering::AcqRel);
        self.spin_lock.store(false, Ordering::Release);
        count
    }

    pub fn set_scheduler_state(&self, state: SchedulerState) {
        self.scheduler_state
            .store(state as usize, Ordering::Release);
    }

    pub fn get_scheduler_state(&self) -> SchedulerState {
        match self.scheduler_state.load(Ordering::Acquire) {
            0 => SchedulerState::Idle,
            1 => SchedulerState::Scheduling,
            2 => SchedulerState::Executing,
            3 => SchedulerState::Completing,
            _ => SchedulerState::Idle,
        }
    }

    pub fn publish_batch(&self, plan: Box<BatchPlan>) {
        let ptr = Box::into_raw(plan);
        let old = self.current_batch.swap(ptr, Ordering::Release);
        if !old.is_null() {
            unsafe { drop(Box::from_raw(old)) };
        }
        self.batch_ready.store(true, Ordering::Release);
    }

    pub fn take_batch(&self) -> Option<Box<BatchPlan>> {
        if !self.batch_ready.load(Ordering::Acquire) {
            return None;
        }
        let ptr = self.current_batch.load(Ordering::Acquire);
        if ptr.is_null() {
            return None;
        }
        unsafe { Some(Box::from_raw(ptr)) }
    }

    pub fn clear_batch(&self) {
        self.batch_ready.store(false, Ordering::Release);
        let ptr = self
            .current_batch
            .swap(std::ptr::null_mut(), Ordering::AcqRel);
        if !ptr.is_null() {
            unsafe { drop(Box::from_raw(ptr)) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::send_sync_ptr::SharedMut;

    /// 创建测试用的 SharedState
    fn create_shared_state() -> SharedState {
        let batch_list = Arc::new(SharedMut::new(Vec::new()));
        let (schedule_tx, _) = broadcast::channel(16);
        SharedState::new(batch_list, 32, 1024, 4, schedule_tx)
    }

    /// 测试 SchedulerState 枚举值
    #[test]
    fn test_scheduler_state_values() {
        assert_eq!(SchedulerState::Idle as u8, 0);
        assert_eq!(SchedulerState::Scheduling as u8, 1);
        assert_eq!(SchedulerState::Executing as u8, 2);
        assert_eq!(SchedulerState::Completing as u8, 3);
    }

    /// 测试 SchedulerState 枚举相等性
    #[test]
    fn test_scheduler_state_equality() {
        assert_eq!(SchedulerState::Idle, SchedulerState::Idle);
        assert_eq!(SchedulerState::Scheduling, SchedulerState::Scheduling);
        assert_eq!(SchedulerState::Executing, SchedulerState::Executing);
        assert_eq!(SchedulerState::Completing, SchedulerState::Completing);

        assert_ne!(SchedulerState::Idle, SchedulerState::Scheduling);
        assert_ne!(SchedulerState::Scheduling, SchedulerState::Executing);
        assert_ne!(SchedulerState::Executing, SchedulerState::Completing);
    }

    /// 测试 SchedulerState Copy 特性
    #[test]
    fn test_scheduler_state_copy() {
        let state = SchedulerState::Executing;
        let copied = state;
        assert_eq!(state, copied);
    }

    /// 测试 SchedulerState Clone 特性
    #[test]
    fn test_scheduler_state_clone() {
        let state = SchedulerState::Scheduling;
        let cloned = state.clone();
        assert_eq!(state, cloned);
    }

    /// 测试 SchedulerState #[repr(u8)]
    #[test]
    fn test_scheduler_state_repr() {
        assert_eq!(std::mem::size_of::<SchedulerState>(), 1);
        assert_eq!(std::mem::align_of::<SchedulerState>(), 1);
    }

    /// 测试 SharedState::new 创建
    #[test]
    fn test_shared_state_new() {
        let state = create_shared_state();
        assert_eq!(state.get_scheduler_state(), SchedulerState::Idle);
        assert_eq!(state.request_count.load(Ordering::Acquire), 0);
        assert!(!state.batch_ready.load(Ordering::Acquire));
    }

    /// 测试 SharedState::push_request
    #[test]
    fn test_shared_state_push_request() {
        let state = create_shared_state();
        assert_eq!(state.request_count.load(Ordering::Acquire), 0);

        state.push_request();
        assert_eq!(state.request_count.load(Ordering::Acquire), 1);

        state.push_request();
        state.push_request();
        assert_eq!(state.request_count.load(Ordering::Acquire), 3);
    }

    /// 测试 SharedState::take_requests
    #[test]
    fn test_shared_state_take_requests() {
        let state = create_shared_state();

        state.push_request();
        state.push_request();
        state.push_request();

        let count = state.take_requests();
        assert_eq!(count, 3);
        assert_eq!(state.request_count.load(Ordering::Acquire), 0);

        // 再次 take 应该返回 0
        let count = state.take_requests();
        assert_eq!(count, 0);
    }

    /// 测试 SharedState::set_scheduler_state
    #[test]
    fn test_shared_state_set_scheduler_state() {
        let state = create_shared_state();

        state.set_scheduler_state(SchedulerState::Scheduling);
        assert_eq!(state.get_scheduler_state(), SchedulerState::Scheduling);

        state.set_scheduler_state(SchedulerState::Executing);
        assert_eq!(state.get_scheduler_state(), SchedulerState::Executing);

        state.set_scheduler_state(SchedulerState::Completing);
        assert_eq!(state.get_scheduler_state(), SchedulerState::Completing);

        state.set_scheduler_state(SchedulerState::Idle);
        assert_eq!(state.get_scheduler_state(), SchedulerState::Idle);
    }

    /// 测试 SharedState::get_scheduler_state 无效值
    #[test]
    fn test_shared_state_get_scheduler_state_invalid() {
        let state = create_shared_state();

        // 手动设置无效值
        state.scheduler_state.store(99, Ordering::Release);

        // 应该返回 Idle 作为默认值
        assert_eq!(state.get_scheduler_state(), SchedulerState::Idle);
    }

    /// 测试 SharedState::publish_batch
    #[test]
    fn test_shared_state_publish_batch() {
        let state = create_shared_state();

        let plan = Box::new(BatchPlan::new(1));
        state.publish_batch(plan);

        assert!(state.batch_ready.load(Ordering::Acquire));

        // 清理
        state.clear_batch();
    }

    /// 测试 SharedState::take_batch
    #[test]
    fn test_shared_state_take_batch() {
        let state = create_shared_state();

        // 没有 batch 时应该返回 None
        assert!(state.take_batch().is_none());

        // 发布 batch
        let plan = Box::new(BatchPlan::new(42));
        state.publish_batch(plan);

        // 取出 batch
        let taken = state.take_batch();
        assert!(taken.is_some());
        let batch = taken.unwrap();
        assert_eq!(batch.task_id, 42);

        // 注意：take_batch 后指针仍然存在，需要手动清理
        state
            .current_batch
            .store(std::ptr::null_mut(), Ordering::Release);
        state.batch_ready.store(false, Ordering::Release);
    }

    /// 测试 SharedState::take_batch 批次未就绪
    #[test]
    fn test_shared_state_take_batch_not_ready() {
        let state = create_shared_state();

        // batch_ready 为 false
        assert!(!state.batch_ready.load(Ordering::Acquire));
        assert!(state.take_batch().is_none());
    }

    /// 测试 SharedState::clear_batch
    #[test]
    fn test_shared_state_clear_batch() {
        let state = create_shared_state();

        let plan = Box::new(BatchPlan::new(1));
        state.publish_batch(plan);

        assert!(state.batch_ready.load(Ordering::Acquire));

        state.clear_batch();

        assert!(!state.batch_ready.load(Ordering::Acquire));
        assert!(state.current_batch.load(Ordering::Acquire).is_null());
    }

    /// 测试 SharedState::clear_batch 空批次
    #[test]
    fn test_shared_state_clear_batch_empty() {
        let state = create_shared_state();

        // 没有 batch 时调用 clear_batch 应该安全
        state.clear_batch();

        assert!(!state.batch_ready.load(Ordering::Acquire));
        assert!(state.current_batch.load(Ordering::Acquire).is_null());
    }

    /// 测试 SharedState 多次 publish_batch
    #[test]
    fn test_shared_state_multiple_publish() {
        let state = create_shared_state();

        // 第一次发布
        let plan1 = Box::new(BatchPlan::new(1));
        state.publish_batch(plan1);

        // 第二次发布（应该释放旧的）
        let plan2 = Box::new(BatchPlan::new(2));
        state.publish_batch(plan2);

        assert!(state.batch_ready.load(Ordering::Acquire));

        // 清理
        state.clear_batch();
    }

    /// 测试 SharedState 并发 push_request
    #[test]
    fn test_shared_state_concurrent_push() {
        use std::sync::Arc;
        use std::thread;

        let state = Arc::new(create_shared_state());
        let mut handles = Vec::new();

        for _ in 0..10 {
            let state_clone = Arc::clone(&state);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    state_clone.push_request();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let total = state.take_requests();
        assert_eq!(total, 1000);
    }

    /// 测试 SharedState 并发 take_requests
    #[test]
    fn test_shared_state_concurrent_take() {
        use std::sync::Arc;
        use std::thread;

        let state = Arc::new(create_shared_state());

        // 先添加一些请求
        for _ in 0..1000 {
            state.push_request();
        }

        let state_clone = Arc::clone(&state);
        let handle = thread::spawn(move || {
            let count = state_clone.take_requests();
            count
        });

        let count = handle.join().unwrap();
        assert_eq!(count, 1000);

        // 再次 take 应该为 0
        let remaining = state.take_requests();
        assert_eq!(remaining, 0);
    }

    /// 测试 SharedState scheduler_state 并发修改
    #[test]
    fn test_shared_state_concurrent_scheduler_state() {
        use std::sync::Arc;
        use std::thread;

        let state = Arc::new(create_shared_state());
        let mut handles = Vec::new();

        for i in 0..4 {
            let state_clone = Arc::clone(&state);
            handles.push(thread::spawn(move || {
                let states = [
                    SchedulerState::Idle,
                    SchedulerState::Scheduling,
                    SchedulerState::Executing,
                    SchedulerState::Completing,
                ];
                state_clone.set_scheduler_state(states[i % 4]);
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // 最终状态应该是某个有效状态
        let final_state = state.get_scheduler_state();
        assert!(matches!(
            final_state,
            SchedulerState::Idle
                | SchedulerState::Scheduling
                | SchedulerState::Executing
                | SchedulerState::Completing
        ));
    }

    /// 测试 SharedState::batch_tracker 功能
    #[test]
    fn test_shared_state_batch_tracker() {
        let state = create_shared_state();

        state.batch_tracker.reset(10);
        assert_eq!(state.batch_tracker.remaining(), 10);
        assert!(!state.batch_tracker.is_complete());

        for _ in 0..10 {
            state.batch_tracker.complete_slot();
        }

        assert!(state.batch_tracker.is_complete());
    }

    /// 测试 SharedState::plan_builder 功能
    #[test]
    fn test_shared_state_plan_builder() {
        let state = create_shared_state();

        let plan = state.plan_builder.build_plan(&[]);
        assert!(plan.is_empty());
    }

    /// 测试 SharedState::batch_list 访问
    #[test]
    fn test_shared_state_batch_list() {
        let batch_list = Arc::new(SharedMut::new(Vec::new()));
        let (schedule_tx, _) = broadcast::channel(16);
        let state = SharedState::new(batch_list.clone(), 32, 1024, 4, schedule_tx);

        state.batch_list.with_mut(|list| {
            list.push(SlotState::new_start_state());
            list.push(SlotState::new_start_state());
        });

        state.batch_list.with(|list| {
            assert_eq!(list.len(), 2);
        });
    }

    /// 测试 SchedulerState 所有变体
    #[test]
    fn test_scheduler_state_all_variants() {
        let states = [
            SchedulerState::Idle,
            SchedulerState::Scheduling,
            SchedulerState::Executing,
            SchedulerState::Completing,
        ];

        for (i, state) in states.iter().enumerate() {
            assert_eq!(*state as usize, i);
        }
    }

    /// 测试 SharedState 状态转换序列
    #[test]
    fn test_shared_state_state_sequence() {
        let state = create_shared_state();

        // Idle -> Scheduling
        state.set_scheduler_state(SchedulerState::Idle);
        assert_eq!(state.get_scheduler_state(), SchedulerState::Idle);

        state.set_scheduler_state(SchedulerState::Scheduling);
        assert_eq!(state.get_scheduler_state(), SchedulerState::Scheduling);

        // Scheduling -> Executing
        state.set_scheduler_state(SchedulerState::Executing);
        assert_eq!(state.get_scheduler_state(), SchedulerState::Executing);

        // Executing -> Completing
        state.set_scheduler_state(SchedulerState::Completing);
        assert_eq!(state.get_scheduler_state(), SchedulerState::Completing);

        // Completing -> Idle
        state.set_scheduler_state(SchedulerState::Idle);
        assert_eq!(state.get_scheduler_state(), SchedulerState::Idle);
    }

    /// 测试 SharedState request_count 边界值
    #[test]
    fn test_shared_state_request_count_boundary() {
        let state = create_shared_state();

        // 大量 push
        for _ in 0..10000 {
            state.push_request();
        }

        let count = state.take_requests();
        assert_eq!(count, 10000);

        // 再次 take 应该为 0
        let count = state.take_requests();
        assert_eq!(count, 0);
    }

    /// 测试 SharedState spin_lock 功能
    #[test]
    fn test_shared_state_spin_lock() {
        let state = create_shared_state();

        // spin_lock 初始应该为 false
        assert!(!state.spin_lock.load(Ordering::Acquire));

        // push_request 和 take_requests 应该正确使用 spin_lock
        state.push_request();
        assert!(!state.spin_lock.load(Ordering::Acquire)); // 操作完成后应该释放

        state.take_requests();
        assert!(!state.spin_lock.load(Ordering::Acquire)); // 操作完成后应该释放
    }

    /// 测试 SharedState 批次发布和清理序列
    #[test]
    fn test_shared_state_batch_sequence() {
        let state = create_shared_state();

        // 发布第一个批次
        let plan1 = Box::new(BatchPlan::new(1));
        state.publish_batch(plan1);
        assert!(state.batch_ready.load(Ordering::Acquire));

        // 清理
        state.clear_batch();
        assert!(!state.batch_ready.load(Ordering::Acquire));

        // 发布第二个批次
        let plan2 = Box::new(BatchPlan::new(2));
        state.publish_batch(plan2);
        assert!(state.batch_ready.load(Ordering::Acquire));

        // 清理
        state.clear_batch();
        assert!(!state.batch_ready.load(Ordering::Acquire));
    }

    /// 测试 SharedState 创建时的默认值
    #[test]
    fn test_shared_state_defaults() {
        let batch_list = Arc::new(SharedMut::new(Vec::new()));
        let (schedule_tx, _) = broadcast::channel(16);
        let state = SharedState::new(batch_list, 32, 1024, 4, schedule_tx);

        assert_eq!(state.request_count.load(Ordering::Acquire), 0);
        assert!(state.current_batch.load(Ordering::Acquire).is_null());
        assert!(!state.batch_ready.load(Ordering::Acquire));
        assert_eq!(state.scheduler_state.load(Ordering::Acquire), 0);
        assert!(!state.spin_lock.load(Ordering::Acquire));
    }
}
