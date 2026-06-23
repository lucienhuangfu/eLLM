use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::broadcast;

use super::strategy::{DefaultSchedulerStrategy, SchedulerStrategy};
use super::task::ScheduleTask;
use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::plan::BatchPlan;
use crate::runtime::session::SlotManager;
use crate::runtime::state::core::SlotState;
use crate::runtime::state::shared::SharedState;

pub struct Scheduler {
    batch_list: Arc<SharedMut<Vec<SlotState>>>,
    slot_manager: Arc<SlotManager<f16>>,
    strategy: Box<dyn SchedulerStrategy>,
    thread_num: AtomicUsize,

    needs_schedule: AtomicBool,
    schedule_tx: broadcast::Sender<()>,
    timeout: Duration,
    broadcast_sender: broadcast::Sender<ScheduleTask>,
    shared_state: Arc<SharedState>,
}

impl Scheduler {
    pub fn new(
        _sequence_length: usize,
        batch_size: usize,
        thread_num: usize,
        _threshold: usize,
        timeout: Duration,
        broadcast_sender: broadcast::Sender<ScheduleTask>,
        batch_list: Arc<SharedMut<Vec<SlotState>>>,
        slot_manager: Arc<SlotManager<f16>>,
    ) -> Self {
        Self::build(
            batch_size,
            _sequence_length * batch_size,
            thread_num,
            timeout,
            broadcast_sender,
            batch_list,
            slot_manager,
            Box::new(DefaultSchedulerStrategy::new(
                batch_size,
                _sequence_length * batch_size,
                thread_num,
            )),
        )
    }

    pub fn with_mode(
        _sequence_length: usize,
        batch_size: usize,
        chunk_size: usize,
        thread_num: usize,
        _threshold: usize,
        timeout: Duration,
        broadcast_sender: broadcast::Sender<ScheduleTask>,
        batch_list: Arc<SharedMut<Vec<SlotState>>>,
        slot_manager: Arc<SlotManager<f16>>,
    ) -> Self {
        Self::build(
            batch_size,
            chunk_size,
            thread_num,
            timeout,
            broadcast_sender,
            batch_list,
            slot_manager,
            Box::new(DefaultSchedulerStrategy::new(
                batch_size, chunk_size, thread_num,
            )),
        )
    }

    pub fn with_strategy(
        _sequence_length: usize,
        batch_size: usize,
        chunk_size: usize,
        thread_num: usize,
        timeout: Duration,
        broadcast_sender: broadcast::Sender<ScheduleTask>,
        batch_list: Arc<SharedMut<Vec<SlotState>>>,
        slot_manager: Arc<SlotManager<f16>>,
        strategy: Box<dyn SchedulerStrategy>,
    ) -> Self {
        Self::build(
            batch_size,
            chunk_size,
            thread_num,
            timeout,
            broadcast_sender,
            batch_list,
            slot_manager,
            strategy,
        )
    }

    pub fn with_shared_state(
        _sequence_length: usize,
        batch_size: usize,
        chunk_size: usize,
        thread_num: usize,
        _threshold: usize,
        timeout: Duration,
        broadcast_sender: broadcast::Sender<ScheduleTask>,
        batch_list: Arc<SharedMut<Vec<SlotState>>>,
        slot_manager: Arc<SlotManager<f16>>,
        shared_state: Arc<SharedState>,
    ) -> Self {
        let (schedule_tx, _) = broadcast::channel(16);
        Self {
            batch_list,
            slot_manager,
            thread_num: AtomicUsize::new(thread_num),
            strategy: Box::new(DefaultSchedulerStrategy::new(
                batch_size, chunk_size, thread_num,
            )),
            needs_schedule: AtomicBool::new(false),
            schedule_tx,
            timeout,
            broadcast_sender,
            shared_state,
        }
    }

    fn build(
        batch_size: usize,
        chunk_size: usize,
        thread_num: usize,
        timeout: Duration,
        broadcast_sender: broadcast::Sender<ScheduleTask>,
        batch_list: Arc<SharedMut<Vec<SlotState>>>,
        slot_manager: Arc<SlotManager<f16>>,
        strategy: Box<dyn SchedulerStrategy>,
    ) -> Self {
        let (schedule_tx, _) = broadcast::channel(16);
        let shared_state = Arc::new(SharedState::new(Arc::clone(&batch_list)));
        Self {
            batch_list,
            slot_manager,
            thread_num: AtomicUsize::new(thread_num),
            strategy,
            needs_schedule: AtomicBool::new(false),
            schedule_tx,
            timeout,
            broadcast_sender,
            shared_state,
        }
    }

    pub fn thread_num(&self) -> usize {
        self.thread_num.load(Ordering::Acquire)
    }

    pub fn set_thread_num(&self, thread_num: usize) {
        self.thread_num.store(thread_num.max(1), Ordering::Release);
    }

    pub fn batch_list(&self) -> Arc<SharedMut<Vec<SlotState>>> {
        Arc::clone(&self.batch_list)
    }

    pub fn shared_state(&self) -> Arc<SharedState> {
        Arc::clone(&self.shared_state)
    }

    pub fn schedule_batch(&self) -> Option<BatchPlan> {
        self.batch_list.with(|batch_list| {
            let plan = self.strategy.plan_next_round(
                batch_list,
                self.thread_num.load(Ordering::Acquire),
                0,
            );
            if plan.is_empty() {
                None
            } else {
                Some(plan)
            }
        })
    }

    pub fn reset(&self) {
        self.needs_schedule.store(false, Ordering::Release);
    }

    pub async fn notify_tokens(&self, count: usize) -> bool {
        if count == 0 {
            return false;
        }

        self.needs_schedule.store(true, Ordering::Release);
        let _ = self.schedule_tx.send(());
        true
    }

    pub async fn run(self: Arc<Self>) {
        let mut interval = tokio::time::interval(self.timeout);
        let mut schedule_rx = self.schedule_tx.subscribe();

        loop {
            tokio::select! {
                _ = schedule_rx.recv() => {
                    // 接收到信号时，总是触发调度检查
                    self.trigger_schedule();
                }
                _ = interval.tick() => {
                    if self.needs_schedule.load(Ordering::Acquire) {
                        self.trigger_schedule();
                        continue;
                    }

                    let has_work = self.slot_manager.has_work().await;
                    if has_work {
                        self.needs_schedule.store(true, Ordering::Release);
                        self.trigger_schedule();
                    }
                }
            }
        }
    }

    fn trigger_schedule(&self) {
        println!("[Scheduler] trigger_schedule 被调用");

        if self
            .shared_state
            .task_in_flight
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            // 已经有任务在执行，重置 needs_schedule
            println!("[Scheduler] 任务已在执行中，重置 needs_schedule");
            self.needs_schedule.store(false, Ordering::Release);
            return;
        }

        println!("[Scheduler] 开始调度...");
        let plan = match self.schedule_batch() {
            Some(p) => {
                println!(
                    "[Scheduler] 调度计划: prefill_size={}, decode_size={}",
                    p.prefill_size, p.decode_size
                );
                p
            }
            None => {
                println!("[Scheduler] 没有待处理的任务");
                self.shared_state
                    .task_in_flight
                    .store(false, Ordering::Release);
                self.needs_schedule.store(false, Ordering::Release);
                return;
            }
        };

        let task = ScheduleTask::new(
            plan.prefill_size,
            plan.decode_size,
            plan.prefill_list,
            plan.decode_list,
            plan.task_id,
        );

        println!("[Scheduler] 发送任务到执行器: task_id={}", task.task_id);
        if self.broadcast_sender.send(task).is_err() {
            println!("[Scheduler] 发送任务失败");
            self.shared_state
                .task_in_flight
                .store(false, Ordering::Release);
            self.needs_schedule.store(true, Ordering::Release);
        } else {
            println!("[Scheduler] 发送任务成功");
            self.needs_schedule.store(false, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::scheduler::strategy::DefaultSchedulerStrategy;
    use crate::runtime::session::{SessionMode, SlotManager};
    use crate::runtime::state::core::SlotState;

    fn decode_state(sequence_index: usize, kv_index: usize) -> SlotState {
        SlotState::new_decode_state(sequence_index, kv_index)
    }

    fn prefill_state(sequence_index: usize, filling_length: usize) -> SlotState {
        SlotState::new_prefill_state(sequence_index, filling_length)
    }

    fn create_slot_manager(batch_size: usize) -> Arc<SlotManager<f16>> {
        use crate::runtime::state::batch::BatchSequence;
        let batch_sequences = Arc::new(crate::operators::send_sync_ptr::SharedMut::new(
            BatchSequence::<f16>::new(
                std::ptr::null_mut(),
                batch_size,
                1024,
                "gpt2",
                "gpt2",
                "gpt2",
            )
            .unwrap(),
        ));
        Arc::new(SlotManager::new(
            batch_size,
            batch_sequences,
            SessionMode::Reusable,
            600000, // 10 minutes default for tests
        ))
    }

    #[test]
    fn schedule_batch_returns_none_for_empty_batch() {
        let (sender, _) = broadcast::channel(16);
        let batch_list = Arc::new(SharedMut::new(Vec::new()));
        let slot_manager = create_slot_manager(4);
        let scheduler = Scheduler::new(
            16,
            4,
            3,
            1,
            Duration::from_millis(100),
            sender,
            batch_list,
            slot_manager,
        );

        assert!(scheduler.schedule_batch().is_none());
    }

    #[test]
    fn schedule_batch_returns_plan_for_decode() {
        let (sender, _) = broadcast::channel(16);
        let batch_list = Arc::new(SharedMut::new(Vec::new()));
        let slot_manager = create_slot_manager(4);
        let scheduler = Scheduler::new(
            16,
            4,
            3,
            1,
            Duration::from_millis(100),
            sender,
            batch_list,
            slot_manager,
        );
        scheduler.batch_list.with_mut(|batch_list| {
            batch_list.push(decode_state(100, 128));
        });

        let plan = scheduler.schedule_batch().unwrap();
        assert_eq!(plan.prefill_size, 0);
        assert_eq!(plan.decode_size, 1);
    }

    #[test]
    fn set_thread_num_updates_thread_count() {
        let (sender, _) = broadcast::channel(16);
        let batch_list = Arc::new(SharedMut::new(Vec::new()));
        let slot_manager = create_slot_manager(4);
        let scheduler = Scheduler::new(
            16,
            4,
            6,
            1,
            Duration::from_millis(100),
            sender,
            batch_list,
            slot_manager,
        );

        scheduler.set_thread_num(3);
        assert_eq!(scheduler.thread_num(), 3);

        scheduler.set_thread_num(5);
        assert_eq!(scheduler.thread_num(), 5);
    }

    #[test]
    fn custom_strategy_can_be_used() {
        let (sender, _) = broadcast::channel(16);
        let batch_list = Arc::new(SharedMut::new(Vec::new()));
        let slot_manager = create_slot_manager(4);

        let strategy = Box::new(DefaultSchedulerStrategy::new(4, 32, 2));
        let scheduler = Scheduler::with_strategy(
            16,
            4,
            32,
            2,
            Duration::from_millis(100),
            sender,
            batch_list,
            slot_manager,
            strategy,
        );

        assert_eq!(scheduler.thread_num(), 2);
    }
}
