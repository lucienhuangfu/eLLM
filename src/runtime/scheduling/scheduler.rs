use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::broadcast;

use super::sequence_slice::{DecodeList, SequenceSlice};
use super::strategy::{BatchPlan, DefaultSchedulerStrategy, PrefillCandidate, SchedulerStrategy};
use super::types::{Phase, ScheduleTask, SequenceState};
use crate::operators::send_sync_ptr::SharedMut;

pub struct Scheduler {
    prefill_list: UnsafeCell<Vec<Vec<SequenceSlice>>>,
    decode_list: UnsafeCell<DecodeList>,
    batch_list: Arc<SharedMut<Vec<SequenceState>>>,
    strategy: Box<dyn SchedulerStrategy>,
    max_prefill_size: usize,
    max_decode_size: usize,
    thread_num: AtomicUsize,

    needs_schedule: AtomicBool,
    schedule_tx: broadcast::Sender<()>,
    timeout: Duration,
    broadcast_sender: broadcast::Sender<ScheduleTask>,
    next_task_id: AtomicU64,
    task_in_flight: Arc<AtomicBool>,
}

unsafe impl Sync for Scheduler {}

unsafe impl Send for Scheduler {}

impl Scheduler {
    pub fn new(
        sequence_length: usize,
        batch_size: usize,
        thread_num: usize,
        _threshold: usize,
        timeout: Duration,
        broadcast_sender: broadcast::Sender<ScheduleTask>,
        batch_list: Arc<SharedMut<Vec<SequenceState>>>,
    ) -> Self {
        Self::build(
            sequence_length,
            batch_size,
            sequence_length * batch_size,
            thread_num,
            timeout,
            broadcast_sender,
            batch_list,
            Box::new(DefaultSchedulerStrategy::new(
                batch_size,
                sequence_length * batch_size,
            )),
        )
    }

    pub fn with_mode(
        sequence_length: usize,
        batch_size: usize,
        chunk_size: usize,
        thread_num: usize,
        _threshold: usize,
        timeout: Duration,
        broadcast_sender: broadcast::Sender<ScheduleTask>,
        batch_list: Arc<SharedMut<Vec<SequenceState>>>,
    ) -> Self {
        Self::build(
            sequence_length,
            batch_size,
            chunk_size,
            thread_num,
            timeout,
            broadcast_sender,
            batch_list,
            Box::new(DefaultSchedulerStrategy::new(batch_size, chunk_size)),
        )
    }

    pub fn with_strategy(
        sequence_length: usize,
        batch_size: usize,
        chunk_size: usize,
        thread_num: usize,
        timeout: Duration,
        broadcast_sender: broadcast::Sender<ScheduleTask>,
        batch_list: Arc<SharedMut<Vec<SequenceState>>>,
        strategy: Box<dyn SchedulerStrategy>,
    ) -> Self {
        Self::build(
            sequence_length,
            batch_size,
            chunk_size,
            thread_num,
            timeout,
            broadcast_sender,
            batch_list,
            strategy,
        )
    }

    fn build(
        _sequence_length: usize,
        batch_size: usize,
        chunk_size: usize,
        thread_num: usize,
        timeout: Duration,
        broadcast_sender: broadcast::Sender<ScheduleTask>,
        batch_list: Arc<SharedMut<Vec<SequenceState>>>,
        strategy: Box<dyn SchedulerStrategy>,
    ) -> Self {
        let (schedule_tx, _) = broadcast::channel(16);
        Self {
            max_decode_size: batch_size,
            max_prefill_size: chunk_size,
            batch_list,
            thread_num: AtomicUsize::new(thread_num),
            strategy,
            prefill_list: UnsafeCell::new(
                (0..thread_num)
                    .map(|_| Vec::with_capacity(batch_size))
                    .collect(),
            ),
            decode_list: UnsafeCell::new(DecodeList::with_capacity(batch_size)),
            needs_schedule: AtomicBool::new(false),
            schedule_tx,
            timeout,
            broadcast_sender,
            next_task_id: AtomicU64::new(1),
            task_in_flight: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn thread_num(&self) -> usize {
        self.thread_num.load(Ordering::Acquire)
    }

    pub fn set_thread_num(&self, thread_num: usize) {
        let thread_num = thread_num.max(1);
        self.thread_num.store(thread_num, Ordering::Release);
        let prefill_list = unsafe { &mut *self.prefill_list.get() };
        if prefill_list.len() > thread_num {
            prefill_list.truncate(thread_num);
        } else {
            prefill_list.resize_with(thread_num, || Vec::with_capacity(self.max_decode_size));
        }
    }

    pub fn batch_list(&self) -> Arc<SharedMut<Vec<SequenceState>>> {
        Arc::clone(&self.batch_list)
    }

    pub fn task_in_flight(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.task_in_flight)
    }

    pub fn prefill_list(&self) -> Vec<Vec<SequenceSlice>> {
        unsafe { (*self.prefill_list.get()).clone() }
    }

    pub fn decode_list(&self) -> DecodeList {
        unsafe { (*self.decode_list.get()).clone() }
    }

    pub fn schedule_batch(&self) -> (usize, usize) {
        let thread_num = self.thread_num.load(Ordering::Acquire);
        let prefill_list = unsafe { &*self.prefill_list.get() };
        let prefill_task_count = thread_num.min(prefill_list.len());

        if prefill_task_count == 0 {
            let plan = self.plan_next_round();
            match plan {
                BatchPlan::Decode(decode_candidates) => {
                    let decode_count = self.schedule_decode_round(decode_candidates);
                    return (0, decode_count);
                }
                BatchPlan::Idle => return (0, 0),
                BatchPlan::Prefill { .. } => unreachable!(),
            }
        }

        match self.plan_next_round() {
            BatchPlan::Decode(decode_candidates) => {
                let decode_count = self.schedule_decode_round(decode_candidates);
                (0, decode_count)
            }
            BatchPlan::Prefill {
                candidates,
                total_tokens,
            } => {
                let prefill_count = self.schedule_prefill_round(candidates, total_tokens);
                let decode_list = unsafe { &*self.decode_list.get() };
                (prefill_count, decode_list.len())
            }
            BatchPlan::Idle => {
                self.clear_round_outputs();
                (0, 0)
            }
        }
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
                    if self.needs_schedule.load(Ordering::Acquire) {
                        self.trigger_schedule();
                    }
                }
                _ = interval.tick() => {
                    if self.needs_schedule.load(Ordering::Acquire) {
                        self.trigger_schedule();
                        continue;
                    }

                    let has_work = self.batch_list.with(|batch_list| {
                        batch_list
                            .iter()
                            .any(|r| r.phase == Phase::Decode || r.phase == Phase::Prefill)
                    });
                    if has_work {
                        self.needs_schedule.store(true, Ordering::Release);
                        self.trigger_schedule();
                    }
                }
            }
        }
    }

    fn clear_round_outputs(&self) {
        let prefill_list = unsafe { &mut *self.prefill_list.get() };
        prefill_list.iter_mut().for_each(Vec::clear);
        let decode_list = unsafe { &mut *self.decode_list.get() };
        decode_list.clear();
    }

    fn plan_next_round(&self) -> BatchPlan {
        self.batch_list.with(|batch_list| {
            self.strategy
                .plan_next_round(batch_list, self.max_decode_size, self.max_prefill_size)
        })
    }

    fn schedule_decode_round(&self, decode_candidates: Vec<(usize, usize)>) -> usize {
        self.clear_round_outputs();
        let decode_list = unsafe { &mut *self.decode_list.get() };
        self.strategy
            .schedule_decode_round(decode_candidates, decode_list)
    }

    fn schedule_prefill_round(
        &self,
        candidates: Vec<PrefillCandidate>,
        total_tokens: usize,
    ) -> usize {
        self.clear_round_outputs();
        let prefill_list = unsafe { &mut *self.prefill_list.get() };
        let decode_list = unsafe { &mut *self.decode_list.get() };
        let thread_num = self.thread_num.load(Ordering::Acquire);

        self.strategy.schedule_prefill_round(
            candidates,
            total_tokens,
            prefill_list,
            decode_list,
            thread_num,
        )
    }

    fn trigger_schedule(&self) {
        if !self.needs_schedule.swap(false, Ordering::AcqRel) {
            return;
        }

        if self
            .task_in_flight
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            self.needs_schedule.store(true, Ordering::Release);
            return;
        }

        let (prefill_size, decode_size) = self.schedule_batch();
        if prefill_size == 0 && decode_size == 0 {
            self.task_in_flight.store(false, Ordering::Release);
            self.needs_schedule.store(true, Ordering::Release);
            return;
        }

        let task = ScheduleTask::new(
            prefill_size,
            decode_size,
            unsafe { (*self.prefill_list.get()).clone() },
            unsafe { (*self.decode_list.get()).clone() },
            self.next_task_id.fetch_add(1, Ordering::Relaxed),
        );

        if self.broadcast_sender.send(task).is_ok() {
        } else {
            self.task_in_flight.store(false, Ordering::Release);
            self.needs_schedule.store(true, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::scheduling::strategy::DefaultSchedulerStrategy;

    fn decode_state(sequence_index: usize, kv_index: usize) -> SequenceState {
        SequenceState::new_decode_state(sequence_index, kv_index)
    }

    fn prefill_state(sequence_index: usize, filling_length: usize) -> SequenceState {
        SequenceState::new_prefill_state(sequence_index, filling_length)
    }

    #[test]
    fn plan_next_round_returns_idle_for_empty_batch() {
        let (sender, _) = broadcast::channel(16);
        let batch_list = Arc::new(SharedMut::new(Vec::new()));
        let scheduler = Scheduler::new(16, 4, 3, 1, Duration::from_millis(100), sender, batch_list);

        match scheduler.plan_next_round() {
            BatchPlan::Idle => {}
            _ => panic!("expected idle plan for an empty batch"),
        }
    }

    #[test]
    fn schedule_decode_round_uses_one_decode_sequence() {
        let (sender, _) = broadcast::channel(16);
        let batch_list = Arc::new(SharedMut::new(Vec::new()));
        let scheduler = Scheduler::new(16, 4, 3, 1, Duration::from_millis(100), sender, batch_list);
        scheduler.batch_list.with_mut(|batch_list| {
            batch_list.push(decode_state(100, 128));
        });

        let (prefill, decode_tokens) = scheduler.schedule_batch();

        assert_eq!(prefill, 0);
        assert_eq!(decode_tokens, 1);

        assert!(scheduler.prefill_list().iter().all(Vec::is_empty));
        assert_eq!(scheduler.decode_list().len(), 1);

        let slice = &scheduler.decode_list()[0];
        assert_eq!(slice.batch_index, 0);
        assert_eq!(slice.sequence_index, 100);
        assert_eq!(slice.token_start_index, 0);
        assert_eq!(slice.length, 1);
        assert!(slice.last_token_flag);
    }

    #[test]
    fn set_thread_num_resizes_prefill_work_lists() {
        let (sender, _) = broadcast::channel(16);
        let batch_list = Arc::new(SharedMut::new(Vec::new()));
        let scheduler = Scheduler::new(16, 4, 6, 1, Duration::from_millis(100), sender, batch_list);

        scheduler.set_thread_num(3);

        assert_eq!(scheduler.thread_num(), 3);
        assert_eq!(scheduler.prefill_list().len(), 3);

        scheduler.set_thread_num(5);

        assert_eq!(scheduler.thread_num(), 5);
        assert_eq!(scheduler.prefill_list().len(), 5);
    }

    #[test]
    fn schedule_prefill_round_limits_one_sequence_to_max_prefill_size() {
        let (sender, _) = broadcast::channel(16);
        let batch_list = Arc::new(SharedMut::new(Vec::new()));
        let scheduler = Scheduler::new(8, 4, 3, 1, Duration::from_millis(100), sender, batch_list);
        scheduler.batch_list.with_mut(|batch_list| {
            batch_list.push(prefill_state(0, 23));
        });

        let (prefill_tokens, decode_slices) = scheduler.schedule_batch();

        assert_eq!(prefill_tokens, 23.min(8 * 4));
        assert_eq!(decode_slices, 1);
        assert_eq!(scheduler.decode_list().len(), 1);

        let attention_slice = &scheduler.decode_list()[0];
        assert_eq!(attention_slice.batch_index, 0);
        assert_eq!(attention_slice.sequence_index, 0);
        assert_eq!(attention_slice.token_start_index, 0);
        assert_eq!(attention_slice.length, 23);
        assert!(attention_slice.last_token_flag);

        assert_eq!(scheduler.prefill_list().len(), 3);
        assert_eq!(scheduler.prefill_list()[0].len(), 1);
        assert_eq!(scheduler.prefill_list()[1].len(), 1);
        assert_eq!(scheduler.prefill_list()[2].len(), 1);

        let t0 = &scheduler.prefill_list()[0][0];
        assert_eq!(t0.batch_index, 0);
        assert_eq!(t0.sequence_index, 0);
        assert_eq!(t0.token_start_index, 0);
        assert_eq!(t0.length, 8);

        let t1 = &scheduler.prefill_list()[1][0];
        assert_eq!(t1.batch_index, 0);
        assert_eq!(t1.sequence_index, 8);
        assert_eq!(t1.token_start_index, 8);
        assert_eq!(t1.length, 8);

        let t2 = &scheduler.prefill_list()[2][0];
        assert_eq!(t2.batch_index, 0);
        assert_eq!(t2.sequence_index, 16);
        assert_eq!(t2.token_start_index, 16);
        assert_eq!(t2.length, 7);
    }

    #[test]
    fn schedule_prefill_round_truncates_to_max_prefill_size() {
        let (sender, _) = broadcast::channel(16);
        let batch_list = Arc::new(SharedMut::new(Vec::new()));
        let scheduler = Scheduler::new(5, 2, 2, 1, Duration::from_millis(100), sender, batch_list);
        scheduler.batch_list.with_mut(|batch_list| {
            batch_list.push(prefill_state(0, 13));
        });

        let (prefill_tokens, decode_slices) = scheduler.schedule_batch();

        assert_eq!(prefill_tokens, 10);
        assert_eq!(decode_slices, 1);
        assert_eq!(scheduler.decode_list().len(), 1);

        let attention_slice = &scheduler.decode_list()[0];
        assert_eq!(attention_slice.batch_index, 0);
        assert_eq!(attention_slice.sequence_index, 0);
        assert_eq!(attention_slice.token_start_index, 0);
        assert_eq!(attention_slice.length, 10);
        assert!(!attention_slice.last_token_flag);

        assert_eq!(scheduler.prefill_list().len(), 2);
        assert_eq!(scheduler.prefill_list()[0].len(), 1);
        assert_eq!(scheduler.prefill_list()[1].len(), 1);

        let first = &scheduler.prefill_list()[0][0];
        assert_eq!(first.batch_index, 0);
        assert_eq!(first.sequence_index, 0);
        assert_eq!(first.token_start_index, 0);
        assert_eq!(first.length, 5);
        assert!(!first.last_token_flag);

        let second = &scheduler.prefill_list()[1][0];
        assert_eq!(second.batch_index, 0);
        assert_eq!(second.sequence_index, 5);
        assert_eq!(second.token_start_index, 5);
        assert_eq!(second.length, 5);
        assert!(!second.last_token_flag);
    }

    #[test]
    fn schedule_batch_finishes_prefill_when_both_phases_exist() {
        let (sender, _) = broadcast::channel(16);
        let batch_list = Arc::new(SharedMut::new(Vec::new()));
        let scheduler = Scheduler::new(16, 4, 3, 1, Duration::from_millis(100), sender, batch_list);
        scheduler.batch_list.with_mut(|batch_list| {
            batch_list.push(prefill_state(0, 6));
            batch_list.push(decode_state(100, 128));
            batch_list.push(prefill_state(32, 3));
        });

        let (prefill_tokens, decode_tokens) = scheduler.schedule_batch();

        assert_eq!(prefill_tokens, 9);
        assert_eq!(decode_tokens, 2);
        assert_eq!(scheduler.decode_list().len(), 2);

        let first = &scheduler.decode_list()[0];
        assert_eq!(first.batch_index, 0);
        assert_eq!(first.sequence_index, 0);
        assert_eq!(first.token_start_index, 0);
        assert_eq!(first.length, 6);
        assert!(first.last_token_flag);

        let second = &scheduler.decode_list()[1];
        assert_eq!(second.batch_index, 2);
        assert_eq!(second.sequence_index, 32);
        assert_eq!(second.token_start_index, 6);
        assert_eq!(second.length, 3);
        assert!(second.last_token_flag);
    }

    #[test]
    fn sequence_state_transitions() {
        let mut state = SequenceState::new_prefill_state(0, 10);
        assert_eq!(state.phase, Phase::Prefill);
        assert_eq!(state.filling_length, 10);

        state.advance_sequence(5);
        assert_eq!(state.sequence_index, 5);
        assert_eq!(state.filling_length, 5);
        assert_eq!(state.phase, Phase::Prefill);

        state.advance_sequence(5);
        assert_eq!(state.sequence_index, 10);
        assert_eq!(state.filling_length, 0);
        assert_eq!(state.phase, Phase::Decode);

        state.transition_to_eos();
        assert_eq!(state.phase, Phase::Eos);

        state.reset_to_start();
        assert_eq!(state.phase, Phase::Start);
        assert_eq!(state.sequence_index, usize::MAX);
    }

    #[test]
    fn custom_strategy_can_be_used() {
        let (sender, _) = broadcast::channel(16);
        let batch_list = Arc::new(SharedMut::new(Vec::new()));

        let strategy = Box::new(DefaultSchedulerStrategy::new(4, 32));
        let scheduler = Scheduler::with_strategy(
            16,
            4,
            32,
            2,
            Duration::from_millis(100),
            sender,
            batch_list,
            strategy,
        );

        assert_eq!(scheduler.thread_num(), 2);
    }
}
