use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::broadcast;

use super::slice_scheduler::{PrefillCandidate, SliceScheduler};
use super::types::{Phase, ScheduleTask, SequenceState};
use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::scheduling::sequence_slice::{DecodeList, SequenceSlice};

pub struct Scheduler {
    prefill_list: UnsafeCell<Vec<Vec<SequenceSlice>>>,
    decode_list: UnsafeCell<DecodeList>,
    batch_list: Arc<SharedMut<Vec<SequenceState>>>,
    prefill_scheduler: UnsafeCell<SliceScheduler>,
    max_prefill_size: usize,
    max_decode_size: usize,
    thread_num: AtomicUsize,

    // Event-driven scheduling using broadcast channel
    needs_schedule: AtomicBool,
    schedule_tx: broadcast::Sender<()>,
    timeout: Duration,
    broadcast_sender: broadcast::Sender<ScheduleTask>,
    next_task_id: AtomicU64,
    task_in_flight: Arc<AtomicBool>,
}

unsafe impl Sync for Scheduler {}

unsafe impl Send for Scheduler {}

enum BatchPlan {
    Decode(Vec<(usize, usize)>),
    Prefill {
        candidates: Vec<PrefillCandidate>,
        total_tokens: usize,
    },
    Idle,
}

impl Scheduler {
    pub fn new(
        sequence_length: usize,
        batch_size: usize,
        thread_num: usize,
        _threshold: usize, // Keep for compatibility
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
        )
    }

    pub fn with_mode(
        sequence_length: usize,
        batch_size: usize,
        chunk_size: usize,
        thread_num: usize,
        _threshold: usize, // Keep for compatibility
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
    ) -> Self {
        let (schedule_tx, _) = broadcast::channel(16);
        Self {
            max_decode_size: batch_size,
            max_prefill_size: chunk_size,
            batch_list,
            thread_num: AtomicUsize::new(thread_num),
            prefill_scheduler: UnsafeCell::new(SliceScheduler::new(batch_size * thread_num)),
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
        let prefill_scheduler = unsafe { &mut *self.prefill_scheduler.get() };
        prefill_scheduler.set_task_count(thread_num);
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

        let prefill_scheduler = unsafe { &mut *self.prefill_scheduler.get() };
        prefill_scheduler.set_task_count(prefill_task_count);

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

        // Set flag and signal the scheduler
        self.needs_schedule.store(true, Ordering::Release);
        let _ = self.schedule_tx.send(());
        true
    }

    pub async fn run(self: Arc<Self>) {
        let mut interval = tokio::time::interval(self.timeout);
        let mut schedule_rx = self.schedule_tx.subscribe();

        loop {
            tokio::select! {
                // Event-driven: wake up when there's a schedule request
                _ = schedule_rx.recv() => {
                    if self.needs_schedule.load(Ordering::Acquire) {
                        self.trigger_schedule();
                    }
                }
                // Fallback: periodic check in case events were missed
                _ = interval.tick() => {
                    if self.needs_schedule.load(Ordering::Acquire) {
                        self.trigger_schedule();
                        continue;
                    }

                    // Also check batch state as fallback
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
        let max_decode_size = self.max_decode_size;
        self.batch_list.with(|batch_list| {
            let mut decode_candidates = Vec::with_capacity(max_decode_size);
            let mut total_tokens = 0usize;
            let mut candidates = Vec::with_capacity(batch_list.len());
            let mut has_decode = false;

            for (batch_index, record) in batch_list.iter().enumerate() {
                match record.phase {
                    Phase::Decode => {
                        has_decode = true;
                        if decode_candidates.len() < max_decode_size {
                            decode_candidates.push((batch_index, record.sequence_index));
                        }
                    }
                    Phase::Prefill => {
                        total_tokens += record.filling_length;
                        candidates.push(PrefillCandidate {
                            batch_index,
                            sequence_index: record.sequence_index,
                            remaining: record.filling_length,
                        });
                    }
                    _ => {}
                }
            }

            if !candidates.is_empty() {
                BatchPlan::Prefill {
                    candidates,
                    total_tokens: total_tokens.min(self.max_prefill_size),
                }
            } else if has_decode {
                BatchPlan::Decode(decode_candidates)
            } else {
                BatchPlan::Idle
            }
        })
    }

    fn schedule_decode_round(&self, decode_candidates: Vec<(usize, usize)>) -> usize {
        self.clear_round_outputs();
        let decode_count = decode_candidates.len();
        let decode_list = unsafe { &mut *self.decode_list.get() };

        for (idx, (batch_index, sequence_index)) in decode_candidates.into_iter().enumerate() {
            decode_list.push(SequenceSlice {
                batch_index,
                sequence_index,
                token_start_index: idx,
                length: 1,
                last_token_flag: true,
            });
        }

        decode_count
    }

    fn schedule_prefill_round(
        &self,
        candidates: Vec<PrefillCandidate>,
        total_tokens: usize,
    ) -> usize {
        self.clear_round_outputs();
        let mut prefill_count = 0usize;
        let prefill_scheduler = unsafe { &mut *self.prefill_scheduler.get() };
        let decode_list = unsafe { &mut *self.decode_list.get() };
        let prefill_list = unsafe { &mut *self.prefill_list.get() };

        prefill_scheduler.init(total_tokens);

        for candidate in candidates {
            if prefill_scheduler.is_done() {
                break;
            }

            let attention_length = candidate
                .remaining
                .min(prefill_scheduler.remaining_tokens());
            if attention_length > 0 {
                decode_list.push(SequenceSlice {
                    batch_index: candidate.batch_index,
                    sequence_index: candidate.sequence_index,
                    token_start_index: prefill_count,
                    length: attention_length,
                    last_token_flag: attention_length == candidate.remaining,
                });
            }

            prefill_scheduler.schedule_for_sequence(
                candidate.batch_index,
                candidate.sequence_index,
                candidate.remaining,
                0,
                prefill_list,
                &mut prefill_count,
            );
        }

        prefill_count
    }

    fn trigger_schedule(&self) {
        // Try to take the schedule flag
        if !self.needs_schedule.swap(false, Ordering::AcqRel) {
            return;
        }

        // Try to claim the scheduling slot
        if self
            .task_in_flight
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            // Another task is already scheduling, restore the flag
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
            // Task sent successfully
        } else {
            self.task_in_flight.store(false, Ordering::Release);
            self.needs_schedule.store(true, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut scheduler =
            Scheduler::new(16, 4, 3, 1, Duration::from_millis(100), sender, batch_list);
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
        let mut scheduler =
            Scheduler::new(16, 4, 6, 1, Duration::from_millis(100), sender, batch_list);

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
        let mut scheduler =
            Scheduler::new(8, 4, 3, 1, Duration::from_millis(100), sender, batch_list);
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
        let mut scheduler =
            Scheduler::new(5, 2, 2, 1, Duration::from_millis(100), sender, batch_list);
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
        let mut scheduler =
            Scheduler::new(16, 4, 3, 1, Duration::from_millis(100), sender, batch_list);
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
}
