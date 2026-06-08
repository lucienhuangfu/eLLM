use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::{broadcast, Mutex as AsyncMutex};

use super::scheduler::BatchScheduler;
use super::types::ScheduleTask;

pub struct TokenCounter {
    current_tokens: AtomicUsize,
    threshold: usize,
    timeout: Duration,
    scheduler: Arc<AsyncMutex<BatchScheduler>>,
    broadcast_sender: broadcast::Sender<ScheduleTask>,
    schedule_gate: AsyncMutex<()>,
    last_schedule_time: Mutex<Instant>,
    next_task_id: AtomicU64,
    task_in_flight: Arc<AtomicBool>,
}

impl TokenCounter {
    pub fn new(
        threshold: usize,
        timeout: Duration,
        scheduler: Arc<AsyncMutex<BatchScheduler>>,
        broadcast_sender: broadcast::Sender<ScheduleTask>,
    ) -> Self {
        Self {
            current_tokens: AtomicUsize::new(0),
            threshold: threshold.max(1),
            timeout,
            scheduler,
            broadcast_sender,
            schedule_gate: AsyncMutex::new(()),
            last_schedule_time: Mutex::new(Instant::now()),
            next_task_id: AtomicU64::new(1),
            task_in_flight: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn task_in_flight(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.task_in_flight)
    }

    pub fn get(&self) -> usize {
        self.current_tokens.load(Ordering::Acquire)
    }

    pub fn reset(&self) {
        self.current_tokens.store(0, Ordering::Release);
    }

    pub async fn increment(&self, count: usize) -> bool {
        if count == 0 {
            return false;
        }

        let total = self.current_tokens.fetch_add(count, Ordering::AcqRel) + count;
        if total >= self.threshold {
            self.trigger_schedule().await;
            true
        } else {
            false
        }
    }

    pub async fn run(self: Arc<Self>) {
        let mut interval = tokio::time::interval(self.timeout);

        loop {
            interval.tick().await;
            if self.get() > 0 {
                self.trigger_schedule().await;
            }
        }
    }

    async fn trigger_schedule(&self) {
        let _guard = self.schedule_gate.lock().await;
        let pending = self.current_tokens.swap(0, Ordering::AcqRel);
        if pending == 0 {
            return;
        }

        if self
            .task_in_flight
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            self.current_tokens.fetch_add(pending, Ordering::AcqRel);
            return;
        }

        let mut scheduler = self.scheduler.lock().await;
        let (prefill_size, decode_size) = scheduler.schedule_batch();
        if prefill_size == 0 && decode_size == 0 {
            self.task_in_flight.store(false, Ordering::Release);
            self.current_tokens.fetch_add(pending, Ordering::AcqRel);
            return;
        }

        let task = ScheduleTask::new(
            prefill_size,
            decode_size,
            scheduler.prefill_list.clone(),
            scheduler.decode_list.clone(),
            self.next_task_id.fetch_add(1, Ordering::Relaxed),
        );

        if self.broadcast_sender.send(task).is_ok() {
            if let Ok(mut last_schedule_time) = self.last_schedule_time.lock() {
                *last_schedule_time = Instant::now();
            }
        } else {
            self.task_in_flight.store(false, Ordering::Release);
            self.current_tokens.fetch_add(pending, Ordering::AcqRel);
        }
    }
}
