use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

pub struct BatchTracker {
    total: AtomicUsize,
    remaining: AtomicUsize,
    completed: AtomicBool,
}

impl BatchTracker {
    pub fn new() -> Self {
        Self {
            total: AtomicUsize::new(0),
            remaining: AtomicUsize::new(0),
            completed: AtomicBool::new(true),
        }
    }

    pub fn reset(&self, count: usize) {
        self.total.store(count, Ordering::Release);
        self.remaining.store(count, Ordering::Release);
        self.completed.store(count == 0, Ordering::Release);
    }

    pub fn complete_slot(&self) -> bool {
        let prev = self.remaining.fetch_sub(1, Ordering::AcqRel);
        if prev == 1 {
            self.completed.store(true, Ordering::Release);
            true
        } else {
            false
        }
    }

    pub fn is_complete(&self) -> bool {
        self.completed.load(Ordering::Acquire)
    }

    pub fn wait_complete(&self) {
        if self.is_complete() {
            return;
        }

        for _ in 0..10000 {
            if self.is_complete() {
                return;
            }
            std::hint::spin_loop();
        }

        while !self.is_complete() {
            std::thread::yield_now();
        }
    }

    pub fn total(&self) -> usize {
        self.total.load(Ordering::Acquire)
    }

    pub fn remaining(&self) -> usize {
        self.remaining.load(Ordering::Acquire)
    }
}