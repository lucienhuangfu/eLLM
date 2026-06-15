use std::sync::atomic::{AtomicUsize, Ordering};

pub struct SpinBarrier {
    count: AtomicUsize,
    generation: AtomicUsize,
    num_threads: usize,
}

impl SpinBarrier {
    pub fn new(num_threads: usize) -> Self {
        assert!(num_threads > 0);
        Self {
            count: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
            num_threads,
        }
    }

    pub fn wait(&self) -> bool {
        let gen = self.generation.load(Ordering::Acquire);
        let prev = self.count.fetch_add(1, Ordering::AcqRel);
        if prev == self.num_threads - 1 {
            self.count.store(0, Ordering::Release);
            self.generation.fetch_add(1, Ordering::Release);
            true
        } else {
            while self.generation.load(Ordering::Acquire) == gen {
                std::hint::spin_loop();
            }
            false
        }
    }
}
