use std::sync::atomic::{AtomicUsize, Ordering};

/// A fast spin-based barrier for fine-grained operator synchronization.
/// Avoids the kernel-syscall overhead of std::sync::Barrier (futex).
/// Suitable when threads are expected to arrive within microseconds of each other.
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

    /// Returns true for the last thread to arrive (the leader).
    pub fn wait(&self) -> bool {
        self.wait_with(self.num_threads)
    }

    /// Like wait() but only expects `n` threads to arrive.
    /// All callers in the same barrier round MUST pass the same `n`.
    pub fn wait_with(&self, n: usize) -> bool {
        debug_assert!(n > 0);
        let gen = self.generation.load(Ordering::Acquire);
        let prev = self.count.fetch_add(1, Ordering::AcqRel);
        if prev == n - 1 {
            // Last thread to arrive — reset and flip generation.
            self.count.store(0, Ordering::Release);
            self.generation.fetch_add(1, Ordering::Release);
            true
        } else {
            // Spin until generation changes (last thread flips it).
            while self.generation.load(Ordering::Acquire) == gen {
                std::hint::spin_loop();
            }
            false
        }
    }
}
