use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

const SPIN_LIMIT: u32 = 100;
const YIELD_LIMIT: u32 = 50;

pub fn adaptive_spin_loop<F>(condition: F)
where
    F: Fn() -> bool,
{
    let mut spin_count = 0u32;
    let mut yield_count = 0u32;

    loop {
        if condition() {
            return;
        }

        if spin_count < SPIN_LIMIT {
            std::hint::spin_loop();
            spin_count += 1;
        } else if yield_count < YIELD_LIMIT {
            std::thread::yield_now();
            yield_count += 1;
        } else {
            let sleep_us = 1 << (yield_count - YIELD_LIMIT).min(6);
            std::thread::sleep(Duration::from_micros(sleep_us));
            yield_count += 1;
        }
    }
}

#[derive(Debug)]
pub struct SpinBarrier {
    count: AtomicUsize,
    generation: AtomicU64,
    num_threads: usize,
}

impl SpinBarrier {
    #[inline]
    pub fn new(num_threads: usize) -> Self {
        assert!(num_threads > 0);
        Self {
            count: AtomicUsize::new(0),
            generation: AtomicU64::new(0),
            num_threads,
        }
    }

    #[inline]
    pub fn wait(&self) -> bool {
        let gen = self.generation.load(Ordering::Acquire);
        let prev = self.count.fetch_add(1, Ordering::AcqRel);

        if prev == self.num_threads - 1 {
            self.count.store(0, Ordering::Release);
            self.generation.fetch_add(1, Ordering::Release);
            true
        } else {
            adaptive_spin_loop(|| self.generation.load(Ordering::Acquire) != gen);
            false
        }
    }
}

#[derive(Debug, Default)]
pub struct AdaptiveWait;

impl AdaptiveWait {
    #[inline]
    pub fn new() -> Self {
        Self
    }

    #[inline]
    pub fn wait<F>(&mut self, condition: F)
    where
        F: Fn() -> bool,
    {
        adaptive_spin_loop(condition);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_spin_barrier_single_thread() {
        let barrier = SpinBarrier::new(1);
        let result = barrier.wait();
        assert!(result, "Single thread should return true from wait()");
    }

    #[test]
    fn test_spin_barrier_two_threads() {
        let barrier = Arc::new(SpinBarrier::new(2));
        let barrier_clone = barrier.clone();

        let handle = thread::spawn(move || barrier_clone.wait());

        let result1 = barrier.wait();
        let result2 = handle.join().unwrap();

        assert_ne!(
            result1, result2,
            "One thread should return true, the other false"
        );
        assert!(result1 || result2, "At least one thread should return true");
    }

    #[test]
    fn test_spin_barrier_multiple_threads() {
        let barrier = Arc::new(SpinBarrier::new(4));
        let mut handles = Vec::with_capacity(4);

        for _ in 0..4 {
            let b = barrier.clone();
            handles.push(thread::spawn(move || b.wait()));
        }

        let results: Vec<bool> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        let true_count = results.iter().filter(|&&r| r).count();
        assert_eq!(true_count, 1, "Exactly one thread should return true");
        assert_eq!(results.len(), 4, "All four threads should complete");
    }

    #[test]
    fn test_spin_barrier_multiple_generations() {
        let barrier = Arc::new(SpinBarrier::new(3));
        let mut handles = Vec::with_capacity(3);

        for _ in 0..3 {
            let b = barrier.clone();
            handles.push(thread::spawn(move || {
                let first_result = b.wait();
                thread::yield_now();
                let second_result = b.wait();
                (first_result, second_result)
            }));
        }

        let results: Vec<(bool, bool)> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        let first_true_count = results.iter().filter(|(r, _)| *r).count();
        assert_eq!(
            first_true_count, 1,
            "First barrier: exactly one thread should return true"
        );

        let second_true_count = results.iter().filter(|(_, r)| *r).count();
        assert_eq!(
            second_true_count, 1,
            "Second barrier: exactly one thread should return true"
        );
    }

    #[test]
    fn test_adaptive_wait() {
        let flag = Arc::new(AtomicBool::new(false));
        let flag_clone = flag.clone();

        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(10));
            flag_clone.store(true, Ordering::Release);
        });

        let mut wait = AdaptiveWait::new();
        wait.wait(|| flag.load(Ordering::Acquire));

        handle.join().unwrap();
        assert!(flag.load(Ordering::Acquire));
    }
}
