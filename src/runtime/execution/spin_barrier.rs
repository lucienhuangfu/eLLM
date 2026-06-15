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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    /// 测试单线程 barrier 行为
    /// 验证: 单个线程调用 wait() 应返回 true（因为它是唯一的线程）
    #[test]
    fn test_single_thread() {
        let barrier = SpinBarrier::new(1);
        
        // 单个线程调用 wait() 应该立即返回 true
        let result = barrier.wait();
        assert!(result, "Single thread should return true from wait()");
    }

    /// 测试两个线程的 barrier 同步
    /// 验证: 两个线程在 barrier 处正确同步，且只有最后一个到达的线程返回 true
    #[test]
    fn test_two_threads() {
        let barrier = Arc::new(SpinBarrier::new(2));
        let barrier_clone = barrier.clone();
        
        let handle = thread::spawn(move || {
            barrier_clone.wait()
        });
        
        let result1 = barrier.wait();
        let result2 = handle.join().unwrap();
        
        // 一个返回 true（最后到达的），一个返回 false（第一个到达的）
        assert_ne!(result1, result2, "One thread should return true, the other false");
        assert!(result1 || result2, "At least one thread should return true");
    }

    /// 测试多个线程（4个）的 barrier 同步
    /// 验证: 所有线程在 barrier 处正确同步，且只有最后一个到达的线程返回 true
    #[test]
    fn test_multiple_threads() {
        let barrier = Arc::new(SpinBarrier::new(4));
        let mut handles = Vec::with_capacity(4);
        
        for _ in 0..4 {
            let b = barrier.clone();
            handles.push(thread::spawn(move || b.wait()));
        }
        
        let results: Vec<bool> = handles.into_iter()
            .map(|h| h.join().unwrap())
            .collect();
        
        // 应该恰好有一个线程返回 true（最后到达的）
        let true_count = results.iter().filter(|&&r| r).count();
        assert_eq!(true_count, 1, "Exactly one thread should return true");
        assert_eq!(results.len(), 4, "All four threads should complete");
    }

    /// 测试 barrier 的多次使用（多个 generation）
    /// 验证: barrier 可以被重复使用，每次使用时正确同步
    #[test]
    fn test_multiple_generations() {
        let barrier = Arc::new(SpinBarrier::new(3));
        let mut handles = Vec::with_capacity(3);
        
        for _ in 0..3 {
            let b = barrier.clone();
            handles.push(thread::spawn(move || {
                // 第一次 barrier
                let first_result = b.wait();
                
                // 短暂延迟确保所有线程都通过第一个 barrier
                thread::yield_now();
                
                // 第二次 barrier
                let second_result = b.wait();
                
                (first_result, second_result)
            }));
        }
        
        let results: Vec<(bool, bool)> = handles.into_iter()
            .map(|h| h.join().unwrap())
            .collect();
        
        // 第一次 barrier 应该恰好有一个 true
        let first_true_count = results.iter().filter(|(r, _)| *r).count();
        assert_eq!(first_true_count, 1, "First barrier: exactly one thread should return true");
        
        // 第二次 barrier 应该恰好有一个 true（可能是不同的线程）
        let second_true_count = results.iter().filter(|(_, r)| *r).count();
        assert_eq!(second_true_count, 1, "Second barrier: exactly one thread should return true");
    }

    /// 测试大量线程场景（10个线程）
    /// 验证: barrier 在高并发场景下仍能正确工作
    #[test]
    fn test_large_thread_count() {
        const NUM_THREADS: usize = 10;
        let barrier = Arc::new(SpinBarrier::new(NUM_THREADS));
        let mut handles = Vec::with_capacity(NUM_THREADS);
        
        for _ in 0..NUM_THREADS {
            let b = barrier.clone();
            handles.push(thread::spawn(move || b.wait()));
        }
        
        let results: Vec<bool> = handles.into_iter()
            .map(|h| h.join().unwrap())
            .collect();
        
        // 恰好有一个线程返回 true
        let true_count = results.iter().filter(|&&r| r).count();
        assert_eq!(true_count, 1, "Exactly one thread should return true among {} threads", NUM_THREADS);
    }

    /// 测试零线程 panic
    /// 验证: 创建零线程的 barrier 会触发断言失败
    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_zero_threads_panic() {
        let _ = SpinBarrier::new(0);
    }
}
