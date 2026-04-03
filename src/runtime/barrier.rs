use std::hint::spin_loop;
use std::sync::atomic::{AtomicUsize, Ordering};

#[repr(align(64))]
struct CachePadded<T>(T);

pub struct Barrier {
    arrived: CachePadded<AtomicUsize>,
    generation: CachePadded<AtomicUsize>,
    tids: usize,
}

impl Barrier {
    pub fn new(tids: usize) -> Barrier {
        debug_assert!(tids > 0, "Barrier requires at least one participant");

        Barrier {
            arrived: CachePadded(AtomicUsize::new(0)),
            generation: CachePadded(AtomicUsize::new(0)),
            tids,
        }
    }

    #[inline]
    pub fn wait(&self) {
        let generation = self.generation.0.load(Ordering::Relaxed);
        let arrived = self.arrived.0.fetch_add(1, Ordering::AcqRel) + 1;

        if arrived == self.tids {
            self.arrived.0.store(0, Ordering::Relaxed);
            self.generation
                .0
                .store(generation.wrapping_add(1), Ordering::Release);
            return;
        }

        while self.generation.0.load(Ordering::Acquire) == generation {
            spin_loop();
        }
    }
    /*
    pub fn is_leader(&self) {
        todo!("MyStruct is not yet quxable");
    }
    */
}

#[cfg(test)]
mod test {
    use std::sync::Arc;
    use std::thread;

    use super::*;

    #[test]
    fn test_wait() {
        let thread_num: usize = num_cpus::get();
        let barrier = Barrier::new(thread_num);
        let barrier_arc = Arc::new(barrier);
        let num = AtomicUsize::new(thread_num);
        let barrier_num = Arc::new(num);
        thread::scope(|s| {
            let mut handles = Vec::with_capacity(thread_num);
            for thread_id in (0..thread_num) {
                let _thread_id = thread_id;
                // let runner_arc_clone = Arc::clone(&runner_arc);
                // let _runner = &runner;
                let b_a = barrier_arc.clone();
                let num_a = barrier_num.clone();
                let handle = s.spawn(move || {
                    // _runner.run(batch_size, thread_num, _thread_id);
                    num_a.fetch_sub(1, Ordering::SeqCst);
                    b_a.wait();
                    assert_eq!(num_a.load(Ordering::SeqCst), 0);
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.join().unwrap();
            }
        });
    }

    /*
    #[test]
    fn test_is_leader() {
        let thread_num: usize = num_cpus::get();
        let barrier = Barrier::new(thread_num);
        let barrier_arc = Arc::new(barrier);
        let num = AtomicUsize::new(thread_num);
        thread::scope(|s| {
            let mut handles = Vec::with_capacity(thread_num);
            for thread_id in (0..thread_num) {
                let _thread_id = thread_id;
                // let runner_arc_clone = Arc::clone(&runner_arc);
                let _runner = &runner;
                let handle = s.spawn(move || {
                    // _runner.run(batch_size, thread_num, _thread_id);
                    num.fetch_sub(1, Ordering::SeqCst);
                    let wait_result = barrier_arc.wait();
                    if wait_result.is_leader() {
                        assert_eq!(num.load(Ordering::SeqCst), 0);
                    }
                });
                handles.push(handle);
            }
            for handle in handles {
                handle.join().unwrap();
            }
        });
    }*/
}
