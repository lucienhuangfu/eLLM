use core_affinity;
use std::cell::SyncUnsafeCell;
use std::ops::{AddAssign, Neg, Sub};
use std::sync::Arc;
use std::sync::Barrier;
use std::thread;

use crate::runtime::operator::Operator;

use crate::common::num_traits::{
    exp::Exp, neg_infinity::NegInfinity, sigmoid::Sigmoid, sqrt::Sqrt,
};
use crate::runtime::BatchScheduler;

/// Runs the inference serving loop.
///
/// This initializes a thread pool where Thread 0 schedules tasks by monitoring
/// user request phases (Prefill/Decode) and populating the token list. All threads
/// then synchronize to execute the operators in the queue for the current batch.
pub struct ServingRunner<T> {
    operator_queue: Vec<Operator<T>>,
    batch_scheduler: BatchScheduler,
    temperature_list: Arc<[T]>,
}

impl<T> ServingRunner<T>
where
    T: Copy
        + Default
        + Sub<Output = T>
        + Neg<Output = T>
        + AddAssign
        + Exp
        + Sqrt
        + NegInfinity
        + Sigmoid
        + PartialOrd
        + Send
        + Sync
        + 'static,
{
    pub fn new(operator_queue: Vec<Operator<T>>, batch_scheduler: BatchScheduler) -> Self {
        Self {
            operator_queue,
            batch_scheduler,
            temperature_list: Vec::<T>::new().into(),
        }
    }

    pub fn with_temperature_list(
        operator_queue: Vec<Operator<T>>,
        batch_scheduler: BatchScheduler,
        temperature_list: Vec<T>,
    ) -> Self {
        Self {
            operator_queue,
            batch_scheduler,
            temperature_list: temperature_list.into(),
        }
    }

    pub fn start(self) {
        let core_ids = core_affinity::get_core_ids().unwrap_or_default();
        let thread_num = core_ids.len().max(1);

        let operator_queue: Arc<[Operator<T>]> = self.operator_queue.into();
        let temperature_list = self.temperature_list;

        let barrier = Arc::new(Barrier::new(thread_num));
        let shared_sizes = Arc::new(SyncUnsafeCell::new((0usize, 0usize)));
        let shared_scheduler = Arc::new(SyncUnsafeCell::new(self.batch_scheduler));

        let mut handles = Vec::with_capacity(thread_num);

        for thread_id in 0..thread_num {
            let barrier = Arc::clone(&barrier);
            let queue = Arc::clone(&operator_queue);
            let temperature_list = Arc::clone(&temperature_list);
            let shared_sizes = Arc::clone(&shared_sizes);
            let shared_scheduler = Arc::clone(&shared_scheduler);
            let core_id = core_ids.get(thread_id).copied();

            let handle = thread::spawn(move || {
                if let Some(core_id) = core_id {
                    core_affinity::set_for_current(core_id);
                }

                let sizes_ptr = shared_sizes.get();
                let scheduler_ptr = shared_scheduler.get();

                loop {
                    if thread_id == 0 {
                        unsafe {
                            let scheduler = &mut *scheduler_ptr;
                            *sizes_ptr = scheduler.schedule_batch();
                        }
                    }

                    barrier.wait();

                    let (prefill_size, decode_size) = unsafe { *sizes_ptr };
                    let (prefill_list, decode_list, batch_list) = unsafe {
                        let scheduler = &mut *scheduler_ptr;
                        (
                            &scheduler.prefill_list,
                            &scheduler.decode_list,
                            &mut *scheduler.batch_list.get(),
                        )
                    };

                    for operator in queue.iter() {
                        operator.run(
                            prefill_size,
                            decode_size,
                            thread_num,
                            thread_id,
                            prefill_list,
                            decode_list,
                            &temperature_list,
                            batch_list,
                        );
                        barrier.wait();
                    }
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            let _ = handle.join();
        }
    }
}
