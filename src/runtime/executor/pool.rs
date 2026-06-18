use std::sync::Arc;
use std::thread;

use super::barrier::SpinBarrier;
use super::worker::Worker;
use crate::operators::operator::Operator;
use crate::runtime::state::shared::SharedState;

pub struct ExecutorPool<T> {
    shared_state: Arc<SharedState>,
    operator_queue: Arc<[Operator<T>]>,
    thread_num: usize,
}

impl<T> ExecutorPool<T>
where
    T: Copy
        + Default
        + std::ops::Sub<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::AddAssign
        + crate::num_traits::exp::Exp
        + crate::num_traits::sqrt::Sqrt
        + crate::num_traits::neg_infinity::NegInfinity
        + crate::num_traits::sigmoid::Sigmoid
        + PartialOrd
        + Send
        + Sync
        + 'static,
{
    pub fn new(
        operator_queue: Vec<Operator<T>>,
        shared_state: Arc<SharedState>,
    ) -> Self {
        Self {
            shared_state,
            operator_queue: operator_queue.into(),
            thread_num: num_cpus::get(),
        }
    }

    pub fn with_thread_count(mut self, thread_num: usize) -> Self {
        self.thread_num = thread_num.max(1);
        self
    }

    pub fn start(self) {
        let barrier = Arc::new(SpinBarrier::new(self.thread_num));

        for thread_id in 0..self.thread_num {
            let shared_state = Arc::clone(&self.shared_state);
            let operator_queue = Arc::clone(&self.operator_queue);
            let barrier = Arc::clone(&barrier);
            let thread_num = self.thread_num;

            thread::spawn(move || {
                let worker = Worker::new(
                    shared_state,
                    operator_queue,
                    barrier,
                    thread_id,
                    thread_num,
                );
                worker.run();
            });
        }
    }

    pub fn shared_state(&self) -> Arc<SharedState> {
        Arc::clone(&self.shared_state)
    }
}