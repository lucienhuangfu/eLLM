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
use crate::runtime::schedule::BatchScheduler;

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

struct SharedState {
    sizes: (usize, usize),
    scheduler: BatchScheduler,
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
}
