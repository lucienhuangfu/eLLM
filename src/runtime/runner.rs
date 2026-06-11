use std::ops::{AddAssign, Neg, Sub};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tokio::sync::broadcast;
use tokio::task::JoinSet;

use crate::operators::operator::Operator;
use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::scheduling::types::Phase;
use crate::runtime::scheduling::types::ScheduleTask;
use crate::runtime::sequence_slice::{DecodeList, SequenceSlice};
use crate::runtime::spin_barrier::SpinBarrier;
use crate::runtime::SequenceState;

use crate::num_traits::{exp::Exp, neg_infinity::NegInfinity, sigmoid::Sigmoid, sqrt::Sqrt};

#[derive(Clone, Copy)]
struct SequenceSnapshot {
    sequence_index: usize,
    phase: Phase,
}

fn snapshot_sequences(batch_list: &[SequenceState]) -> Vec<SequenceSnapshot> {
    batch_list
        .iter()
        .map(|record| SequenceSnapshot {
            sequence_index: record.sequence_index,
            phase: record.phase,
        })
        .collect()
}

fn notify_completed_sequences(before: &[SequenceSnapshot], batch_list: &[SequenceState]) {
    for (index, record) in batch_list.iter().enumerate() {
        let Some(previous) = before.get(index) else {
            continue;
        };

        if matches!(record.phase, Phase::Prefill) {
            continue;
        }

        let token_or_eos_ready =
            record.sequence_index != previous.sequence_index || record.phase != previous.phase;
        if token_or_eos_ready {
            record.notify.notify_one();
        }
    }
}

/// Runs the inference serving loop.
///
/// Each worker subscribes to the schedule broadcast stream. When a task arrives,
/// all workers synchronize on a barrier, run the operator queue in order, and
/// then return to waiting for the next schedule event.
pub struct ServingRunner<T> {
    operator_queue: Vec<Operator<T>>,
    batch_list: Arc<SharedMut<Vec<SequenceState>>>,
    task_sender: broadcast::Sender<ScheduleTask>,
    runner_count: usize,
    task_in_flight: Option<Arc<AtomicBool>>,
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
    pub fn new(
        operator_queue: Vec<Operator<T>>,
        batch_list: Arc<SharedMut<Vec<SequenceState>>>,
        task_sender: broadcast::Sender<ScheduleTask>,
    ) -> Self {
        Self {
            operator_queue,
            batch_list,
            task_sender,
            runner_count: num_cpus::get(),
            task_in_flight: None,
        }
    }

    pub fn with_runner_count(mut self, runner_count: usize) -> Self {
        self.runner_count = runner_count.max(1);
        self
    }

    pub fn with_task_in_flight(mut self, task_in_flight: Arc<AtomicBool>) -> Self {
        self.task_in_flight = Some(task_in_flight);
        self
    }

    pub async fn start(self) {
        let ServingRunner {
            operator_queue,
            batch_list,
            task_sender,
            runner_count,
            task_in_flight,
        } = self;
        let thread_num = runner_count;

        let operator_queue: Arc<[Operator<T>]> = operator_queue.into();
        let barrier = Arc::new(SpinBarrier::new(thread_num));
        let batch_list = Arc::clone(&batch_list);

        let mut join_set = JoinSet::new();

        for thread_id in 0..thread_num {
            let barrier = Arc::clone(&barrier);
            let queue = Arc::clone(&operator_queue);
            let batch_list = Arc::clone(&batch_list);
            let task_in_flight = task_in_flight.clone();
            let mut receiver = task_sender.subscribe();

            join_set.spawn(async move {
                while let Ok(task) = receiver.recv().await {
                    let (prefill_size, decode_size) = (task.prefill_size, task.decode_size);
                    // Dereference Arc to get references to underlying data
                    let prefill_list: &Vec<Vec<SequenceSlice>> = task.prefill_list.as_ref();
                    let decode_list: &DecodeList = task.decode_list.as_ref();
                    let before = {
                        let batch_list_ptr = batch_list.get();
                        unsafe { snapshot_sequences(&*batch_list_ptr) }
                    };

                    barrier.wait();
                    for operator in queue.iter() {
                        barrier.wait();
                        let batch_list_ptr = batch_list.get();
                        unsafe {
                            let batch_list_ref = &mut *batch_list_ptr;
                            operator.run(
                                prefill_size,
                                decode_size,
                                thread_num,
                                thread_id,
                                prefill_list,
                                decode_list,
                                batch_list_ref,
                            );
                        }
                        barrier.wait();
                    }

                    let is_leader = barrier.wait();
                    if is_leader {
                        let batch_list_ptr = batch_list.get();
                        unsafe {
                            notify_completed_sequences(&before, &*batch_list_ptr);
                        }
                        if let Some(task_in_flight) = &task_in_flight {
                            task_in_flight.store(false, Ordering::Release);
                        }
                    }
                    barrier.wait();
                }
            });
        }
        drop(task_sender);

        while let Some(res) = join_set.join_next().await {
            if let Err(e) = res {
                eprintln!("Task failed: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ServingRunner;
    use crate::runtime::scheduling::types::ScheduleTask;
    use crate::runtime::InferenceScheduler;
    use tokio::sync::broadcast;

    #[tokio::test]
    async fn new_preserves_operator_queue_and_batch_layout() {
        use crate::operators::send_sync_ptr::SharedMut;
        use std::sync::Arc;

        let operator_queue = Vec::<crate::operators::operator::Operator<f32>>::new();
        let (sender, _) = broadcast::channel(4);
        let batch_list = Arc::new(SharedMut::new(Vec::new()));
        let batch_scheduler = InferenceScheduler::new(
            16,
            4,
            3,
            1,
            std::time::Duration::from_millis(100),
            sender.clone(),
            batch_list,
        );

        let runner = ServingRunner::new(operator_queue, batch_scheduler.batch_list(), sender);

        assert_eq!(runner.operator_queue.len(), 0);
        assert_eq!(runner.batch_list.with(|list| list.len()), 0);
    }

    #[tokio::test]
    async fn schedule_task_can_be_constructed() {
        let task = ScheduleTask::new(0, 0, Vec::new(), Default::default(), 1);
        assert_eq!(task.prefill_size, 0);
        assert_eq!(task.decode_size, 0);
        assert_eq!(task.task_id, 1);
    }
}
