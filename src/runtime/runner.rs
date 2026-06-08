use std::ops::{AddAssign, Neg, Sub};
use std::sync::Arc;

use tokio::sync::broadcast;
use tokio::sync::Barrier;
use tokio::task::JoinSet;

use crate::operators::operator::Operator;
use crate::operators::send_sync_ptr::SharedMut;

use crate::num_traits::{exp::Exp, neg_infinity::NegInfinity, sigmoid::Sigmoid, sqrt::Sqrt};
use crate::runtime::scheduling::types::ScheduleTask;
use crate::runtime::SequenceState;

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
        }
    }

    pub fn with_runner_count(mut self, runner_count: usize) -> Self {
        self.runner_count = runner_count.max(1);
        self
    }

    pub async fn start(self) {
        let thread_num = self.runner_count;

        let operator_queue: Arc<[Operator<T>]> = self.operator_queue.into();
        let barrier = Arc::new(Barrier::new(thread_num));
        let batch_list = Arc::clone(&self.batch_list);

        let mut join_set = JoinSet::new();

        for thread_id in 0..thread_num {
            let barrier = Arc::clone(&barrier);
            let queue = Arc::clone(&operator_queue);
            let batch_list = Arc::clone(&batch_list);
            let mut receiver = self.task_sender.subscribe();

            join_set.spawn(async move {
                while let Ok(task) = receiver.recv().await {
                    let (prefill_size, decode_size) = (task.prefill_size, task.decode_size);
                    let (prefill_list, decode_list) = (&task.prefill_list, &task.decode_list);

                    let batch_list_ptr = batch_list.get();
                    for operator in queue.iter() {
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
                    }

                    let _ = barrier.wait().await;
                }
            });
        }

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
    use crate::runtime::BatchScheduler;
    use tokio::sync::broadcast;

    #[tokio::test]
    async fn new_preserves_operator_queue_and_batch_layout() {
        let operator_queue = Vec::<crate::operators::operator::Operator<f32>>::new();
        let batch_scheduler = BatchScheduler::new(16, 4, 3);
        let (sender, _) = broadcast::channel(4);

        let runner = ServingRunner::new(operator_queue, batch_scheduler.batch_list.clone(), sender);

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
