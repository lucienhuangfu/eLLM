use core_affinity;
use std::ops::{AddAssign, Neg, Sub};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::Barrier;
use std::thread;

use tokio::sync::broadcast;

use crate::operators::operator::Operator;
use crate::operators::send_sync_ptr::SharedMut;

use crate::num_traits::{exp::Exp, neg_infinity::NegInfinity, sigmoid::Sigmoid, sqrt::Sqrt};
use crate::runtime::scheduling::task::ScheduleTask;
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
    stop_flag: Arc<AtomicBool>,
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
            stop_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn start(self) {
        let core_ids = core_affinity::get_core_ids().unwrap_or_default();
        let thread_num = core_ids.len().max(1);

        let operator_queue: Arc<[Operator<T>]> = self.operator_queue.into();
        let barrier = Arc::new(Barrier::new(thread_num));
        let batch_list = Arc::clone(&self.batch_list);
        let task_sender = self.task_sender.clone();
        let stop_flag = self.stop_flag;

        let mut handles = Vec::with_capacity(thread_num);

        for thread_id in 0..thread_num {
            let barrier = Arc::clone(&barrier);
            let queue = Arc::clone(&operator_queue);
            let batch_list = Arc::clone(&batch_list);
            let stop_flag = Arc::clone(&stop_flag);
            let task_sender = task_sender.clone();
            let core_id = core_ids.get(thread_id).copied();
            let mut receiver = task_sender.subscribe();

            let handle = thread::spawn(move || {
                if let Some(core_id) = core_id {
                    core_affinity::set_for_current(core_id);
                }

                loop {
                    if stop_flag.load(Ordering::Relaxed) {
                        break;
                    }

                    let task = match receiver.blocking_recv() {
                        Ok(task) => task,
                        Err(broadcast::error::RecvError::Closed) => break,
                        Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    };

                    if task.prefill_size == 0
                        && task.decode_size == 0
                        && task.prefill_list.is_empty()
                        && task.decode_list.is_empty()
                    {
                        if stop_flag.load(Ordering::Relaxed) {
                            break;
                        }
                        continue;
                    }

                    let batch_list_ptr = batch_list.get();
                    let (prefill_size, decode_size) = (task.prefill_size, task.decode_size);
                    let (prefill_list, decode_list) = (&task.prefill_list, &task.decode_list);

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
                        barrier.wait();
                    }

                    barrier.wait();

                    if thread_id == 0 {
                        let all_eos = unsafe {
                            (&*batch_list_ptr)
                                .iter()
                                .all(|s| matches!(s.phase, crate::runtime::Phase::Eos))
                        };
                        if all_eos && unsafe { (&*batch_list_ptr).is_empty() } == false {
                            stop_flag.store(true, Ordering::Relaxed);
                            let _ = task_sender.send(ScheduleTask::new(
                                0,
                                0,
                                Vec::new(),
                                Default::default(),
                                u64::MAX,
                            ));
                        }
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

#[cfg(test)]
mod tests {
    use super::ServingRunner;
    use crate::runtime::scheduling::task::ScheduleTask;
    use crate::runtime::BatchScheduler;
    use tokio::sync::broadcast;

    #[test]
    fn new_preserves_operator_queue_and_batch_layout() {
        let operator_queue = Vec::<crate::operators::operator::Operator<f32>>::new();
        let batch_scheduler = BatchScheduler::new(16, 4, 3);
        let (sender, _) = broadcast::channel(4);

        let runner =
            ServingRunner::new(operator_queue, batch_scheduler.batch_list.clone(), sender);

        assert_eq!(runner.operator_queue.len(), 0);
        assert_eq!(runner.batch_list.with(|list| list.len()), 0);
    }

    #[test]
    fn schedule_task_can_be_constructed() {
        let task = ScheduleTask::new(0, 0, Vec::new(), Default::default(), 1);
        assert_eq!(task.prefill_size, 0);
        assert_eq!(task.decode_size, 0);
        assert_eq!(task.task_id, 1);
    }
}
