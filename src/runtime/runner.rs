use std::collections::BTreeMap;
use std::ops::{AddAssign, Neg, Sub};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::broadcast;
use tokio::task::JoinSet;

use crate::operators::operator::Operator;
use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::spin_barrier::SpinBarrier;

use crate::num_traits::{exp::Exp, neg_infinity::NegInfinity, sigmoid::Sigmoid, sqrt::Sqrt};
use crate::runtime::scheduling::types::Phase;
use crate::runtime::scheduling::types::ScheduleTask;
use crate::runtime::SequenceState;

struct ProfileRow {
    kind: &'static str,
    pre_barrier: f64,
    run: f64,
    post_barrier: f64,
}

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
        let profile_ops = std::env::var("ELLM_PROFILE_OPS")
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);
        let profile_decode_ops = std::env::var("ELLM_PROFILE_DECODE_OPS")
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);
        let profile_decode_step = std::env::var("ELLM_PROFILE_DECODE_STEP")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(1);

        let mut join_set = JoinSet::new();

        for thread_id in 0..thread_num {
            let barrier = Arc::clone(&barrier);
            let queue = Arc::clone(&operator_queue);
            let batch_list = Arc::clone(&batch_list);
            let task_in_flight = task_in_flight.clone();
            let mut receiver = task_sender.subscribe();
            let profile_ops = profile_ops;
            let profile_decode_ops = profile_decode_ops;

            join_set.spawn(async move {
                let mut profile_rows: Vec<ProfileRow> = Vec::new();
                let mut profile_task_start_barrier = 0.0f64;
                let mut profile_leader_barrier = 0.0f64;
                let mut decode_task_index = 0usize;
                while let Ok(task) = receiver.recv().await {
                    let n_threads = task.thread_count;
                    // Idle threads skip this task entirely — no barriers, no work.
                    if thread_id >= n_threads {
                        continue;
                    }
                    let oper_thread_num = n_threads; // operators see the active count
                    let (prefill_size, decode_size) = (task.prefill_size, task.decode_size);
                    let (prefill_list, decode_list) = (&task.prefill_list, &task.decode_list);
                    if prefill_size == 0 && decode_size > 0 {
                        decode_task_index += 1;
                    }
                    let profile_prefix = if profile_ops && thread_id == 0 && prefill_size > 0 {
                        Some("prefill_profile")
                    } else if profile_decode_ops
                        && thread_id == 0
                        && prefill_size == 0
                        && decode_size > 0
                        && decode_task_index == profile_decode_step
                    {
                        Some("decode_profile")
                    } else {
                        None
                    };
                    let profile_this_task = profile_prefix.is_some();
                    if profile_this_task {
                        profile_rows.clear();
                    }
                    let before = {
                        let batch_list_ptr = batch_list.get();
                        unsafe { snapshot_sequences(&*batch_list_ptr) }
                    };

                    let barrier_start = if profile_this_task {
                        Some(Instant::now())
                    } else {
                        None
                    };
                    barrier.wait_with(n_threads);
                    if let Some(start) = barrier_start {
                        profile_task_start_barrier = start.elapsed().as_secs_f64();
                    }
                    for operator in queue.iter() {
                        let run_start = if profile_this_task {
                            Some(Instant::now())
                        } else {
                            None
                        };
                        let batch_list_ptr = batch_list.get();
                        unsafe {
                            let batch_list_ref = &mut *batch_list_ptr;
                            operator.run(
                                prefill_size,
                                decode_size,
                                oper_thread_num,
                                thread_id,
                                prefill_list,
                                decode_list,
                                batch_list_ref,
                            );
                        }
                        let run = run_start
                            .map(|start| start.elapsed().as_secs_f64())
                            .unwrap_or(0.0);
                        let post_barrier_start = if profile_this_task {
                            Some(Instant::now())
                        } else {
                            None
                        };
                        barrier.wait_with(n_threads);
                        if let Some(start) = post_barrier_start {
                            profile_rows.push(ProfileRow {
                                kind: operator.kind(),
                                pre_barrier: 0.0,
                                run,
                                post_barrier: start.elapsed().as_secs_f64(),
                            });
                        }
                    }

                    let leader_barrier_start = if profile_this_task {
                        Some(Instant::now())
                    } else {
                        None
                    };
                    let is_leader = barrier.wait_with(n_threads);
                    if let Some(start) = leader_barrier_start {
                        profile_leader_barrier = start.elapsed().as_secs_f64();
                    }
                    if is_leader {
                        let batch_list_ptr = batch_list.get();
                        unsafe {
                            notify_completed_sequences(&before, &*batch_list_ptr);
                        }
                        if let Some(task_in_flight) = &task_in_flight {
                            task_in_flight.store(false, Ordering::Release);
                        }
                    }
                    if profile_this_task {
                        let mut by_kind: BTreeMap<&'static str, (usize, f64, f64, f64)> =
                            BTreeMap::new();
                        let mut run_total = 0.0f64;
                        let mut pre_barrier_total = profile_task_start_barrier;
                        let mut post_barrier_total = 0.0f64;
                        for row in &profile_rows {
                            run_total += row.run;
                            pre_barrier_total += row.pre_barrier;
                            post_barrier_total += row.post_barrier;
                            let entry = by_kind.entry(row.kind).or_insert((0, 0.0, 0.0, 0.0));
                            entry.0 += 1;
                            entry.1 += row.run;
                            entry.2 += row.pre_barrier;
                            entry.3 += row.post_barrier;
                        }
                        post_barrier_total += profile_leader_barrier;
                        let profile_prefix = profile_prefix.unwrap();
                        eprintln!(
                            "{profile_prefix} total_ops={} run_sum={:.6}s barrier_sum={:.6}s pre_barrier_sum={:.6}s post_barrier_sum={:.6}s prefill_size={} decode_size={} decode_step={}",
                            profile_rows.len(),
                            run_total,
                            pre_barrier_total + post_barrier_total,
                            pre_barrier_total,
                            post_barrier_total,
                            prefill_size,
                            decode_size,
                            decode_task_index
                        );
                        eprintln!(
                            "{profile_prefix} barriers task_start={:.6}s leader={:.6}s final_pending",
                            profile_task_start_barrier,
                            profile_leader_barrier
                        );
                        for (kind, (count, run, pre_barrier, post_barrier)) in by_kind {
                            eprintln!(
                                "{profile_prefix} kind={kind} count={count} run={:.6}s run_avg={:.6}s pre_barrier={:.6}s post_barrier={:.6}s barrier={:.6}s",
                                run,
                                run / count as f64,
                                pre_barrier,
                                post_barrier,
                                pre_barrier + post_barrier
                            );
                        }
                    }
                    let final_barrier_start = if profile_this_task {
                        Some(Instant::now())
                    } else {
                        None
                    };
                    barrier.wait_with(n_threads);
                    if let Some(start) = final_barrier_start {
                        let profile_final_barrier = start.elapsed().as_secs_f64();
                        if let Some(profile_prefix) = profile_prefix {
                            eprintln!(
                                "{profile_prefix} barriers final={:.6}s",
                                profile_final_barrier
                            );
                        }
                    }
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
