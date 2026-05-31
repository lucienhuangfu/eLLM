use core_affinity;
use std::cell::SyncUnsafeCell;
use std::ops::{AddAssign, Neg, Sub};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crate::operators::operator::Operator;
use crate::runtime::spin_barrier::SpinBarrier;

use crate::num_traits::{exp::Exp, neg_infinity::NegInfinity, sigmoid::Sigmoid, sqrt::Sqrt};
use crate::runtime::BatchScheduler;

/// Runs the inference serving loop.
///
/// This initializes a thread pool where Thread 0 schedules tasks by monitoring
/// user request phases (Prefill/Decode) and populating the token list. All threads
/// then synchronize to execute the operators in the queue for the current batch.
pub struct ServingRunner<T> {
    operator_queue: Vec<Operator<T>>,
    batch_scheduler: BatchScheduler,
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
    pub fn new(operator_queue: Vec<Operator<T>>, batch_scheduler: BatchScheduler) -> Self {
        Self {
            operator_queue,
            batch_scheduler,
            stop_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn start(self) {
        let ServingRunner {
            operator_queue,
            mut batch_scheduler,
            stop_flag,
        } = self;
        let all_core_ids = core_affinity::get_core_ids().unwrap_or_default();
        // Filter to physical cores only: skip hyperthread siblings to avoid
        // AVX-512 execution-unit contention. 每个物理核只用一条超线程,
        // 避免 AVX-512 执行单元竞争.
        let core_ids: Vec<_> = all_core_ids
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 0)
            .map(|(_, &id)| id)
            .collect();
        let requested_thread_num = batch_scheduler.thread_num().max(1);
        let thread_num = if core_ids.is_empty() {
            requested_thread_num
        } else {
            requested_thread_num.min(core_ids.len()).max(1)
        };
        batch_scheduler.set_thread_num(thread_num);

        let operator_queue: Arc<[Operator<T>]> = operator_queue.into();

        let barrier = Arc::new(SpinBarrier::new(thread_num));
        let shared_sizes = Arc::new(SyncUnsafeCell::new((0usize, 0usize)));
        let shared_scheduler = Arc::new(SyncUnsafeCell::new(batch_scheduler));
        let profile_enabled = std::env::var_os("ELLM_PROFILE").is_some();

        let mut handles = Vec::with_capacity(thread_num);

        for thread_id in 0..thread_num {
            let barrier = Arc::clone(&barrier);
            let queue = Arc::clone(&operator_queue);
            let shared_sizes = Arc::clone(&shared_sizes);
            let shared_scheduler = Arc::clone(&shared_scheduler);
            let stop_flag = Arc::clone(&stop_flag);
            let core_id = core_ids.get(thread_id).copied();
            let profile_enabled = profile_enabled && thread_id == 0;

            let handle = thread::spawn(move || {
                if let Some(core_id) = core_id {
                    core_affinity::set_for_current(core_id);
                }

                let sizes_ptr = shared_sizes.get();
                let scheduler_ptr = shared_scheduler.get();
                let mut profile_run_totals = if profile_enabled {
                    vec![Duration::ZERO; queue.len()]
                } else {
                    Vec::new()
                };
                let mut profile_wait_totals = if profile_enabled {
                    vec![Duration::ZERO; queue.len()]
                } else {
                    Vec::new()
                };
                let mut profile_counts = if profile_enabled {
                    vec![0usize; queue.len()]
                } else {
                    Vec::new()
                };

                loop {
                    if stop_flag.load(Ordering::Relaxed) {
                        break;
                    }

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

                    for (operator_index, operator) in queue.iter().enumerate() {
                        let profile_start = profile_enabled.then(Instant::now);
                        operator.run(
                            prefill_size,
                            decode_size,
                            thread_num,
                            thread_id,
                            prefill_list,
                            decode_list,
                            batch_list,
                        );
                        let profile_wait_start = profile_enabled.then(Instant::now);
                        barrier.wait();
                        if let (Some(start), Some(wait_start)) = (profile_start, profile_wait_start)
                        {
                            profile_run_totals[operator_index] += wait_start.duration_since(start);
                            profile_wait_totals[operator_index] += wait_start.elapsed();
                            profile_counts[operator_index] += 1;
                        }
                    }

                    if thread_id == 0 {
                        let all_eos = batch_list
                            .iter()
                            .all(|s| matches!(s.phase, crate::runtime::Phase::Eos));
                        if all_eos && !batch_list.is_empty() {
                            stop_flag.store(true, Ordering::Relaxed);
                        }
                    }
                    barrier.wait();
                }

                if profile_enabled {
                    let mut rows = profile_run_totals
                        .iter()
                        .enumerate()
                        .filter_map(|(index, run_total)| {
                            let wait_total = profile_wait_totals[index];
                            ((*run_total + wait_total) > Duration::ZERO).then_some((
                                index,
                                queue[index].kind(),
                                *run_total,
                                wait_total,
                                profile_counts[index],
                            ))
                        })
                        .collect::<Vec<_>>();
                    rows.sort_by(|a, b| (b.2 + b.3).cmp(&(a.2 + a.3)));

                    eprintln!("=== ELLM operator profile (thread 0 run/wait) ===");
                    let total_run = profile_run_totals
                        .iter()
                        .copied()
                        .fold(Duration::ZERO, |acc, item| acc + item);
                    let total_wait = profile_wait_totals
                        .iter()
                        .copied()
                        .fold(Duration::ZERO, |acc, item| acc + item);
                    eprintln!(
                        "TOTAL run={:.3}s wait={:.3}s total={:.3}s",
                        total_run.as_secs_f64(),
                        total_wait.as_secs_f64(),
                        (total_run + total_wait).as_secs_f64()
                    );

                    let mut kind_rows: Vec<(&'static str, Duration, Duration, usize)> = Vec::new();
                    for (index, run_total) in profile_run_totals.iter().enumerate() {
                        let wait_total = profile_wait_totals[index];
                        if (*run_total + wait_total) == Duration::ZERO {
                            continue;
                        }
                        let kind = queue[index].kind();
                        if let Some(row) = kind_rows.iter_mut().find(|row| row.0 == kind) {
                            row.1 += *run_total;
                            row.2 += wait_total;
                            row.3 += profile_counts[index];
                        } else {
                            kind_rows.push((kind, *run_total, wait_total, profile_counts[index]));
                        }
                    }
                    kind_rows.sort_by(|a, b| (b.1 + b.2).cmp(&(a.1 + a.2)));
                    eprintln!("--- by kind ---");
                    for (kind, run_total, wait_total, count) in kind_rows.into_iter().take(12) {
                        eprintln!(
                            "{kind:<24} run={:.3}s wait={:.3}s total={:.3}s count={count}",
                            run_total.as_secs_f64(),
                            wait_total.as_secs_f64(),
                            (run_total + wait_total).as_secs_f64()
                        );
                    }
                    eprintln!("--- top operators ---");
                    for (index, kind, run_total, wait_total, count) in rows.into_iter().take(20) {
                        eprintln!(
                            "#{index:03} {kind:<24} run={:.3}s wait={:.3}s total={:.3}s count={count}",
                            run_total.as_secs_f64(),
                            wait_total.as_secs_f64(),
                            (run_total + wait_total).as_secs_f64()
                        );
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
    use crate::runtime::BatchScheduler;

    #[test]
    fn new_preserves_operator_queue_and_scheduler_layout() {
        let operator_queue = Vec::<crate::operators::operator::Operator<f32>>::new();
        let batch_scheduler = BatchScheduler::new(16, 4, 3);

        let runner = ServingRunner::new(operator_queue, batch_scheduler);

        assert_eq!(runner.operator_queue.len(), 0);
        assert_eq!(runner.batch_scheduler.prefill_list.len(), 3);
        assert_eq!(runner.batch_scheduler.prefill_list[0].capacity(), 4);
        assert_eq!(runner.batch_scheduler.decode_list.len(), 0);
    }
}
