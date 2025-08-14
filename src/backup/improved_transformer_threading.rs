use std::sync::{Arc, Barrier};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::Duration;
use core_affinity;

// 定义任务类型
#[derive(Debug, Clone)]
pub enum TransformerTask {
    ProcessSequence {
        batch_size: usize,
        prompt_begin: usize,
        prompt_end: usize,
        generation_end: usize,
        task_id: u64,
    },
    Shutdown,
}

// 定义结果类型
#[derive(Debug)]
pub struct TransformerResult {
    task_id: u64,
    processing_time: Duration,
    thread_id: usize,
    positions_processed: usize,
}

// 改进后的 Transformer 结构
pub struct ImprovedTransformer {
    worker_handles: Vec<thread::JoinHandle<()>>,
    task_senders: Vec<Sender<TransformerTask>>,
    result_receiver: Receiver<TransformerResult>,
    cpu_num: usize,
}

impl ImprovedTransformer {
    pub fn new(cpu_num: usize) -> Self {
        let mut worker_handles = Vec::new();
        let mut task_senders = Vec::new();
        let (result_sender, result_receiver) = mpsc::channel();
        
        // 获取CPU核心ID
        let core_ids = core_affinity::get_core_ids().unwrap_or_else(|| {
            (0..cpu_num).map(|i| core_affinity::CoreId { id: i }).collect()
        });
        
        // 为每个核心创建工作线程
        for (thread_id, core_id) in core_ids.into_iter().take(cpu_num).enumerate() {
            let (task_sender, task_receiver) = mpsc::channel();
            let result_sender_clone = result_sender.clone();
            
            let handle = thread::spawn(move || {
                // 设置线程亲和性
                core_affinity::set_for_current(core_id);
                println!("Worker thread {} started on core {:?}", thread_id, core_id);
                
                // 工作线程主循环
                Self::worker_loop(thread_id, task_receiver, result_sender_clone);
            });
            
            worker_handles.push(handle);
            task_senders.push(task_sender);
        }
        
        ImprovedTransformer {
            worker_handles,
            task_senders,
            result_receiver,
            cpu_num,
        }
    }
    
    fn worker_loop(
        thread_id: usize,
        task_receiver: Receiver<TransformerTask>,
        result_sender: Sender<TransformerResult>,
    ) {
        loop {
            match task_receiver.recv() {
                Ok(TransformerTask::ProcessSequence {
                    batch_size,
                    prompt_begin,
                    prompt_end,
                    generation_end,
                    task_id,
                }) => {
                    let start_time = std::time::Instant::now();
                    
                    // 模拟实际的 transformer 处理逻辑
                    let positions_processed = Self::process_sequence(
                        thread_id,
                        batch_size,
                        prompt_begin,
                        prompt_end,
                        generation_end,
                    );
                    
                    let processing_time = start_time.elapsed();
                    
                    // 发送结果
                    let result = TransformerResult {
                        task_id,
                        processing_time,
                        thread_id,
                        positions_processed,
                    };
                    
                    if result_sender.send(result).is_err() {
                        println!("Thread {} failed to send result", thread_id);
                        break;
                    }
                },
                Ok(TransformerTask::Shutdown) => {
                    println!("Thread {} received shutdown signal", thread_id);
                    break;
                },
                Err(_) => {
                    println!("Thread {} channel closed", thread_id);
                    break;
                }
            }
        }
        
        println!("Worker thread {} finished", thread_id);
    }
    
    fn process_sequence(
        thread_id: usize,
        batch_size: usize,
        prompt_begin: usize,
        prompt_end: usize,
        generation_end: usize,
    ) -> usize {
        println!(
            "Thread {} processing: batch_size={}, range={}..{}",
            thread_id, batch_size, prompt_end, generation_end
        );
        
        let mut positions_processed = 0;
        
        // 模拟处理每个位置
        for position_index in prompt_end..generation_end {
            // 这里是你原来的 operator 处理逻辑
            // for operator in queue.iter() {
            //     operator.run(batch_size, position_index, thread_id);
            //     barrier.wait(); // 同步点
            // }
            
            // 模拟处理时间
            thread::sleep(Duration::from_millis(1));
            positions_processed += 1;
            
            if positions_processed % 10 == 0 {
                println!("Thread {} processed {} positions", thread_id, positions_processed);
            }
        }
        
        positions_processed
    }
    
    // 提交任务到工作线程
    pub fn submit_task(&self, task: TransformerTask) -> Result<(), String> {
        // 简单的负载均衡：轮询分配
        static mut CURRENT_WORKER: usize = 0;
        
        unsafe {
            let worker_id = CURRENT_WORKER % self.cpu_num;
            CURRENT_WORKER += 1;
            
            self.task_senders[worker_id]
                .send(task)
                .map_err(|e| format!("Failed to send task to worker {}: {}", worker_id, e))
        }
    }
    
    // 等待并收集结果
    pub fn collect_results(&self, expected_count: usize) -> Vec<TransformerResult> {
        let mut results = Vec::new();
        
        for _ in 0..expected_count {
            match self.result_receiver.recv_timeout(Duration::from_secs(30)) {
                Ok(result) => {
                    println!("Received result from thread {}: {:?}", result.thread_id, result);
                    results.push(result);
                },
                Err(_) => {
                    println!("Timeout waiting for result");
                    break;
                }
            }
        }
        
        results
    }
    
    // 关闭所有工作线程
    pub fn shutdown(self) {
        println!("Shutting down transformer workers...");
        
        // 发送关闭信号给所有工作线程
        for sender in &self.task_senders {
            let _ = sender.send(TransformerTask::Shutdown);
        }
        
        // 等待所有线程结束
        for (i, handle) in self.worker_handles.into_iter().enumerate() {
            match handle.join() {
                Ok(_) => println!("Worker thread {} joined successfully", i),
                Err(_) => println!("Worker thread {} join failed", i),
            }
        }
        
        println!("All transformer workers shutdown complete");
    }
}

// 使用示例
fn main() {
    println!("=== Improved Transformer Threading Example ===");
    
    let cpu_num = std::thread::available_parallelism().unwrap().get().min(4);
    let transformer = ImprovedTransformer::new(cpu_num);
    
    println!("Created transformer with {} worker threads", cpu_num);
    
    // 提交一些任务
    let tasks = vec![
        TransformerTask::ProcessSequence {
            batch_size: 1,
            prompt_begin: 0,
            prompt_end: 0,
            generation_end: 20,
            task_id: 1,
        },
        TransformerTask::ProcessSequence {
            batch_size: 1,
            prompt_begin: 0,
            prompt_end: 20,
            generation_end: 40,
            task_id: 2,
        },
        TransformerTask::ProcessSequence {
            batch_size: 1,
            prompt_begin: 0,
            prompt_end: 40,
            generation_end: 60,
            task_id: 3,
        },
    ];
    
    // 提交任务
    for task in tasks {
        if let Err(e) = transformer.submit_task(task) {
            println!("Failed to submit task: {}", e);
        }
    }
    
    // 收集结果
    let results = transformer.collect_results(3);
    
    println!("Collected {} results:", results.len());
    for result in results {
        println!("  Task {}: {} positions in {:?} (thread {})",
            result.task_id,
            result.positions_processed,
            result.processing_time,
            result.thread_id
        );
    }
    
    // 关闭
    transformer.shutdown();
    
    println!("Example completed");
}
