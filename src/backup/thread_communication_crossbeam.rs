// 需要在 Cargo.toml 中添加: crossbeam = "0.8"

use crossbeam::channel::{self, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug)]
enum Message {
    Work { id: u64, data: Vec<f32> },
    GetStatus,
    Shutdown,
}

#[derive(Debug)]
enum Response {
    WorkCompleted { id: u64, result: Vec<f32>, duration: Duration },
    Status { processed_count: u64, uptime: Duration },
    ShutdownAck,
}

struct Worker {
    task_rx: Receiver<Message>,
    response_tx: Sender<Response>,
    processed_count: u64,
    start_time: Instant,
}

impl Worker {
    fn new(task_rx: Receiver<Message>, response_tx: Sender<Response>) -> Self {
        Worker {
            task_rx,
            response_tx,
            processed_count: 0,
            start_time: Instant::now(),
        }
    }
    
    fn run(&mut self) {
        println!("Worker 线程开始运行");
        
        loop {
            match self.task_rx.recv() {
                Ok(Message::Work { id, data }) => {
                    let start = Instant::now();
                    
                    // 模拟复杂计算
                    let result: Vec<f32> = data
                        .iter()
                        .map(|x| {
                            // 模拟一些计算密集的操作
                            let mut sum = *x;
                            for i in 1..100 {
                                sum += (i as f32).sin().cos();
                            }
                            sum
                        })
                        .collect();
                    
                    let duration = start.elapsed();
                    self.processed_count += 1;
                    
                    let response = Response::WorkCompleted { id, result, duration };
                    
                    if self.response_tx.send(response).is_err() {
                        println!("无法发送响应，主线程可能已退出");
                        break;
                    }
                },
                Ok(Message::GetStatus) => {
                    let status = Response::Status {
                        processed_count: self.processed_count,
                        uptime: self.start_time.elapsed(),
                    };
                    let _ = self.response_tx.send(status);
                },
                Ok(Message::Shutdown) => {
                    println!("收到关闭信号");
                    let _ = self.response_tx.send(Response::ShutdownAck);
                    break;
                },
                Err(_) => {
                    println!("通道关闭，Worker 退出");
                    break;
                }
            }
        }
        
        println!("Worker 线程结束，共处理 {} 个任务", self.processed_count);
    }
}

fn main() {
    // 创建通道
    let (task_tx, task_rx) = channel::unbounded::<Message>();
    let (response_tx, response_rx) = channel::unbounded::<Response>();
    
    // 启动工作线程
    let worker_handle = {
        let task_rx = task_rx.clone();
        let response_tx = response_tx.clone();
        
        thread::spawn(move || {
            let mut worker = Worker::new(task_rx, response_tx);
            worker.run();
        })
    };
    
    // 发送工作任务
    println!("发送工作任务...");
    for i in 0..5 {
        let data = vec![i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32];
        let message = Message::Work { id: i, data };
        
        if task_tx.send(message).is_err() {
            println!("发送任务失败");
            break;
        }
        
        println!("发送任务 {}", i);
    }
    
    // 请求状态信息
    task_tx.send(Message::GetStatus).unwrap();
    
    // 接收响应
    let mut completed_tasks = 0;
    let total_tasks = 5;
    
    loop {
        match response_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Response::WorkCompleted { id, result, duration }) => {
                println!("任务 {} 完成: 结果前3个元素={:?}, 耗时={:?}", 
                    id, &result[..3.min(result.len())], duration);
                completed_tasks += 1;
                
                if completed_tasks >= total_tasks {
                    break;
                }
            },
            Ok(Response::Status { processed_count, uptime }) => {
                println!("状态: 已处理 {} 任务, 运行时间 {:?}", processed_count, uptime);
            },
            Ok(Response::ShutdownAck) => {
                println!("收到关闭确认");
                break;
            },
            Err(_) => {
                println!("接收响应超时");
                break;
            }
        }
    }
    
    // 发送关闭信号
    println!("发送关闭信号...");
    task_tx.send(Message::Shutdown).unwrap();
    
    // 等待确认
    match response_rx.recv_timeout(Duration::from_secs(2)) {
        Ok(Response::ShutdownAck) => println!("工作线程已确认关闭"),
        _ => println!("未收到关闭确认"),
    }
    
    // 等待线程结束
    worker_handle.join().unwrap();
    println!("主程序结束");
}

// 如果不想使用 crossbeam，可以用标准库的实现
mod std_implementation {
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;
    
    pub fn run_example() {
        let (tx, rx) = mpsc::channel();
        let (result_tx, result_rx) = mpsc::channel();
        
        // 工作线程
        thread::spawn(move || {
            loop {
                match rx.recv() {
                    Ok(data) => {
                        // 处理数据
                        let result = format!("处理结果: {}", data);
                        result_tx.send(result).unwrap();
                    },
                    Err(_) => break,
                }
            }
        });
        
        // 发送数据
        tx.send("测试数据".to_string()).unwrap();
        
        // 接收结果
        if let Ok(result) = result_rx.recv_timeout(Duration::from_secs(1)) {
            println!("{}", result);
        }
    }
}
