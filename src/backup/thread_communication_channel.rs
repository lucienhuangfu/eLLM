use std::sync::mpsc;
use std::thread;
use std::time::Duration;

// 定义任务类型
#[derive(Debug)]
enum Task {
    Process { data: Vec<i32>, id: u64 },
    Stop,
}

// 定义结果类型
#[derive(Debug)]
struct TaskResult {
    id: u64,
    result: Vec<i32>,
    processing_time: Duration,
}

fn main() {
    // 创建发送任务的通道
    let (task_sender, task_receiver) = mpsc::channel::<Task>();

    // 创建接收结果的通道
    let (result_sender, result_receiver) = mpsc::channel::<TaskResult>();

    // 启动工作线程
    let worker_handle = thread::spawn(move || {
        println!("工作线程启动");

        // 死循环等待任务
        loop {
            match task_receiver.recv() {
                Ok(Task::Process { data, id }) => {
                    let start = std::time::Instant::now();

                    // 模拟处理数据
                    println!("处理任务 ID: {}, 数据: {:?}", id, data);
                    thread::sleep(Duration::from_millis(100)); // 模拟耗时操作

                    // 处理数据 (这里简单地将每个数字乘以2)
                    let processed: Vec<i32> = data.iter().map(|x| x * 2).collect();

                    let processing_time = start.elapsed();

                    // 发送结果回主线程
                    let result = TaskResult {
                        id,
                        result: processed,
                        processing_time,
                    };

                    if result_sender.send(result).is_err() {
                        println!("结果发送失败，主线程可能已退出");
                        break;
                    }
                }
                Ok(Task::Stop) => {
                    println!("收到停止信号，工作线程退出");
                    break;
                }
                Err(_) => {
                    println!("通道关闭，工作线程退出");
                    break;
                }
            }
        }

        println!("工作线程结束");
    });

    // 主线程发送任务并接收结果
    println!("开始发送任务...");

    // 发送几个任务
    for i in 0..5 {
        let data = vec![i, i + 1, i + 2];
        let task = Task::Process { data, id: i as u64 };

        if task_sender.send(task).is_err() {
            println!("任务发送失败");
            break;
        }

        println!("发送任务 {}", i);
    }

    // 接收结果
    for _ in 0..5 {
        match result_receiver.recv_timeout(Duration::from_secs(5)) {
            Ok(result) => {
                println!(
                    "收到结果: ID={}, result={:?}, 耗时={:?}",
                    result.id, result.result, result.processing_time
                );
            }
            Err(_) => {
                println!("接收结果超时");
                break;
            }
        }
    }

    // 发送停止信号
    task_sender.send(Task::Stop).unwrap();

    // 等待工作线程结束
    worker_handle.join().unwrap();

    println!("程序结束");
}
