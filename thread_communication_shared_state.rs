use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

#[derive(Debug, Clone)]
struct WorkItem {
    id: u64,
    data: Vec<i32>,
}

#[derive(Debug, Clone)]
struct WorkResult {
    id: u64,
    result: Vec<i32>,
}

// 共享的工作队列
struct WorkQueue {
    tasks: Mutex<VecDeque<WorkItem>>,
    results: Mutex<VecDeque<WorkResult>>,
    task_condvar: Condvar,
    result_condvar: Condvar,
    should_stop: Mutex<bool>,
}

impl WorkQueue {
    fn new() -> Self {
        WorkQueue {
            tasks: Mutex::new(VecDeque::new()),
            results: Mutex::new(VecDeque::new()),
            task_condvar: Condvar::new(),
            result_condvar: Condvar::new(),
            should_stop: Mutex::new(false),
        }
    }

    fn add_task(&self, item: WorkItem) {
        let mut tasks = self.tasks.lock().unwrap();
        tasks.push_back(item);
        self.task_condvar.notify_one();
    }

    fn get_task(&self) -> Option<WorkItem> {
        let mut tasks = self.tasks.lock().unwrap();

        while tasks.is_empty() {
            let should_stop = *self.should_stop.lock().unwrap();
            if should_stop {
                return None;
            }

            tasks = self.task_condvar.wait(tasks).unwrap();
        }

        tasks.pop_front()
    }

    fn add_result(&self, result: WorkResult) {
        let mut results = self.results.lock().unwrap();
        results.push_back(result);
        self.result_condvar.notify_one();
    }

    fn get_result(&self) -> Option<WorkResult> {
        let mut results = self.results.lock().unwrap();

        while results.is_empty() {
            results = self
                .result_condvar
                .wait_timeout(results, Duration::from_secs(1))
                .unwrap()
                .0;
            if results.is_empty() {
                return None; // 超时
            }
        }

        results.pop_front()
    }

    fn stop(&self) {
        *self.should_stop.lock().unwrap() = true;
        self.task_condvar.notify_all();
    }
}

fn main() {
    let work_queue = Arc::new(WorkQueue::new());
    let work_queue_clone = Arc::clone(&work_queue);

    // 启动工作线程
    let worker_handle = thread::spawn(move || {
        println!("工作线程启动");

        loop {
            match work_queue_clone.get_task() {
                Some(item) => {
                    println!("处理任务 ID: {}, 数据: {:?}", item.id, item.data);

                    // 模拟处理
                    thread::sleep(Duration::from_millis(200));
                    let processed: Vec<i32> = item.data.iter().map(|x| x * 3).collect();

                    let result = WorkResult {
                        id: item.id,
                        result: processed,
                    };

                    work_queue_clone.add_result(result);
                }
                None => {
                    println!("工作线程退出");
                    break;
                }
            }
        }
    });

    // 主线程发送任务
    println!("发送任务到工作队列...");
    for i in 0..3 {
        let item = WorkItem {
            id: i,
            data: vec![i as i32 * 10, (i as i32 * 10) + 1, (i as i32 * 10) + 2],
        };
        work_queue.add_task(item);
        println!("已添加任务 {}", i);
    }

    // 接收结果
    println!("等待结果...");
    for _ in 0..3 {
        if let Some(result) = work_queue.get_result() {
            println!("收到结果: ID={}, result={:?}", result.id, result.result);
        } else {
            println!("获取结果超时");
        }
    }

    // 停止工作线程
    work_queue.stop();
    worker_handle.join().unwrap();

    println!("程序结束");
}
