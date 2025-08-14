use async_stream::stream;
use futures::stream::{Stream, StreamExt};
use std::time::Duration;
use tokio::time::sleep;

// 1. 基本用法 - 简单的数值流
fn simple_number_stream() -> impl Stream<Item = i32> {
    stream! {
        for i in 0..5 {
            yield i;
        }
    }
}

// 2. 异步操作 - 带延迟的流
fn delayed_stream() -> impl Stream<Item = String> {
    stream! {
        for i in 0..3 {
            sleep(Duration::from_millis(500)).await;
            yield format!("Item {}", i);
        }
    }
}

// 3. 错误处理 - 可能产生错误的流
fn fallible_stream() -> impl Stream<Item = Result<String, &'static str>> {
    stream! {
        for i in 0..5 {
            if i == 3 {
                yield Err("Something went wrong at item 3");
            } else {
                yield Ok(format!("Success: {}", i));
            }
        }
    }
}

// 4. 条件控制 - 根据条件提前返回
fn conditional_stream(max_items: usize) -> impl Stream<Item = String> {
    stream! {
        for i in 0..10 {
            if i >= max_items {
                return; // 提前结束流
            }
            yield format!("Item {}", i);
        }
    }
}

// 5. 组合其他流 - 处理来自其他流的数据
fn processing_stream(input: impl Stream<Item = i32>) -> impl Stream<Item = String> {
    stream! {
        let mut input = std::pin::Pin::new(Box::new(input));
        while let Some(value) = input.next().await {
            // 处理每个输入值
            let processed = value * 2;
            yield format!("Processed: {} -> {}", value, processed);
        }
    }
}

// 6. 聊天机器人响应流 - 模拟分词输出
fn chat_response_stream(message: String) -> impl Stream<Item = String> {
    stream! {
        let response = format!("You said: '{}'. Here's my response: ", message);
        let words = response.split(' ').collect::<Vec<_>>();

        for word in words {
            sleep(Duration::from_millis(100)).await;
            yield format!("{} ", word);
        }

        // 添加一些动态内容
        for i in 1..=3 {
            sleep(Duration::from_millis(200)).await;
            yield format!("Point {}. ", i);
        }

        yield "Done!".to_string();
    }
}

// 7. 文件读取流 - 逐行读取
fn file_lines_stream(content: String) -> impl Stream<Item = Result<String, &'static str>> {
    stream! {
        let lines = content.lines();
        for (i, line) in lines.enumerate() {
            sleep(Duration::from_millis(50)).await;

            if line.is_empty() {
                yield Err("Empty line encountered");
            } else {
                yield Ok(format!("Line {}: {}", i + 1, line));
            }
        }
    }
}

// 8. 服务器发送事件 (SSE) 流
fn sse_events_stream() -> impl Stream<Item = String> {
    stream! {
        // 发送初始连接事件
        yield "event: connected\ndata: Connection established\n\n".to_string();

        // 发送周期性心跳
        for i in 1..=5 {
            sleep(Duration::from_secs(1)).await;
            yield format!("event: heartbeat\ndata: Heartbeat {}\n\n", i);
        }

        // 发送数据事件
        for i in 1..=3 {
            sleep(Duration::from_millis(500)).await;
            yield format!("event: data\ndata: {{\"message\": \"Data chunk {}\"}}\n\n", i);
        }

        // 发送关闭事件
        yield "event: close\ndata: Connection closing\n\n".to_string();
    }
}

#[tokio::main]
async fn main() {
    println!("=== async_stream 使用示例 ===\n");

    // 示例 1: 基本数值流
    println!("1. 基本数值流:");
    let mut stream = simple_number_stream();
    while let Some(value) = stream.next().await {
        println!("  {}", value);
    }

    println!("\n2. 带延迟的流:");
    let mut stream = delayed_stream();
    while let Some(value) = stream.next().await {
        println!("  {}", value);
    }

    println!("\n3. 错误处理流:");
    let mut stream = fallible_stream();
    while let Some(result) = stream.next().await {
        match result {
            Ok(value) => println!("  ✓ {}", value),
            Err(error) => println!("  ✗ Error: {}", error),
        }
    }

    println!("\n4. 条件控制流 (最多3个item):");
    let mut stream = conditional_stream(3);
    while let Some(value) = stream.next().await {
        println!("  {}", value);
    }

    println!("\n5. 处理其他流:");
    let input_stream = simple_number_stream();
    let mut processing = processing_stream(input_stream);
    while let Some(value) = processing.next().await {
        println!("  {}", value);
    }

    println!("\n6. 聊天响应流:");
    let mut chat_stream = chat_response_stream("Hello AI!".to_string());
    print!("  ");
    while let Some(chunk) = chat_stream.next().await {
        print!("{}", chunk);
        tokio::task::yield_now().await; // 让其他任务有机会运行
    }
    println!();

    println!("\n7. 文件行流:");
    let content = "Line 1\nLine 2\n\nLine 4".to_string();
    let mut file_stream = file_lines_stream(content);
    while let Some(result) = file_stream.next().await {
        match result {
            Ok(line) => println!("  ✓ {}", line),
            Err(error) => println!("  ✗ Error: {}", error),
        }
    }

    println!("\n8. SSE 事件流:");
    let mut sse_stream = sse_events_stream();
    while let Some(event) = sse_stream.next().await {
        print!("  {}", event);
    }
}
