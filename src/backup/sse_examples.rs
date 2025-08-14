use async_stream::stream;
use axum::{
    extract::State,
    response::{sse::Event, IntoResponse, Sse},
    routing::get,
    Router,
};
use serde_json::json;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::time::sleep;

#[derive(Clone)]
struct AppState;

// 基本的 SSE 流示例
async fn basic_sse() -> impl IntoResponse {
    let stream = stream! {
        for i in 1..=5 {
            // 发送计数事件
            yield Ok::<_, std::convert::Infallible>(
                Event::default()
                    .event("count")  // 事件类型
                    .data(format!("计数: {}", i))  // 数据内容
                    .id(format!("msg-{}", i))  // 事件ID
            );

            sleep(Duration::from_secs(1)).await;
        }

        // 发送完成事件
        yield Ok(
            Event::default()
                .event("complete")
                .data("计数完成!")
        );
    };

    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(5))
            .text("keep-alive"),
    )
}

// 聊天流式响应示例
async fn chat_stream() -> impl IntoResponse {
    let stream = stream! {
        let message = "你好！我是AI助手，很高兴与你交流。";
        let chars: Vec<char> = message.chars().collect();

        // 发送开始事件
        yield Ok::<_, std::convert::Infallible>(
            Event::default()
                .event("start")
                .data(json!({
                    "type": "start",
                    "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
                }).to_string())
        );

        // 逐字符发送
        let mut accumulated = String::new();
        for (i, ch) in chars.iter().enumerate() {
            accumulated.push(*ch);

            yield Ok(
                Event::default()
                    .event("token")
                    .data(json!({
                        "type": "token",
                        "content": ch.to_string(),
                        "accumulated": accumulated,
                        "position": i
                    }).to_string())
                    .id(format!("token-{}", i))
            );

            sleep(Duration::from_millis(50)).await;
        }

        // 发送完成事件
        yield Ok(
            Event::default()
                .event("done")
                .data(json!({
                    "type": "done",
                    "final_text": accumulated,
                    "length": chars.len()
                }).to_string())
        );
    };

    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(10))
            .text("ping"),
    )
}

// 系统状态监控流
async fn system_monitor() -> impl IntoResponse {
    let stream = stream! {
        for i in 1..=10 {
            let cpu_usage = rand::random::<f32>() * 100.0;
            let memory_usage = rand::random::<f32>() * 100.0;

            yield Ok::<_, std::convert::Infallible>(
                Event::default()
                    .event("system_stats")
                    .data(json!({
                        "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                        "cpu_usage": format!("{:.1}%", cpu_usage),
                        "memory_usage": format!("{:.1}%", memory_usage),
                        "sequence": i
                    }).to_string())
                    .id(format!("stats-{}", i))
            );

            sleep(Duration::from_secs(2)).await;
        }

        // 发送监控结束事件
        yield Ok(
            Event::default()
                .event("monitoring_complete")
                .data("系统监控结束")
        );
    };

    Sse::new(stream)
}

// 错误处理示例
async fn error_prone_stream() -> impl IntoResponse {
    let stream = stream! {
        for i in 1..=5 {
            if i == 3 {
                // 模拟错误
                yield Ok::<_, std::convert::Infallible>(
                    Event::default()
                        .event("error")
                        .data(json!({
                            "error": "模拟的错误",
                            "code": 500,
                            "message": "处理第3项时发生错误"
                        }).to_string())
                        .retry(Duration::from_secs(3))  // 3秒后客户端自动重连
                );
                return; // 提前结束流
            }

            yield Ok(
                Event::default()
                    .event("progress")
                    .data(json!({
                        "step": i,
                        "status": "正常处理"
                    }).to_string())
            );

            sleep(Duration::from_millis(500)).await;
        }
    };

    Sse::new(stream)
}

// HTML 客户端测试页面
async fn index() -> axum::response::Html<&'static str> {
    axum::response::Html(
        r#"
<!DOCTYPE html>
<html>
<head>
    <title>SSE 测试</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; }
        .event-box { 
            border: 1px solid #ccc; 
            margin: 10px 0; 
            padding: 10px; 
            background: #f9f9f9; 
        }
        .error { background: #ffebee; border-color: #f44336; }
        .success { background: #e8f5e8; border-color: #4caf50; }
        button { 
            padding: 10px 20px; 
            margin: 5px; 
            background: #007bff; 
            color: white; 
            border: none; 
            cursor: pointer; 
        }
        button:hover { background: #0056b3; }
    </style>
</head>
<body>
    <div class="container">
        <h1>SSE (Server-Sent Events) 测试</h1>
        
        <div>
            <button onclick="startBasicSSE()">基本 SSE 流</button>
            <button onclick="startChatStream()">聊天流</button>
            <button onclick="startSystemMonitor()">系统监控</button>
            <button onclick="startErrorStream()">错误处理示例</button>
            <button onclick="clearEvents()">清除事件</button>
        </div>
        
        <div id="events"></div>
    </div>

    <script>
        let eventSources = [];
        
        function addEvent(type, data, className = '') {
            const eventsDiv = document.getElementById('events');
            const eventDiv = document.createElement('div');
            eventDiv.className = 'event-box ' + className;
            eventDiv.innerHTML = `
                <strong>事件类型:</strong> ${type}<br>
                <strong>时间:</strong> ${new Date().toLocaleString()}<br>
                <strong>数据:</strong> <pre>${data}</pre>
            `;
            eventsDiv.appendChild(eventDiv);
            eventsDiv.scrollTop = eventsDiv.scrollHeight;
        }
        
        function closeAllConnections() {
            eventSources.forEach(es => es.close());
            eventSources = [];
        }
        
        function startBasicSSE() {
            closeAllConnections();
            const eventSource = new EventSource('/basic-sse');
            eventSources.push(eventSource);
            
            eventSource.addEventListener('count', function(e) {
                addEvent('count', e.data, 'success');
            });
            
            eventSource.addEventListener('complete', function(e) {
                addEvent('complete', e.data, 'success');
            });
            
            eventSource.onerror = function(e) {
                addEvent('error', '连接错误', 'error');
            };
        }
        
        function startChatStream() {
            closeAllConnections();
            const eventSource = new EventSource('/chat-stream');
            eventSources.push(eventSource);
            
            eventSource.addEventListener('start', function(e) {
                addEvent('start', e.data, 'success');
            });
            
            eventSource.addEventListener('token', function(e) {
                addEvent('token', e.data);
            });
            
            eventSource.addEventListener('done', function(e) {
                addEvent('done', e.data, 'success');
            });
        }
        
        function startSystemMonitor() {
            closeAllConnections();
            const eventSource = new EventSource('/system-monitor');
            eventSources.push(eventSource);
            
            eventSource.addEventListener('system_stats', function(e) {
                addEvent('system_stats', e.data);
            });
            
            eventSource.addEventListener('monitoring_complete', function(e) {
                addEvent('monitoring_complete', e.data, 'success');
            });
        }
        
        function startErrorStream() {
            closeAllConnections();
            const eventSource = new EventSource('/error-stream');
            eventSources.push(eventSource);
            
            eventSource.addEventListener('progress', function(e) {
                addEvent('progress', e.data);
            });
            
            eventSource.addEventListener('error', function(e) {
                addEvent('error', e.data, 'error');
            });
            
            eventSource.onerror = function(e) {
                addEvent('connection_error', '连接发生错误，将自动重连', 'error');
            };
        }
        
        function clearEvents() {
            document.getElementById('events').innerHTML = '';
        }
    </script>
</body>
</html>
    "#,
    )
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/", get(index))
        .route("/basic-sse", get(basic_sse))
        .route("/chat-stream", get(chat_stream))
        .route("/system-monitor", get(system_monitor))
        .route("/error-stream", get(error_prone_stream))
        .with_state(AppState);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("SSE 示例服务器运行在 http://localhost:3000");
    println!("在浏览器中打开 http://localhost:3000 查看各种 SSE 流示例");

    axum::serve(listener, app).await.unwrap();
}
