use axum::{Router, routing::post, Json, http::StatusCode, response::Sse, response::sse::ServerSentEvent, extract::State};
use futures::stream::StreamExt;
use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use async_stream::stream;
use rand::Rng;
// use ellm::llama::generation::generate_text;


// OpenAI API request/response models
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: Option<bool>,
    temperature: Option<f32>,
    max_tokens: Option<usize>,
    top_p: Option<f32>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatCompletionChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    index: u32,
    message: ChatMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct StreamResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Serialize)]
pub struct StreamChoice {
    index: u32,
    delta: ChatMessage,
    finish_reason: Option<String>,
}

#[derive(Clone)]
struct AppState {
    // Shared application state can be added here
}

// Fake text generation function for demonstration
use futures::stream::BoxStream;
async fn fake_generate_text(_model: &str, prompt: &str) -> Result<BoxStream<'static, String>, Box<dyn std::error::Error>> {
    use async_stream::stream;
    // Generate fake response stream
    let response = format!("Fake response to: {}", prompt);
    let stream = stream! {
        yield response;
    };
    Ok(stream.boxed())
}

// Format messages into a prompt string
fn format_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(&format!("<|{}|>\n{}\n", msg.role, msg.content));
    }
    prompt.push_str("<|assistant|>\n");
    prompt
}

// Handler for chat completions endpoint
async fn chat_completions(
    State(_state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>
) -> Result<impl axum::response::IntoResponse, StatusCode> {
    // Extract prompt from messages
    let prompt = format_prompt(&request.messages);
    let request_id = format!("chatcmpl-{}", rand::thread_rng().gen::<u64>());
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // Check if streaming is requested
    if let Some(true) = request.stream {
        // Create SSE response for streaming
        let sse_stream = stream! {
            let mut generated_text = String::new();
            let mut finish_reason: Option<String> = None;

            // Get LLM generation stream
            let mut generation_stream = match fake_generate_text(&request.model, &prompt).await {
                Ok(stream) => stream,
                Err(_) => {
                    yield Ok(ServerSentEvent::default()
                        .event("error")
                        .data("Failed to generate text")
                        .id("error")
                        .retry(Duration::from_secs(1)));
                    return;
                }
            };

            while let Some(chunk) = generation_stream.next().await {
                generated_text.push_str(&chunk);
                
                // Send SSE event with chunk
                yield Ok(ServerSentEvent::default()
                    .data(serde_json::to_string(&StreamResponse {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: request.model.clone(),
                        choices: vec![StreamChoice {
                            index: 0,
                            delta: ChatMessage {
                                role: "assistant".to_string(),
                                content: chunk,
                            },
                            finish_reason: None,
                        }],
                    }).unwrap()));
            }

            // Send final SSE event with finish reason
            yield Ok(ServerSentEvent::default()
                .data(serde_json::to_string(&StreamResponse {
                    id: request_id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: request.model.clone(),
                    choices: vec![StreamChoice {
                        index: 0,
                        delta: ChatMessage {
                            role: "assistant".to_string(),
                            content: "".to_string(),
                        },
                        finish_reason: Some("stop".to_string()),
                    }],
                }).unwrap()));
        };

        Ok(Sse::new(sse_stream).keep_alive(Some(ServerSentEvent::default().event("keep-alive").data("ping").retry(Duration::from_secs(10)))))
    } else {
        // Non-streaming response
        let generated_text = match fake_generate_text(&request.model, &prompt).await {
            Ok(mut stream) => {
                let mut result = String::new();
                while let Some(chunk) = stream.next().await {
                    result.push_str(&chunk);
                }
                result
            },
            Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
        };

        Ok((
            StatusCode::OK,
            Json(ChatCompletionResponse {
                id: request_id,
                object: "chat.completion".to_string(),
                created,
                model: request.model,
                choices: vec![ChatCompletionChoice {
                    index: 0,
                    message: ChatMessage {
                        role: "assistant".to_string(),
                        content: generated_text,
                    },
                    finish_reason: Some("stop".to_string()),
                }],
            })
        ))
    }
}

// Create and run the HTTP server
pub async fn run_server() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize application state
    let app_state = AppState {};

    // Define API routes
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(app_state);

    // Run the server on port 8000
    let addr = "0.0.0.0:8000".parse()?;
    println!("Server running on http://{}", addr);

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    run_server().await
}