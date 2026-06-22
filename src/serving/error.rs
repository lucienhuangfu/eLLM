use std::fmt;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

/// Serving 模块的统一错误类型
#[derive(Debug)]
pub enum ApiError {
    /// Slot 相关错误
    SlotError(crate::runtime::error::SlotError),
    /// Tokenization 失败
    TokenizationError(String),
    /// Slot 不可用
    SlotUnavailable(String),
    /// 内部服务器错误
    InternalError(String),
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApiError::SlotError(e) => write!(f, "Slot error: {}", e),
            ApiError::TokenizationError(msg) => write!(f, "Tokenization failed: {}", msg),
            ApiError::SlotUnavailable(msg) => write!(f, "Slot unavailable: {}", msg),
            ApiError::InternalError(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for ApiError {}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            ApiError::SlotError(e) => match e {
                crate::runtime::error::SlotError::AllocatorUnavailable => {
                    (StatusCode::INTERNAL_SERVER_ERROR, "Slot allocator unavailable".to_string())
                }
                crate::runtime::error::SlotError::SlotQueueEmpty => {
                    (StatusCode::INTERNAL_SERVER_ERROR, "No available slots".to_string())
                }
                crate::runtime::error::SlotError::SlotNotFound => {
                    (StatusCode::NOT_FOUND, "Slot not found".to_string())
                }
            },
            ApiError::TokenizationError(msg) => {
                eprintln!("Tokenization error: {}", msg);
                (StatusCode::INTERNAL_SERVER_ERROR, format!("Tokenization failed: {}", msg))
            }
            ApiError::SlotUnavailable(msg) => {
                eprintln!("Slot unavailable: {}", msg);
                (StatusCode::INTERNAL_SERVER_ERROR, format!("Slot unavailable: {}", msg))
            }
            ApiError::InternalError(msg) => {
                eprintln!("Internal error: {}", msg);
                (StatusCode::INTERNAL_SERVER_ERROR, format!("Operation failed: {}", msg))
            }
        };

        (status, message).into_response()
    }
}

impl From<crate::runtime::error::SlotError> for ApiError {
    fn from(err: crate::runtime::error::SlotError) -> Self {
        ApiError::SlotError(err)
    }
}

pub type ApiResult<T> = Result<T, ApiError>;