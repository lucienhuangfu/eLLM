use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::fmt;

/// Serving 模块的统一错误类型
#[derive(Debug)]
pub enum ApiError {
    /// Slot 相关错误
    SlotError(crate::runtime::session::SlotError),
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
                crate::runtime::session::SlotError::AllocatorUnavailable => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Slot allocator unavailable".to_string(),
                ),
                crate::runtime::session::SlotError::SlotQueueEmpty => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "No available slots".to_string(),
                ),
                crate::runtime::session::SlotError::SlotNotFound => {
                    (StatusCode::NOT_FOUND, "Slot not found".to_string())
                }
            },
            ApiError::TokenizationError(msg) => {
                eprintln!("Tokenization error: {}", msg);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Tokenization failed: {}", msg),
                )
            }
            ApiError::SlotUnavailable(msg) => {
                eprintln!("Slot unavailable: {}", msg);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Slot unavailable: {}", msg),
                )
            }
            ApiError::InternalError(msg) => {
                eprintln!("Internal error: {}", msg);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Operation failed: {}", msg),
                )
            }
        };

        (status, message).into_response()
    }
}

impl From<crate::runtime::session::SlotError> for ApiError {
    fn from(err: crate::runtime::session::SlotError) -> Self {
        ApiError::SlotError(err)
    }
}

pub type ApiResult<T> = Result<T, ApiError>;

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;

    #[test]
    fn error_display_messages() {
        assert_eq!(
            format!("{}", ApiError::TokenizationError("oops".into())),
            "Tokenization failed: oops"
        );
        assert_eq!(
            format!("{}", ApiError::SlotUnavailable("busy".into())),
            "Slot unavailable: busy"
        );
        assert_eq!(
            format!("{}", ApiError::InternalError("fail".into())),
            "Internal error: fail"
        );
    }

    #[test]
    fn tokenization_error_status_500() {
        let err = ApiError::TokenizationError("bad token".into());
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn slot_error_from_conversion() {
        let slot_err = crate::runtime::session::SlotError::SlotNotFound;
        let api_err: ApiError = slot_err.into();
        let resp = api_err.into_response();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }
}
