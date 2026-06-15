use std::fmt;

#[derive(Debug)]
pub enum SlotError {
    AllocatorUnavailable,
    SlotQueueEmpty,
    SlotNotFound,
}

impl fmt::Display for SlotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SlotError::AllocatorUnavailable => write!(f, "Slot allocator unavailable"),
            SlotError::SlotQueueEmpty => write!(f, "Slot queue empty while permit acquired"),
            SlotError::SlotNotFound => write!(f, "Slot not found"),
        }
    }
}

impl std::error::Error for SlotError {}

pub type SlotResult<T> = Result<T, SlotError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    /// 测试 SlotError 的 Display 实现
    /// 验证: 每个错误变体都能正确转换为字符串
    #[test]
    fn test_slot_error_display() {
        // 测试 AllocatorUnavailable
        let err = SlotError::AllocatorUnavailable;
        assert_eq!(
            err.to_string(),
            "Slot allocator unavailable",
            "AllocatorUnavailable should have correct display string"
        );

        // 测试 SlotQueueEmpty
        let err = SlotError::SlotQueueEmpty;
        assert_eq!(
            err.to_string(),
            "Slot queue empty while permit acquired",
            "SlotQueueEmpty should have correct display string"
        );

        // 测试 SlotNotFound
        let err = SlotError::SlotNotFound;
        assert_eq!(
            err.to_string(),
            "Slot not found",
            "SlotNotFound should have correct display string"
        );
    }

    /// 测试 SlotError 的 Debug 实现
    /// 验证: Debug 输出包含错误变体名称
    #[test]
    fn test_slot_error_debug() {
        let err = SlotError::AllocatorUnavailable;
        let debug_str = format!("{:?}", err);
        assert!(
            debug_str.contains("AllocatorUnavailable"),
            "Debug should contain variant name"
        );

        let err = SlotError::SlotQueueEmpty;
        let debug_str = format!("{:?}", err);
        assert!(
            debug_str.contains("SlotQueueEmpty"),
            "Debug should contain variant name"
        );

        let err = SlotError::SlotNotFound;
        let debug_str = format!("{:?}", err);
        assert!(
            debug_str.contains("SlotNotFound"),
            "Debug should contain variant name"
        );
    }

    /// 测试 SlotError 实现 std::error::Error trait
    /// 验证: 可以作为 trait object 使用
    #[test]
    fn test_slot_error_as_error_trait() {
        let err: Box<dyn Error> = Box::new(SlotError::AllocatorUnavailable);

        // 验证 Error trait 方法可用
        assert!(err.to_string().contains("Slot allocator unavailable"));
        assert!(
            err.source().is_none(),
            "SlotError should have no source error"
        );
    }

    /// 测试 SlotResult 类型别名
    /// 验证: SlotResult 正确工作
    #[test]
    fn test_slot_result_type() {
        // 测试 Ok 情况
        let ok_result: SlotResult<usize> = Ok(42);
        assert!(ok_result.is_ok(), "Should be Ok");
        assert_eq!(ok_result.unwrap(), 42, "Ok value should be 42");

        // 测试 Err 情况
        let err_result: SlotResult<usize> = Err(SlotError::SlotNotFound);
        assert!(err_result.is_err(), "Should be Err");
        match err_result {
            Err(SlotError::SlotNotFound) => (),
            _ => panic!("Expected SlotNotFound error"),
        }
    }

    /// 测试 SlotError 的部分匹配
    /// 验证: 错误类型可以正确区分
    #[test]
    fn test_slot_error_matching() {
        let errors = vec![
            SlotError::AllocatorUnavailable,
            SlotError::SlotQueueEmpty,
            SlotError::SlotNotFound,
        ];

        for err in errors {
            match err {
                SlotError::AllocatorUnavailable => assert!(true, "Matched AllocatorUnavailable"),
                SlotError::SlotQueueEmpty => assert!(true, "Matched SlotQueueEmpty"),
                SlotError::SlotNotFound => assert!(true, "Matched SlotNotFound"),
            }
        }
    }
}
