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

    /// 测试 SlotError 所有变体
    #[test]
    fn test_slot_error_all_variants() {
        let variants = [
            SlotError::AllocatorUnavailable,
            SlotError::SlotQueueEmpty,
            SlotError::SlotNotFound,
        ];

        // 验证所有变体都不同
        for (i, v1) in variants.iter().enumerate() {
            for (j, v2) in variants.iter().enumerate() {
                if i != j {
                    assert_ne!(
                        format!("{}", v1),
                        format!("{}", v2),
                        "Different variants should have different display strings"
                    );
                }
            }
        }
    }

    /// 测试 SlotError 在函数返回值中使用
    #[test]
    fn test_slot_error_in_function_return() {
        fn may_fail(should_fail: bool) -> SlotResult<String> {
            if should_fail {
                Err(SlotError::AllocatorUnavailable)
            } else {
                Ok("success".to_string())
            }
        }

        let ok = may_fail(false);
        assert!(ok.is_ok());
        assert_eq!(ok.unwrap(), "success");

        let err = may_fail(true);
        assert!(err.is_err());
        assert!(matches!(err, Err(SlotError::AllocatorUnavailable)));
    }

    /// 测试 SlotError 错误链
    #[test]
    fn test_slot_error_chain() {
        fn outer_function() -> SlotResult<i32> {
            inner_function().map(|x| x * 2)
        }

        fn inner_function() -> SlotResult<i32> {
            Err(SlotError::SlotNotFound)
        }

        let result = outer_function();
        assert!(result.is_err());
        assert!(matches!(result, Err(SlotError::SlotNotFound)));
    }

    /// 测试 SlotError 与 ? 操作符
    #[test]
    fn test_slot_error_with_try_operator() {
        fn try_operation() -> SlotResult<i32> {
            let value = get_value()?;
            Ok(value + 1)
        }

        fn get_value() -> SlotResult<i32> {
            Ok(10)
        }

        assert_eq!(try_operation().unwrap(), 11);
    }

    /// 测试 SlotError 与 ? 操作符失败情况
    #[test]
    fn test_slot_error_with_try_operator_failure() {
        fn try_operation() -> SlotResult<i32> {
            let _value = get_value()?;
            Ok(0)
        }

        fn get_value() -> SlotResult<i32> {
            Err(SlotError::SlotQueueEmpty)
        }

        assert!(matches!(try_operation(), Err(SlotError::SlotQueueEmpty)));
    }

    /// 测试 SlotError 作为集合元素
    #[test]
    fn test_slot_error_in_collection() {
        let errors: Vec<SlotError> = vec![
            SlotError::AllocatorUnavailable,
            SlotError::SlotQueueEmpty,
            SlotError::SlotNotFound,
        ];

        assert_eq!(errors.len(), 3);

        // 验证可以迭代
        let count = errors.iter().count();
        assert_eq!(count, 3);
    }

    /// 测试 SlotError 格式化输出
    #[test]
    fn test_slot_error_formatting() {
        let err = SlotError::SlotNotFound;

        // 使用 format! 格式化
        let msg = format!("Error occurred: {}", err);
        assert!(msg.contains("Slot not found"));

        // 使用 println! 格式化（不实际打印）
        let _ = format!("Error: {:?}", err);
    }

    /// 测试 SlotError 多态使用
    #[test]
    fn test_slot_error_polymorphism() {
        fn handle_error(err: &dyn Error) -> String {
            err.to_string()
        }

        let err = SlotError::AllocatorUnavailable;
        let msg = handle_error(&err);
        assert!(msg.contains("Slot allocator unavailable"));
    }

    /// 测试 SlotError 结构体大小
    #[test]
    fn test_slot_error_size() {
        let size = std::mem::size_of::<SlotError>();
        // 枚举大小应该很小（没有数据字段）
        assert!(size <= 1);
    }

    /// 测试 SlotResult 大小
    #[test]
    fn test_slot_result_size() {
        let size_ok = std::mem::size_of::<SlotResult<i32>>();
        let size_err = std::mem::size_of::<SlotResult<()>>();

        // Result 大小应该合理
        assert!(size_ok > 0);
        assert!(size_err > 0);
    }

    /// 测试 SlotError 错误消息长度
    #[test]
    fn test_slot_error_message_length() {
        assert!(SlotError::AllocatorUnavailable.to_string().len() > 0);
        assert!(SlotError::SlotQueueEmpty.to_string().len() > 0);
        assert!(SlotError::SlotNotFound.to_string().len() > 0);
    }

    /// 测试 SlotError 错误消息内容验证
    #[test]
    fn test_slot_error_message_content() {
        // AllocatorUnavailable 应该包含 "allocator"
        assert!(SlotError::AllocatorUnavailable
            .to_string()
            .contains("allocator"));

        // SlotQueueEmpty 应该包含 "queue"
        assert!(SlotError::SlotQueueEmpty.to_string().contains("queue"));

        // SlotNotFound 应该包含 "not found"
        assert!(SlotError::SlotNotFound.to_string().contains("not found"));
    }

    /// 测试 SlotError Debug 与 Display 不同
    #[test]
    fn test_slot_error_debug_vs_display() {
        let err = SlotError::AllocatorUnavailable;

        let debug = format!("{:?}", err);
        let display = format!("{}", err);

        // Debug 应该包含枚举名称，Display 应该是用户友好的消息
        assert!(debug.contains("AllocatorUnavailable"));
        assert!(!display.contains("AllocatorUnavailable")); // Display 不应该包含枚举名称
    }

    /// 测试 SlotError 多次转换
    #[test]
    fn test_slot_error_multiple_conversions() {
        let err = SlotError::SlotNotFound;

        // 多次调用 to_string 应该返回相同结果
        let s1 = err.to_string();
        let s2 = err.to_string();
        let s3 = err.to_string();

        assert_eq!(s1, s2);
        assert_eq!(s2, s3);
    }

    /// 测试 SlotError 作为参数传递
    #[test]
    fn test_slot_error_as_parameter() {
        fn check_error_type(err: SlotError) -> &'static str {
            match err {
                SlotError::AllocatorUnavailable => "allocator",
                SlotError::SlotQueueEmpty => "queue",
                SlotError::SlotNotFound => "not_found",
            }
        }

        assert_eq!(
            check_error_type(SlotError::AllocatorUnavailable),
            "allocator"
        );
        assert_eq!(check_error_type(SlotError::SlotQueueEmpty), "queue");
        assert_eq!(check_error_type(SlotError::SlotNotFound), "not_found");
    }

    /// 测试 SlotError 返回值
    #[test]
    fn test_slot_error_return_value() {
        fn get_error(type_id: u8) -> Option<SlotError> {
            match type_id {
                0 => Some(SlotError::AllocatorUnavailable),
                1 => Some(SlotError::SlotQueueEmpty),
                2 => Some(SlotError::SlotNotFound),
                _ => None,
            }
        }

        assert!(matches!(
            get_error(0),
            Some(SlotError::AllocatorUnavailable)
        ));
        assert!(matches!(get_error(1), Some(SlotError::SlotQueueEmpty)));
        assert!(matches!(get_error(2), Some(SlotError::SlotNotFound)));
        assert!(get_error(3).is_none());
    }

    /// 测试 SlotError 与 Option 组合
    #[test]
    fn test_slot_error_with_option() {
        fn may_return_error() -> Option<SlotError> {
            Some(SlotError::SlotNotFound)
        }

        let err = may_return_error();
        assert!(err.is_some());
        assert!(matches!(err, Some(SlotError::SlotNotFound)));
    }

    /// 测试 SlotError 转换为 Box<dyn Error>
    #[test]
    fn test_slot_error_boxed() {
        let err: Box<dyn Error> = Box::new(SlotError::SlotQueueEmpty);

        // 验证可以调用 Error trait 方法
        assert!(err.to_string().contains("queue"));

        // 验证 source() 返回 None
        assert!(err.source().is_none());
    }

    /// 测试 SlotError 在 Result::Err 中使用
    #[test]
    fn test_slot_error_in_result_err() {
        let result: Result<(), SlotError> = Err(SlotError::AllocatorUnavailable);

        // 使用 unwrap_err 获取错误
        let err = result.unwrap_err();
        assert!(matches!(err, SlotError::AllocatorUnavailable));
    }

    /// 测试 SlotError 与 map_err
    #[test]
    fn test_slot_error_map_err() {
        let result: Result<i32, SlotError> = Err(SlotError::SlotNotFound);

        // map_err 应该可以转换错误
        let mapped = result.map_err(|e| format!("Wrapped: {}", e));
        assert!(mapped.is_err());
        assert!(mapped.unwrap_err().contains("Wrapped"));
    }

    /// 测试 SlotError 与 or_else
    #[test]
    fn test_slot_error_or_else() {
        let result: Result<i32, SlotError> = Err(SlotError::SlotNotFound);

        // or_else 应该可以处理错误
        let handled: Result<i32, SlotError> = result.or_else(|_| Ok(0));
        assert!(handled.is_ok());
        assert_eq!(handled.unwrap(), 0);
    }

    /// 测试 SlotError 与 unwrap_or
    #[test]
    fn test_slot_error_unwrap_or() {
        let result: SlotResult<i32> = Err(SlotError::SlotQueueEmpty);

        // unwrap_or 应该返回默认值
        let value = result.unwrap_or(100);
        assert_eq!(value, 100);
    }

    /// 测试 SlotError 与 unwrap_or_else
    #[test]
    fn test_slot_error_unwrap_or_else() {
        let result: SlotResult<i32> = Err(SlotError::SlotNotFound);

        // unwrap_or_else 应该调用函数
        let value = result.unwrap_or_else(|_| 200);
        assert_eq!(value, 200);
    }

    /// 测试 SlotError 与 expect
    #[test]
    #[should_panic(expected = "Expected success")]
    fn test_slot_error_expect_panic() {
        let result: SlotResult<i32> = Err(SlotError::AllocatorUnavailable);
        result.expect("Expected success");
    }

    /// 测试 SlotError 与 unwrap 失败
    #[test]
    #[should_panic]
    fn test_slot_error_unwrap_panic() {
        let result: SlotResult<i32> = Err(SlotError::SlotQueueEmpty);
        result.unwrap();
    }
}
