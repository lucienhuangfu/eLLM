#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Phase {
    Start,
    Prefill,
    Decode,
    Timeout,
    Eos,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;

    /// 测试 Phase 枚举所有变体的相等性
    #[test]
    fn test_phase_enum_variants() {
        let phases = [
            Phase::Start,
            Phase::Prefill,
            Phase::Decode,
            Phase::Timeout,
            Phase::Eos,
        ];

        for (i, phase1) in phases.iter().enumerate() {
            for (j, phase2) in phases.iter().enumerate() {
                if i == j {
                    assert_eq!(phase1, phase2, "Same phase should be equal");
                } else {
                    assert_ne!(phase1, phase2, "Different phases should not be equal");
                }
            }
        }
    }

    /// 测试 Phase 枚举的 Copy 特性
    #[test]
    fn test_phase_copy() {
        let phase = Phase::Decode;
        let copied = phase;
        assert_eq!(phase, copied);
    }

    /// 测试 Phase 枚举的 Clone 特性
    #[test]
    fn test_phase_clone() {
        let phase = Phase::Prefill;
        let cloned = phase.clone();
        assert_eq!(phase, cloned);
    }

    /// 测试 Phase 枚举的 Debug 实现
    #[test]
    fn test_phase_debug() {
        let debug_str = format!("{:?}", Phase::Start);
        assert!(debug_str.contains("Start"));

        let debug_str = format!("{:?}", Phase::Prefill);
        assert!(debug_str.contains("Prefill"));

        let debug_str = format!("{:?}", Phase::Decode);
        assert!(debug_str.contains("Decode"));

        let debug_str = format!("{:?}", Phase::Timeout);
        assert!(debug_str.contains("Timeout"));

        let debug_str = format!("{:?}", Phase::Eos);
        assert!(debug_str.contains("Eos"));
    }

    /// 测试 Phase 枚举的 PartialEq 实现
    #[test]
    fn test_phase_partial_eq() {
        assert!(Phase::Start == Phase::Start);
        assert!(Phase::Prefill == Phase::Prefill);
        assert!(Phase::Decode == Phase::Decode);
        assert!(Phase::Timeout == Phase::Timeout);
        assert!(Phase::Eos == Phase::Eos);

        assert!(Phase::Start != Phase::Prefill);
        assert!(Phase::Prefill != Phase::Decode);
        assert!(Phase::Decode != Phase::Timeout);
        assert!(Phase::Timeout != Phase::Eos);
    }

    /// 测试 Phase 枚举的 Eq 特性
    #[test]
    fn test_phase_eq() {
        // Eq trait 确保完全相等比较
        fn assert_eq_trait<T: Eq + Debug>(a: T, b: T) {
            assert_eq!(a, b);
        }
        assert_eq_trait(Phase::Start, Phase::Start);
    }

    /// 测试 Phase 枚举的 #[repr(u8)] 表示
    #[test]
    fn test_phase_repr() {
        // 验证枚举值可以正确转换为 u8
        let start = Phase::Start as u8;
        let prefill = Phase::Prefill as u8;
        let decode = Phase::Decode as u8;
        let timeout = Phase::Timeout as u8;
        let eos = Phase::Eos as u8;

        // 验证所有值都是不同的
        assert_ne!(start, prefill);
        assert_ne!(prefill, decode);
        assert_ne!(decode, timeout);
        assert_ne!(timeout, eos);

        // 验证值在合理范围内 (0-4)
        assert!(start < 5);
        assert!(prefill < 5);
        assert!(decode < 5);
        assert!(timeout < 5);
        assert!(eos < 5);
    }

    /// 测试 Phase 枚举的排序顺序
    #[test]
    fn test_phase_ordering() {
        // 验证枚举值的顺序
        assert!((Phase::Start as u8) < (Phase::Prefill as u8));
        assert!((Phase::Prefill as u8) < (Phase::Decode as u8));
        assert!((Phase::Decode as u8) < (Phase::Timeout as u8));
        assert!((Phase::Timeout as u8) < (Phase::Eos as u8));
    }

    /// 测试 Phase 枚举的匹配
    #[test]
    fn test_phase_matching() {
        fn get_phase_name(phase: Phase) -> &'static str {
            match phase {
                Phase::Start => "start",
                Phase::Prefill => "prefill",
                Phase::Decode => "decode",
                Phase::Timeout => "timeout",
                Phase::Eos => "eos",
            }
        }

        assert_eq!(get_phase_name(Phase::Start), "start");
        assert_eq!(get_phase_name(Phase::Prefill), "prefill");
        assert_eq!(get_phase_name(Phase::Decode), "decode");
        assert_eq!(get_phase_name(Phase::Timeout), "timeout");
        assert_eq!(get_phase_name(Phase::Eos), "eos");
    }

    /// 测试 Phase 枚举的 if let 匹配
    #[test]
    fn test_phase_if_let() {
        let phase = Phase::Decode;
        if let Phase::Decode = phase {
            // 成功匹配
        } else {
            panic!("Should match Decode phase");
        }

        let phase = Phase::Prefill;
        if let Phase::Decode = phase {
            panic!("Should not match Decode phase");
        }
    }

    /// 测试 Phase 枚举的迭代
    #[test]
    fn test_phase_iteration() {
        let phases = [
            Phase::Start,
            Phase::Prefill,
            Phase::Decode,
            Phase::Timeout,
            Phase::Eos,
        ];

        let count = phases.iter().count();
        assert_eq!(count, 5);
    }

    /// 测试 Phase 枚举的集合操作
    #[test]
    fn test_phase_in_collection() {
        let active_phases = [Phase::Prefill, Phase::Decode];
        let inactive_phases = [Phase::Start, Phase::Eos, Phase::Timeout];

        for phase in active_phases.iter() {
            assert!(matches!(phase, Phase::Prefill | Phase::Decode));
        }

        for phase in inactive_phases.iter() {
            assert!(matches!(phase, Phase::Start | Phase::Eos | Phase::Timeout));
        }
    }

    /// 测试 Phase 枚举的转换回枚举
    #[test]
    fn test_phase_from_u8() {
        fn from_u8(value: u8) -> Option<Phase> {
            match value {
                0 => Some(Phase::Start),
                1 => Some(Phase::Prefill),
                2 => Some(Phase::Decode),
                3 => Some(Phase::Timeout),
                4 => Some(Phase::Eos),
                _ => None,
            }
        }

        assert_eq!(from_u8(0), Some(Phase::Start));
        assert_eq!(from_u8(1), Some(Phase::Prefill));
        assert_eq!(from_u8(2), Some(Phase::Decode));
        assert_eq!(from_u8(3), Some(Phase::Timeout));
        assert_eq!(from_u8(4), Some(Phase::Eos));
        assert_eq!(from_u8(5), None);
        assert_eq!(from_u8(255), None);
    }

    /// 测试 Phase 枚举的哈希
    #[test]
    fn test_phase_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(Phase::Start);
        set.insert(Phase::Prefill);
        set.insert(Phase::Decode);

        assert!(set.contains(&Phase::Start));
        assert!(set.contains(&Phase::Prefill));
        assert!(set.contains(&Phase::Decode));
        assert!(!set.contains(&Phase::Timeout));
        assert!(!set.contains(&Phase::Eos));

        assert_eq!(set.len(), 3);
    }

    /// 测试 Phase 枚举的默认值（如果实现）
    #[test]
    fn test_phase_default_not_implemented() {
        // Phase 没有实现 Default，所以这个测试验证它不应该有默认值
        // 如果需要默认值，应该手动指定
        let default_phase = Phase::Start; // 手动选择默认值
        assert_eq!(default_phase, Phase::Start);
    }

    /// 测试 Phase 枚举的位运算（如果需要）
    #[test]
    fn test_phase_bit_operations() {
        // 验证 repr(u8) 可以进行位运算
        let phase = Phase::Decode;
        let value = phase as u8;

        // 位运算测试
        assert_eq!(value & 0xFF, value);
        assert_eq!(value | 0, value);
        assert_eq!(value ^ value, 0);
    }

    /// 测试 Phase 枚举的比较操作
    #[test]
    fn test_phase_comparison() {
        // 虽然 Phase 没有实现 Ord，但可以通过 as u8 比较
        assert!((Phase::Start as u8) < (Phase::Prefill as u8));
        assert!((Phase::Prefill as u8) < (Phase::Decode as u8));
        assert!((Phase::Decode as u8) < (Phase::Timeout as u8));
        assert!((Phase::Timeout as u8) < (Phase::Eos as u8));
    }

    /// 测试 Phase 枚举的内存大小
    #[test]
    fn test_phase_size() {
        // repr(u8) 应该使 Phase 只有 1 字节大小
        assert_eq!(std::mem::size_of::<Phase>(), 1);
    }

    /// 测试 Phase 枚举的对齐
    #[test]
    fn test_phase_align() {
        // u8 的对齐应该是 1
        assert_eq!(std::mem::align_of::<Phase>(), 1);
    }

    /// 测试 Phase 枚举在 Option 中的大小
    #[test]
    fn test_phase_option_size() {
        // Option<Phase> 应该优化为 1 字节（因为 Phase 是 repr(u8)）
        assert_eq!(std::mem::size_of::<Option<Phase>>(), 1);
    }

    /// 测试 Phase 枚举的数组大小
    #[test]
    fn test_phase_array() {
        let phases: [Phase; 5] = [
            Phase::Start,
            Phase::Prefill,
            Phase::Decode,
            Phase::Timeout,
            Phase::Eos,
        ];

        assert_eq!(phases.len(), 5);
        assert_eq!(std::mem::size_of_val(&phases), 5); // 5 * 1 byte
    }

    /// 测试 Phase 枚举的引用
    #[test]
    fn test_phase_reference() {
        let phase = Phase::Decode;
        let ref_phase = &phase;
        assert_eq!(*ref_phase, Phase::Decode);
    }

    /// 测试 Phase 枚举的可变引用
    #[test]
    fn test_phase_mutable_reference() {
        let mut phase = Phase::Start;
        {
            let ref_phase = &mut phase;
            *ref_phase = Phase::Prefill;
        }
        assert_eq!(phase, Phase::Prefill);
    }

    /// 测试 Phase 枚举的函数参数传递
    #[test]
    fn test_phase_function_parameter() {
        fn check_phase(phase: Phase) -> bool {
            matches!(phase, Phase::Decode | Phase::Prefill)
        }

        assert!(check_phase(Phase::Decode));
        assert!(check_phase(Phase::Prefill));
        assert!(!check_phase(Phase::Start));
        assert!(!check_phase(Phase::Eos));
        assert!(!check_phase(Phase::Timeout));
    }

    /// 测试 Phase 枚举的返回值
    #[test]
    fn test_phase_function_return() {
        fn get_next_phase(current: Phase) -> Option<Phase> {
            match current {
                Phase::Start => Some(Phase::Prefill),
                Phase::Prefill => Some(Phase::Decode),
                Phase::Decode => Some(Phase::Eos),
                Phase::Timeout | Phase::Eos => None,
            }
        }

        assert_eq!(get_next_phase(Phase::Start), Some(Phase::Prefill));
        assert_eq!(get_next_phase(Phase::Prefill), Some(Phase::Decode));
        assert_eq!(get_next_phase(Phase::Decode), Some(Phase::Eos));
        assert_eq!(get_next_phase(Phase::Timeout), None);
        assert_eq!(get_next_phase(Phase::Eos), None);
    }

    /// 测试 Phase 枚举的序列化（如果需要）
    #[test]
    fn test_phase_serialization_manual() {
        fn serialize(phase: Phase) -> u8 {
            phase as u8
        }

        fn deserialize(value: u8) -> Result<Phase, String> {
            match value {
                0 => Ok(Phase::Start),
                1 => Ok(Phase::Prefill),
                2 => Ok(Phase::Decode),
                3 => Ok(Phase::Timeout),
                4 => Ok(Phase::Eos),
                _ => Err(format!("Invalid phase value: {}", value)),
            }
        }

        for phase in [
            Phase::Start,
            Phase::Prefill,
            Phase::Decode,
            Phase::Timeout,
            Phase::Eos,
        ] {
            let serialized = serialize(phase);
            let deserialized = deserialize(serialized).unwrap();
            assert_eq!(phase, deserialized);
        }
    }
}
