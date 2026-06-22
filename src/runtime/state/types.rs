#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}
