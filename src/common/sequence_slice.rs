#[derive(Clone)]
pub struct SequenceSlice {
    pub token_start_index: usize,
    pub batch_index: usize,
    pub sequence_index: usize,
    pub length: usize,
    pub lift_index: usize,
}