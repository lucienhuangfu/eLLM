#[derive(Clone, Default, Debug)]
pub struct SequenceSlice {
    pub token_start_index: usize,
    pub batch_index: usize,
    pub sequence_index: usize,
    pub length: usize,
    pub last_token_flag: bool,
}
