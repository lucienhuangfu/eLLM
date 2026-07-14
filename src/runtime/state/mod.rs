pub mod batch;
pub mod sequence;

pub use batch::{build_batch_sequence, BatchSequence};
pub use sequence::{DecodeLookupResult, SequenceSlice};
