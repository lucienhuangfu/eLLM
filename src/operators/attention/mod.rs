// Core attention structure and implementation
mod attention;
// AttentionTrait implementations for different data types
mod compute;
// Scratch buffers for thread-local computation
mod scratch;
// Utility functions for sequence splitting
mod utils;

pub use attention::Attention;
