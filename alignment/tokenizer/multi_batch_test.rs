#![feature(f16)]
#![feature(sync_unsafe_cell)]

use ellm::auto::AutoTokenizer;
use ellm::config::GenerationConfig;
use ellm::mem_mgr::allocator::AlignedBox;
use ellm::mem_mgr::mem_pool::GlobalMemPool;
use ellm::operators::operator::Operator;
use ellm::runtime::loader::SafeTensorsLoader;
use ellm::runtime::Phase;
use ellm::runtime::SequenceSlice;
use ellm::tensor::GlobalOperatorQueue;
use ellm::transformer::rope::RotaryEmbedding;
use std::cell::SyncUnsafeCell;
use std::f16;
use std::sync::{Arc, Barrier};

fn main() {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "checkpoints/Qwen3-0.6B".to_string());
    let tokenizer = AutoTokenizer::from_pretrained(&model_dir).unwrap();

    let messages = [("user", "你好，请用一句话介绍 Rust。")];
    let prompt = tokenizer.apply_chat_template(&messages, true).unwrap();
    let token_ids = tokenizer.encode(&prompt);

    println!("Prompt: {}", prompt);
    println!("Token count: {}", token_ids.len());
}
