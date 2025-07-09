use std::thread::{self, JoinHandle};
use rtrb::{RingBuffer, PushError, PopError, Producer, Consumer, chunks};
use std::hint;
pub fn generate_data() -> (JoinHandle<()>, Consumer<usize>) {
    let (mut tx_prompt, mut rx_prompt): (Producer<usize>,  Consumer<usize>) = RingBuffer::new(4096);
    // [sequence, batch]
    // 第一块 [prompt length, generation length], 后面才是数据
    // 1, 2, 3
    // 5, 6, 7
    let handle1 = thread::spawn(move || {
        
        
        let sequence_num = 10;
        let prompt_length = 8;
        let generation_length = 16;
        let batch_size = 4;
        let position_prompt = vec![1, 2, 3, 4];
        let head = vec![prompt_length, generation_length];
        let position_num = (prompt_length + 1) * sequence_num;
        for i in 0..position_num {
            if i % ( prompt_length + 1) == 0 {
                let mut input = head.iter();
            } else {
                let mut input = position_prompt.iter();
            }
            loop {
                if let Ok(chunk) = tx_prompt.write_chunk_uninit(batch_size) {
                    chunk.fill_from_iter(input);
                    // Note that we requested 4 slots but we've only written to 3 of them!
                    break;
                } else {
                    // thread::sleep(time::Duration::from_millis(1));  
                    hint::spin_loop();
                }
            }
        }
        drop(tx_prompt);
        
    });
    (handle1, rx_prompt)
}