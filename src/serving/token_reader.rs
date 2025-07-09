use std::{path::Path, time::Duration, thread::JoinHandle};
use smol::{self, stream::StreamExt};
use rtrb::{RingBuffer, Consumer};
use std::thread;
use smol::io::AsyncBufReadExt;

pub fn from_file<P: AsRef<Path>>(filename: P) -> (JoinHandle<()>, Consumer<usize>) {
    let filename = filename.as_ref().to_owned();
    let (mut producer, consumer) = RingBuffer::<usize>::new(4100);
    let handle = thread::spawn(move || {
        smol::block_on(async {
            let file = smol::fs::File::open(filename).await.unwrap();
            let reader = smol::io::BufReader::new(file);
            let mut lines = reader.lines();
            while let Some(line) = lines.next().await {
                let content = line.unwrap();
                let tokens: Vec<usize> = content.split(',').map(|x| x.parse::<usize>().unwrap()).collect();
                while producer.slots() < tokens.len() + 2 {
                    thread::sleep(Duration::from_millis(200));
                }
                producer.push(256).unwrap();            // sequnce_length
                producer.push(tokens.len()).unwrap();   // prompt_size
                if let Ok(chunk) = producer.write_chunk_uninit(tokens.len()) {
                    chunk.fill_from_iter(tokens);
                }
                else {
                    unreachable!();
                }
            }
            drop(producer);
        });      
    });
    (handle, consumer)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_from_file() {
        let (handle, mut consumer) = from_file(r"token_file.txt");
        loop {
            if consumer.is_empty() && consumer.is_abandoned() { break; }
            
            let sequence_length: usize = loop {
                match consumer.pop() {
                    Ok(value) => {
                        break value
                    },
                    Err(_) => thread::sleep(Duration::from_millis(200))
                }
            };
            let prompt_size: usize = loop {
                match consumer.pop() {
                    Ok(value) => {
                        break value
                    },
                    Err(_) => thread::sleep(Duration::from_millis(200))
                }
            };
            let tokens: Vec<usize> = loop {
                let mut tokens: Vec<usize> = Vec::with_capacity(prompt_size);
                match consumer.read_chunk(prompt_size) {
                    Ok(chunk) => {
                        let (first, second) = chunk.as_slices();
                        tokens.extend(first);
                        tokens.extend(second);
                        chunk.commit_all();
                        break tokens
                    },
                    Err(_) => thread::sleep(Duration::from_millis(200))
                }
            };
            println!("{} {}\n{:?}", sequence_length, prompt_size, tokens);
        }
        handle.join().unwrap();
    }
}