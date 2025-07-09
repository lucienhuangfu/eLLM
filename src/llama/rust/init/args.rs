use std::env;

fn read_args() -> (Vec<(usize, usize, usize)>, usize, usize) {
    let mut max_batch_size = 0;
    let mut max_sequence_size = 0;
    let args: Vec<String> = env::args().collect();
    let data: Vec<_> = args[1..].chunks(3).map(|x| {
        let batch_size = x[0].parse::<usize>().unwrap();
        let prompt = x[1].parse::<usize>().unwrap();
        let total = x[2].parse::<usize>().unwrap(); 

        assert!(total >= prompt, "total sequence length should not be less than prompt length");

        if batch_size > max_batch_size { max_batch_size = batch_size; }
        if total > max_sequence_size { max_sequence_size = total; }
        
        (batch_size, prompt, total)
    }).collect();

    let (quo, rem) = (max_batch_size/3, max_batch_size%3);
    if rem == 0 { max_batch_size = quo * 3 } else { max_batch_size = (quo + 1) * 3; }
    (data, max_batch_size, max_sequence_size)
}