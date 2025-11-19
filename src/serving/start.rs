use core_affinity;
use std::rc::Rc;
use std::cell::RefCell;
use std::cell::SyncUnsafeCell;
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use std::sync::Barrier;

// use hurdles::Barrier;
// use super::barrier::Barrier;
// use serde::{Deserialize, Serialize};
// use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use super::super::compiler::operator::Operator;
// use crate::kernel::generic::from_f32::FromF32;
// use crate::kernel::generic::sigmoid::Sigmoid;
// use crate::kernel::generic::sqrt::Sqrt;
// use crate::kernel::generic::{exp::Exp, neg_infinity::NegInfinity};
// use super::state::State;


pub fn start(operator_queue: Rc<RefCell<Vec<Operator<f32>>>>) {
    println!("start");
    // let prompt_operator_num;
    // let data = SyncUnsafeCell::new(DataReader::new(prompt_data));
    let thread_num = thread::available_parallelism().unwrap().get();
    let sync_operator_queue = Arc::new(operator_queue.borrow().clone());

    let barrier = Arc::new(Barrier::new(thread_num));

    let sequence_chunk_size = 64;
    // let mut handles = Vec::with_capacity(thread_num);
    let core_ids = core_affinity::get_core_ids().unwrap();
    for (i, core_id) in core_ids.into_iter().enumerate() {
        println!("thread id {}", i);
        // let _state = &state;
        // let _prompt_begin = &prompt_begin;
        // let _prompt_end = &prompt_end;
        // let _generation_end = &generation_end;
        // let _batch_size = &batch_size;
        let b = Arc::clone(&barrier);
        // let mut b = barrier .clone();
        let queue = Arc::clone(&sync_operator_queue);

        // let start_pos = start_pos; // 显式捕获当前值

        // let decode_start = 40;

        let handle = thread::spawn(move || {
            let thread_id = i;
            core_affinity::set_for_current(core_id);
            println!("{} start", thread_id);
            // let mut counter = 0;

            // 预先创建子切片，避免在热循环中重复操作
            // let prompt_queue_slice = &queue[..decode_start.min(queue.len())];
            // let decode_queue_slice = &queue[decode_start.min(queue.len())..];

             let s = Instant::now();
            let batch_size = 1;
            // only decode part
            for operator in queue.iter() {
                operator.run(0, 1, batch_size, thread_num, thread_id);
                b.wait();
            }
            let t = s.elapsed();
            println!("thread {} decode time {:?}", thread_id, t);


        });

        std::mem::forget(handle);
        // handles.push(handle);
    }

    /*
    for handle in handles {
        handle.join().unwrap();
    }*/
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_start() {
        let operator_queue: Vec<Operator<f32>> = Vec::new();
        start(operator_queue);
    }
}
