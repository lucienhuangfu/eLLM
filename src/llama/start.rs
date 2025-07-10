use std::thread;
use std::time::Instant;
use core_affinity;
use std::cell::SyncUnsafeCell;
use std::sync::Arc;
// use std::sync::Barrier;
// use hurdles::Barrier;
use super::barrier::Barrier;
// use serde::{Deserialize, Serialize};
use std::ops::{Add, Sub, Div, Mul, AddAssign, Neg };


use crate::kernel::generic::sqrt::Sqrt;
use crate::kernel::generic::{neg_infinity::NegInfinity, exp::Exp};
use crate::kernel::generic::sigmoid::Sigmoid;
use crate::kernel::generic::from_f32::FromF32;
use super::super::compiler::operator::Operator;
// use super::state::State;

pub fn start<T>(operator_queue: Vec<Operator<T>>) 
where T: Copy 
    + Default 
    + Sub<Output = T>
    + Neg<Output = T>
    + Exp
    + NegInfinity
    + Sigmoid<T>
    + Sqrt
    + FromF32
    // + for<'de> Deserialize<'de>
+ Send
+ Sync,
{
    println!("start");
    // let prompt_operator_num;
    // let data = SyncUnsafeCell::new(DataReader::new(prompt_data));
    let cpu_num = thread::available_parallelism().unwrap().get();
    // let sync_operator_queue = Arc::new(SyncUnsafeCell::new(operator_queue));
    let sync_operator_queue = Arc::new(operator_queue);
    let barrier = Arc::new(Barrier::new(cpu_num));
   
   

    thread::scope(|s| {
        let mut handles = Vec::with_capacity(cpu_num);
        // let reader = SyncUnsafeCell::new(DataReader::new(prompt_data));
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

            let handle = s.spawn(move || {
                
                let thread_id = i;
                core_affinity::set_for_current(core_id);
                println!("{} start", thread_id);
                // let mut counter = 0;

                loop {
                    unsafe {
                        let batch_size: usize = 1;
                        let prompt_begin = 0;
                        let prompt_end = 0;
                        // let generation_end = 128;
                      
                            let s = Instant::now();
                            for position_index in prompt_end..generation_end {

                                for operator in queue.iter() {
                                    // println!("o index {}", o_index);
                                    operator.run(batch_size, position_index, thread_id);
                                    b.wait();
                                }
                                // println!("position {}", position_index);
                            }
                            let t = s.elapsed();
               

                        // break;
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
    });
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
