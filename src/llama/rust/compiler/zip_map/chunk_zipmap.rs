// use itertools::Itertools;
// use num_traits::Float;
// use super::tensor_utils::get_strides;
use crate::init::send_sync_ptr::{ConstPtr, MutPtr};

pub fn chunk_zipmap<T>(
        shapes: Vec<usize>,
        input_data1: *const T,
        input_strides1: Vec<usize>,
        input_data2: *const T,
        input_strides2: Vec<usize>,
        output_data: *mut T,
        output_strides: Vec<usize>) -> Vec<(ConstPtr<T>, ConstPtr<T>, MutPtr<T>)> {
    let mut tasks: Vec<(ConstPtr<T>, ConstPtr<T>, MutPtr<T>)> = Vec::new();
    unsafe {
        match shapes.len() {
            // [sequence * batch * head_num]  <- [sequence, batch, head_num, head_size]
            4 => {
                println!("chunk zip map shape {:?}", shapes);
                // output_strides[0] = 0;
                for i in 0..shapes[0] {
                    let input1_stride0 = input_data1.add(i * input_strides1[0]);
                    let input2_stride0 = input_data2.add(i * input_strides2[0]);
                    let output_stride0 = output_data.add(i * output_strides[0]);

                    for j in 0..shapes[1] {
                        let input1_stride1 = input1_stride0.add(j * input_strides1[1]);
                        let input2_stride1 = input2_stride0.add(j * input_strides2[1]);
                        let output_stride1 = output_stride0.add(j * output_strides[1]);
                        for k in 0..shapes[2] {
                            let input1_stride2 = input1_stride1.add(k * input_strides1[2]);
                            let input2_stride2 = input2_stride1.add(k * input_strides2[2]);
                            let output_stride2 = output_stride1.add(k * output_strides[2]);
                            tasks.push((
                                ConstPtr {ptr: input1_stride2},
                                ConstPtr {ptr: input2_stride2},
                                MutPtr {ptr: output_stride2}));
                        }
                    }
                }
                tasks
            },
            3 => {
                println!("chunk zip map shape {:?}", shapes);
                // [batch*num]  <- [batch, num, head_size]
                for i in 0..shapes[0] {
                    let input1_stride0 = input_data1.add(i * input_strides1[0]);
                    let input2_stride0 = input_data2.add(i * input_strides2[0]);
                    let output_stride0 = output_data.add(i * output_strides[0]);

                    for j in 0..shapes[1] {
                        let input1_stride1 = input1_stride0.add(j * input_strides1[1]);
                        let input2_stride1 = input2_stride0.add(j * input_strides2[1]);
                        let output_stride1 = output_stride0.add(j * output_strides[1]);
                        tasks.push((
                            ConstPtr {ptr: input1_stride1},
                            ConstPtr {ptr: input2_stride1},
                            MutPtr {ptr: output_stride1}));
                    }
                }
                tasks
            },
            
            
            // [batch * hidden_size]
            2 => {
                for i in 0..shapes[0] {
                    let input1_stride0 = input_data1.add(i * input_strides1[0]);
                    let input2_stride0 = input_data2.add(i * input_strides2[0]);
                    let output_stride0 = output_data.add(i * output_strides[0]);
                    tasks.push((
                        ConstPtr {ptr: input1_stride0},
                        ConstPtr {ptr: input2_stride0},
                        MutPtr {ptr: output_stride0}));
                }
                tasks
            },
            _ => {panic!("wrong shape")}
        }
    }
}


#[cfg(test)]
mod test {
    
    // use std::f16;
    use num_traits::Float;
    use std::slice;
    use approx::assert_ulps_eq;

    use crate::ptensor::tensor_utils::get_strides;
    use super::*;
    // use core::task;

    use std::ops::{Add,Sub, Mul, Div, Neg};

    use crate::init::send_sync_ptr::{ConstPtr, MutPtr};

    fn fun<T:Float>( input_ptr1: ConstPtr<T>,input_ptr2: ConstPtr<T>, output_ptr:MutPtr<T>, length: usize) {
        let mut input_ptr1 = input_ptr1.ptr;
        let mut input_ptr2 = input_ptr2.ptr;
        let mut output_ptr = output_ptr.ptr;
        for i in 0..length {
            unsafe {
                let val1 = input_ptr1.add(i).read();
                let val2 = input_ptr2.add(i).read();
                output_ptr.add(i).write(val1 + val2);
            }
        }
    }

    #[test]
    fn test2() {
        let head_size = 128;
        let head_num  = 64;
        let batch_size = 16;
        let sequence_length = 256;
        let hidden_size = 8192;
        
        let shapes = vec![batch_size, hidden_size];
        let length = shapes.iter().product();
        let data1: Vec<f32> = vec![1.0; length];
        let data2: Vec<f32> = vec![1.0; length];
        let mut output: Vec<f32> = vec![0.0; length];
        let expected: Vec<f32> = vec![2.0; length];
        let input_strides1 = get_strides(&shapes);
        let input_strides2 = input_strides1.clone();
        let output_strides = input_strides1.clone();
        
        let tasks = chunk_zipmap(shapes, data1.as_ptr(), input_strides1, data2.as_ptr(), input_strides2, output.as_mut_ptr(),output_strides);
        for (input1, input2, output) in tasks {
            fun(input1, input2, output, hidden_size);
        }

        assert_ulps_eq!(output[..], expected[..], max_ulps = 4);
        // println!("{:?}", output);
    }

    #[test]
    fn test3() {
        let batch_size = 16;
        let head_num  = 64;
        let head_size = 128;
        // let hidden_size = 8192;
        // let sequence_length = 256;
    
        let shapes = vec![batch_size, head_num, head_size];
        let length = shapes.iter().product();
        let data1: Vec<f32> = vec![1.0; length];
        let data2: Vec<f32> = vec![1.0; length];
        let mut output = vec![0.0; length];
        let expected: Vec<f32> = vec![2.0; length];

        let input_strides1 = get_strides(&shapes);
        let input_strides2 = input_strides1.clone();
        let output_strides = input_strides1.clone();
        let tasks = chunk_zipmap(shapes, data1.as_ptr(), input_strides1, data2.as_ptr(), input_strides2, output.as_mut_ptr(),output_strides);
        for (input1, input2, output) in tasks {
            fun(input1, input2, output, head_size);
        }
        // println!("{:?}", output);
        assert_ulps_eq!(output[..], expected[..], max_ulps = 4);
    }




    #[test]
    fn test4() {
        let head_size = 128;
        let head_num  = 64;
        let hidden_size = 8192;
        let batch_size = 16;
        let sequence_length = 256;
    
        let shapes = vec![sequence_length, batch_size, head_num, head_size];
        let length = shapes.iter().product();
        let data1: Vec<f32> = vec![1.0; length];
        let data2: Vec<f32> = vec![1.0; length];
        let mut output = vec![0.0; length];
        let expected: Vec<f32> = vec![2.0; length];

        let input_strides1 = get_strides(&shapes);
        let input_strides2 = input_strides1.clone();
        let output_strides = input_strides1.clone();
        let tasks = chunk_zipmap(shapes, data1.as_ptr(), input_strides1, data2.as_ptr(), input_strides2, output.as_mut_ptr(),output_strides);
        for (input1, input2, output) in tasks {
            fun(input1, input2, output, head_size);
        }
        // println!("{:?}", output);
        assert_ulps_eq!(output[..], expected[..], max_ulps = 4);
    }


}