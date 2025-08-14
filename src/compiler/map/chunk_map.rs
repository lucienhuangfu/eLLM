// use std::f16;
// use num_traits::Float;

use crate::init::send_sync_ptr::{ConstPtr, MutPtr};

pub fn chunk_map<T>(
    shapes: Vec<usize>,
    strides: Vec<usize>,
    input_data: *const T,
    output_data: *mut T,
) -> Vec<(ConstPtr<T>, MutPtr<T>)> {
    let mut tasks: Vec<(ConstPtr<T>, MutPtr<T>)> = Vec::new();

    unsafe {
        match shapes.len() {
            2 => {
                for i in 0..shapes[0] {
                    let input_stride0 = input_data.add(i * strides[0]);
                    let output_stride0 = output_data.add(i * strides[0]);

                    tasks.push((
                        ConstPtr { ptr: input_stride0 },
                        MutPtr {
                            ptr: output_stride0,
                        },
                    ));
                }
                tasks
            }
            3 => {
                for i in 0..shapes[0] {
                    let input_stride0 = input_data.add(i * strides[0]);
                    let output_stride0 = output_data.add(i * strides[0]);
                    for j in 0..shapes[1] {
                        let input_stride1 = input_stride0.add(j * strides[1]);
                        let output_stride1 = output_stride0.add(j * strides[1]);
                        tasks.push((
                            ConstPtr { ptr: input_stride1 },
                            MutPtr {
                                ptr: output_stride1,
                            },
                        ));
                    }
                }
                tasks
            }
            4 => {
                for i in 0..shapes[0] {
                    let input_stride0 = input_data.add(i * strides[0]);
                    let output_stride0 = output_data.add(i * strides[0]);
                    for j in 0..shapes[1] {
                        let input_stride1 = input_stride0.add(j * strides[1]);
                        let output_stride1 = output_stride0.add(j * strides[1]);
                        for k in 0..shapes[2] {
                            let input_stride2 = input_stride1.add(k * strides[2]);
                            let output_stride2 = output_stride1.add(k * strides[2]);
                            tasks.push((
                                ConstPtr { ptr: input_stride2 },
                                MutPtr {
                                    ptr: output_stride2,
                                },
                            ));
                        }
                    }
                }
                tasks
            }
            _ => {
                panic!()
            }
        }
    }
}

#[cfg(test)]
mod test {
    // use std::f16;
    use num_traits::Float;

    use crate::kernel::generic::from_usize::FromUsize;
    use std::ops::{Add, Div, Mul, Neg, Sub};

    use super::*;
    use crate::init::send_sync_ptr::{ConstPtr, MutPtr};
    use crate::ptensor::tensor_utils::get_strides;
    // use core::ops::Deref;

    fn fun<T: Float>(input_ptr: ConstPtr<T>, output_ptr: MutPtr<T>, length: usize) {
        for i in 0..length {
            unsafe {
                *(output_ptr.ptr.add(i)) = *(input_ptr.ptr.add(i)) + T::from(1).unwrap();
            }
        }
    }

    #[test]
    fn test2dimension() {
        let head_size = 128;
        let head_num = 64;
        let batch_size = 16;
        let sequence_length = 256;
        let hidden_size = 8192;

        let shapes = vec![batch_size, hidden_size];
        let length = shapes.iter().product();
        let strides = get_strides(&shapes);
        let data = vec![1.0; length];
        let mut output = vec![0.0; length];
        let tasks = chunk_map(shapes, strides, data.as_ptr(), output.as_mut_ptr());
        for (input, output) in tasks {
            fun(input, output, head_size);
        }
        //println!("{:?}", output);
    }

    #[test]
    fn test3dimension() {
        let head_size = 128;
        let head_num = 64;
        let batch_size = 16;
        let sequence_length = 256;
        let hidden_size = 8192;

        let shapes = vec![batch_size, head_num, head_size];
        let length = shapes.iter().product();
        let strides = get_strides(&shapes);
        let data = vec![1.0; length];
        let mut output = vec![0.0; length];
        let tasks = chunk_map(shapes, strides, data.as_ptr(), output.as_mut_ptr());
        for (input, output) in tasks {
            fun(input, output, head_size);
        }
        //println!("{:?}", output);
    }

    #[test]
    fn test4dimension() {
        let inner_size = 32;
        let head_size = 128;
        let head_num = 64;
        let batch_size = 16;
        let sequence_length = 256;

        let shapes = vec![batch_size, head_num, head_size, inner_size];
        let length = shapes.iter().product();
        let strides = get_strides(&shapes);
        let data = vec![1.0; length];
        let mut output = vec![0.0; length];
        let tasks = chunk_map(shapes, strides, data.as_ptr(), output.as_mut_ptr());
        for (input, output) in tasks {
            fun(input, output, inner_size);
        }
        //println!("{:?}", output);
    }
}
