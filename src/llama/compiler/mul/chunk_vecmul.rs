use super::super::init::send_sync_ptr::{ConstPtr, MutPtr};

pub fn chunk_vecmul<T>(
    data1: *const T,
    shape1: Vec<usize>,
    strides1: Vec<usize>,
    data2: *const T,
    shape2: Vec<usize>,
    strides2: Vec<usize>,
    data3: *mut T,
    shape3: Vec<usize>,
    strides3: Vec<usize>,
) -> Vec<(ConstPtr<T>, ConstPtr<T>, MutPtr<T>)> {
    //就是向量乘法
    // [batch_size , head_num, 1, sequence_num] <- [batch_size, head_num, 1， head_size] * [batch_size, head_num, sequence_num,  head_size]

    // [batch_size*head_num, sequence]
    let mut task_vec: Vec<(ConstPtr<T>, ConstPtr<T>, MutPtr<T>)> = Vec::new();
    unsafe {
        if shape1[1] == shape2[1] {
            for a in 0..shape1[0] {
                let input1_offset0 = data1.add(a * strides1[0]);
                let input2_offset0 = data2.add(a * strides2[0]);
                let output_offset0 = data3.add(a * strides3[0]);

                for b in 0..shape1[1] {
                    let input1_offset1 = input1_offset0.add(b * strides1[1]);
                    let input2_offset1 = input2_offset0.add(b * strides2[1]);
                    let output_offset1 = output_offset0.add(b * strides3[1]);

                    for c in 0..shape1[2] {
                        let input1_offset2 = input1_offset1.add(c * strides1[2]);
                        let output_offset2 = output_offset1.add(c * strides3[2]);

                        for d in 0..shape2[2] {
                            let input2_offset2 = input2_offset1.add(d * strides2[2]);
                            let output_offset3 = output_offset2.add(d * strides3[3]);
                            task_vec.push((
                                ConstPtr {
                                    ptr: input1_offset2,
                                },
                                ConstPtr {
                                    ptr: input2_offset2,
                                },
                                MutPtr {
                                    ptr: output_offset3,
                                },
                            ));
                        }
                    }
                }
            }
        } else {

            // [batch_size , head_num, 1, sequence_num] <- [batch_size, head_num, 1， head_size] * [batch_size, kv_head_num, sequence_num,  head_size]
            // [batch_size*head_num, sequence]
            let step = shape1[1] / shape2[1];
            for a in 0..shape1[0] {
                let input1_offset0 = data1.add(a * strides1[0]);
                let input2_offset0 = data2.add(a * strides2[0]);
                let output_offset0 = data3.add(a * strides3[0]);

                for b in 0..shape1[1] {
                    let input1_offset1 = input1_offset0.add(b * strides1[1]);
                    let _b = b / step;
                    let input2_offset1 = input2_offset0.add(_b * strides2[1]);
                    let output_offset1 = output_offset0.add(b * strides3[1]);

                    for c in 0..shape1[2] {
                        let input1_offset2 = input1_offset1.add(c * strides1[2]);
                        let output_offset2 = output_offset1.add(c * strides3[2]);

                        for d in 0..shape2[2] {
                            let input2_offset2 = input2_offset1.add(d * strides2[2]);
                            let output_offset3 = output_offset2.add(d * strides3[3]);
                            task_vec.push((
                                ConstPtr {
                                    ptr: input1_offset2,
                                },
                                ConstPtr {
                                    ptr: input2_offset2,
                                },
                                MutPtr {
                                    ptr: output_offset3,
                                },
                            ));
                        }
                    }
                }
            }
        }
    }
    task_vec
}

#[cfg(test)]
mod test {
    use super::super::super::kernel;
    use super::super::tensor_utils::get_strides;
    use super::*;
    use approx::assert_ulps_eq;

    use std::slice;

    #[test]
    fn test_chunk_vec() {
        let head_size = 128;
        let head_num = 64;
        let batch_size = 16;
        let sequence_length = 256;

        let q_shape = vec![batch_size, head_num, 1, head_size];
        let q_size = q_shape.iter().product();
        let q_data: Vec<f32> = vec![1.0; q_size];
        let q_strides = get_strides(&q_shape);

        let k_shape = vec![batch_size, head_num, sequence_length, head_size];
        let k_size = k_shape.iter().product();
        let k_data: Vec<f32> = vec![1.0; k_size];
        let k_strides = get_strides(&k_shape);

        let s_shape = vec![batch_size, head_num, 1, sequence_length];
        let s_size = s_shape.iter().product();
        let mut s_data: Vec<f32> = vec![0.0; s_size];
        let s_strides = get_strides(&s_shape);

        let result = vec![head_size as f32; s_size];

        let tasks = chunk_vecmul(
            q_data.as_ptr(),
            q_shape,
            q_strides,
            k_data.as_ptr(),
            k_shape,
            k_strides,
            s_data.as_mut_ptr(),
            s_shape,
            s_strides,
        );

        for (input1, input2, output) in tasks {
            kernel::generic::dot_product::dot_product(
                input1.ptr, input2.ptr, output.ptr, head_size,
            );
            // let slice1 = unsafe { std::slice::from_raw_parts(input1.ptr, head_size) };
            // let slice2 = unsafe { std::slice::from_raw_parts(input2.ptr, head_size) };
            // let value = unsafe { &*output.ptr };
            // println!("{:?}", slice1);
            // println!("{:?}", slice2);
            // println!("output: {:?}", value);
        }

        assert_ulps_eq!(s_data[..], result[..], max_ulps = 4);
    }
}
