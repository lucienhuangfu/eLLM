use num_traits::Float;

use super::super::init::send_sync_ptr::{ConstPtr, MutPtr};

pub fn chunk_colmul<T>(
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
    //  [batch_size, head_num, head_size]  <- [batch_size , head_num, sequence_num] * [batch_size,  head_num, sequence_num, head_size]
    //  rearrange dimension
    //  [batch_size*head_size, task]
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
                    task_vec.push((
                        ConstPtr {
                            ptr: input1_offset1,
                        },
                        ConstPtr {
                            ptr: input2_offset1,
                        },
                        MutPtr {
                            ptr: output_offset1,
                        },
                    ));
                }
            }
        } else {
            //  [batch_size, head_num, head_size]  <- [batch_size , head_num, sequence_num] * [batch_size,  kv_head_num, sequence_num, head_size]

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
                    task_vec.push((
                        ConstPtr {
                            ptr: input1_offset1,
                        },
                        ConstPtr {
                            ptr: input2_offset1,
                        },
                        MutPtr {
                            ptr: output_offset1,
                        },
                    ));
                }
            }
        }

        task_vec
    }
}

#[cfg(test)]
mod test {
    use super::super::super::kernel;
    use super::super::tensor_utils::get_strides;
    use super::*;
    use approx::assert_ulps_eq;

    #[test]
    fn test_chunk_col1() {
        let head_size = 128;
        let head_num = 64;
        let batch_size = 16;
        let sequence_length = 256;

        let shape1 = vec![batch_size, head_num, sequence_length];
        let size1 = shape1.iter().product();
        let data1 = vec![1.0; size1];
        let strides1 = get_strides(&shape1);

        let shape2 = vec![batch_size, head_num, sequence_length, head_size];
        let size2 = shape2.iter().product();
        let data2 = vec![1.0; size2];
        let strides2 = get_strides(&shape2);

        let shape3 = vec![batch_size, head_num, head_size];
        let size3 = shape3.iter().product();
        let mut data3 = vec![0.0; size3];
        let strides3 = get_strides(&shape3);

        let result = vec![sequence_length as f32; size3];

        let tasks = chunk_colmul(
            data1.as_ptr(),
            shape1,
            strides1,
            data2.as_ptr(),
            shape2,
            strides2,
            data3.as_mut_ptr(),
            shape3,
            strides3,
        );

        // println!("{:?}", data3);
        for (input1, input2, output) in tasks {
            kernel::generic::colmul::colmul(
                input1.ptr,
                input2.ptr,
                output.ptr,
                sequence_length,
                head_size,
            );
        }
        assert_ulps_eq!(data3[..], result[..], max_ulps = 4);
    }
}
