// use num_traits::Float;

use crate::init::send_sync_ptr::{ConstPtr, MutPtr};


// [batch_size, head_num, head_size] <- [batch_size, head_num, head_size] [batch_size,   head_num,  sequence_num, head_size] [batch_size,   head_num, sequence_num, head_size] 
pub fn chunk_attention<T>(
    data1: *const T,
    shape1: Vec<usize>,
    strides1: Vec<usize>,
    data2: *const T,
    shape2: Vec<usize>,
    strides2: Vec<usize>,
    data3: *const T,
    data4: *mut T,
    shape4: Vec<usize>,
    strides4: Vec<usize>,
) -> Vec<(ConstPtr<T>, ConstPtr<T>, ConstPtr<T>, MutPtr<T>)> {
    let mut task_vec: Vec<(ConstPtr<T>, ConstPtr<T>, ConstPtr<T>, MutPtr<T>)> = Vec::new();
    unsafe {
        let step = if shape1[1] == shape2[1] { 1 } else { shape1[1] / shape2[1] };
        for a in 0..shape1[0] {
            let input1_offset0 = data1.add(a * strides1[0]);
            let input2_offset0 = data2.add(a * strides2[0]);
            let input3_offset0 = data3.add(a * strides2[0]);
            let output_offset0 = data4.add(a * strides4[0]);

            for b in 0..shape1[1] {
                let input1_offset1 = input1_offset0.add(b * strides1[1]);
                let _b = b / step;
                let input2_offset1 = input2_offset0.add(_b * strides2[1]);
                let input3_offset1 = input3_offset0.add(_b * strides2[1]);
                let output_offset1 = output_offset0.add(b * strides4[1]);
                task_vec.push((
                    ConstPtr {
                        ptr: input1_offset1,
                    },
                    ConstPtr {
                        ptr: input2_offset1,
                    },
                    ConstPtr {
                        ptr: input3_offset1,
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

#[cfg(test)]
mod test {
    use crate::kernel;
    use crate::ptensor::tensor_utils::get_strides;
    use super::*;
    use approx::assert_ulps_eq;

    #[test]
    fn test_chunk_attention() {
        let head_size = 128;
        let head_num = 64;
        let batch_size = 16;
        let sequence_length = 256;

        let shape1 = vec![batch_size, head_num, head_size];
        let size1 = shape1.iter().product();
        let data1 = vec![1.0; size1];
        let strides1 = get_strides(&shape1);



        // let shape2 = vec![batch_size, sequence_length, head_num,  head_size];
        
        let shape2 = vec![sequence_length, batch_size, head_num, head_size];
        let strides2 = get_strides(&shape2);

        // reshape
        let _shape2 = vec![batch_size, head_num, sequence_length, head_size];
        // [batch_size, head_num, sequence_length, head_size] <- [sequence_length, batch_size, head_num, head_size]
        let _strides2 = vec![strides2[1], strides2[2], strides2[0], strides2[3]];

        let size2 = shape2.iter().product();
        let data2 = vec![1.0; size2];
        

        let shape3 = vec![batch_size, sequence_length, head_num,  head_size];
        let size3 = shape3.iter().product();
        let data3 = vec![1.0; size3];

        let shape4 = vec![batch_size, head_num, head_size];
        let size4 = shape4.iter().product();
        let mut data4 = vec![0.0; size4];
        let strides4 = get_strides(&shape4);

        let result = vec![1.0; size4];

        let tasks = chunk_attention(
            data1.as_ptr(),
            shape1,
            strides1,
            data2.as_ptr(),
            _shape2,
            _strides2.clone(),
            data3.as_ptr(),
            data4.as_mut_ptr(),
            shape4,
            strides4,
        );

   
        for (input1, input2, input3, output) in tasks {
            kernel::generic::flash_attention::flash_attention(
                input1.ptr,
                input2.ptr,
                input3.ptr,
                output.ptr,
                1.0, // inverse_sqrt_head
                head_size,
                _strides2[2],
                sequence_length - 1
            );
        }     
        
        assert_ulps_eq!(data4[..], result[..], max_ulps = 4);
         
    }

    #[test]
    fn test_chunk_attention_gqa() {
        let head_size = 128;
        let head_num = 64;
        let kv_head_num = 8;
        let batch_size = 16;
        let sequence_length = 256;

        let shape1 = vec![batch_size, head_num, head_size];
        let size1 = shape1.iter().product();
        let data1 = vec![1.0; size1];
        let strides1 = get_strides(&shape1);

        let shape2 = vec![sequence_length, batch_size, kv_head_num, head_size];
        let strides2 = get_strides(&shape2);

        // reshape
        let _shape2 = vec![batch_size, kv_head_num, sequence_length, head_size];
        let _strides2 = vec![strides2[1], strides2[2], strides2[0], strides2[3]];

        let size2 = shape2.iter().product();
        let data2 = vec![1.0; size2];

        let shape3 = vec![batch_size, sequence_length, kv_head_num, head_size];
        let size3 = shape3.iter().product();
        let data3 = vec![1.0; size3];

        let shape4 = vec![batch_size, head_num, head_size];
        let size4 = shape4.iter().product();
        let mut data4 = vec![0.0; size4];
        let strides4 = get_strides(&shape4);

        let result = vec![1.0; size4];

        let tasks = chunk_attention(
            data1.as_ptr(),
            shape1,
            strides1,
            data2.as_ptr(),
            _shape2,
            _strides2.clone(),
            data3.as_ptr(),
            data4.as_mut_ptr(),
            shape4,
            strides4,
        );

        for (input1, input2, input3, output) in tasks {
            kernel::generic::flash_attention::flash_attention(
                input1.ptr,
                input2.ptr,
                input3.ptr,
                output.ptr,
                1.0,
                head_size,
                _strides2[2],
                sequence_length - 1,
            );
        }

        assert_ulps_eq!(data4[..], result[..], max_ulps = 4);
    }

}
