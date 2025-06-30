// use itertools::Itertools;
use crate::init::send_sync_ptr::{ConstPtr, MutPtr};

pub fn chunk_reduce<T>(
        shapes: Vec<usize>,
        input_data: *const T,
        input_strides: Vec<usize>,
        output_data: *mut usize,
        output_strides: Vec<usize>) -> Vec<(ConstPtr<T>, MutPtr<usize>)> {
    let mut tasks: Vec<(ConstPtr<T>, MutPtr<usize>)> = Vec::new();
    unsafe {
        match shapes.len() {
            // [batch_size]  <- [batch, vocab_size]
            2 => {
                for i in 0..shapes[0] {
                    let input1_stride0 = input_data.add(i * input_strides[0]);
                    // let input2_stride0 = input_data2.add(i * input_strides2[0]);
                    let output_stride0 = output_data.add(i * output_strides[0]);
                    tasks.push((
                        ConstPtr {ptr: input1_stride0},
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
    use super::*;
    use crate::ptensor::tensor_utils::get_strides;
    use crate::init::send_sync_ptr::{ConstPtr, MutPtr};
    // use num_traits::Float;

    fn fun<T>( input_ptr1: ConstPtr<T>, output_ptr: MutPtr<usize>, length: usize) {
        let input_ptr1 = input_ptr1.ptr;
        // let mut input_ptr2 = input_ptr2.ptr;
        let mut output_ptr = output_ptr.ptr;
        unsafe {
            *output_ptr = 1;
        }
    }

    #[test]
    fn test2() {
        let head_size = 128;
        let head_num  = 64;
        let batch_size = 16;
        let sequence_length = 256;
        let vocab_size = 8;
        
        let shapes = vec![batch_size, vocab_size];
        let length = shapes.iter().product();
        let data1: Vec<f32> = vec![1.0; length];
        // let data2: Vec<f16> = vec![1.0); length];
        let mut output = vec![0usize; batch_size];
        let result = vec![1usize; batch_size];

        let input_strides1 = get_strides(&shapes);
        // let input_strides2 = input_strides1.clone();
        let output_strides = vec![1usize];
        
        let tasks = chunk_reduce(shapes, data1.as_ptr(), input_strides1, output.as_mut_ptr(), output_strides);
        for (input1_ptr, output_ptr) in tasks {
            fun(input1_ptr,  output_ptr, vocab_size);
        }
        assert_eq!(output, result)
        // println!("{:?}", output);
    }

}