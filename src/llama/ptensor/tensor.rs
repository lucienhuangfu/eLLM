use std::cell::RefCell;
use std::iter::zip;
use std::rc::Rc;
use std::ops::{Add, Sub, Div, Mul, AddAssign, Neg };
use crate::kernel::generic::sqrt::Sqrt;
use crate::kernel::generic::{neg_infinity::NegInfinity, exp::Exp};
use crate::kernel::generic::sigmoid::Sigmoid;

use super::super::compiler::operator::Operator;
use super::super::memory::cache::Cache;
use crate::compiler::map::chunk_map::chunk_map;
use crate::compiler::mul::chunk_matmul::chunk_matmul;
use crate::compiler::mul::chunk_attention::chunk_attention;
use crate::compiler::reduce::chunk_reduce::chunk_reduce;
use crate::compiler::zip_map::chunk_zipmap::chunk_zipmap;
// use super::chunk_colmul::chunk_colmul;
// use super::chunk_vecmul::chunk_vecmul;

use super::tensor_utils::{get_aligned_strides, get_broadcast_shape, get_strides};
use crate::init::matmul_params::MatMulParams;

#[derive(Clone)]
pub struct Tensor<T> {
    pub data: *mut T,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub tensor_name: String,
    pub cache: Rc<RefCell<Cache<T>>>,
    pub operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
    // pub size: usize,
    // pub is_contiguous: bool,
}

impl<T> Tensor<T>
where T: Copy 
    + Default 
    + Sub<Output = T>
    + Neg<Output = T>
    + Exp
    + NegInfinity
    + Sigmoid<T>
    + Sqrt
{
    pub fn zeros(
        shape: Vec<usize>,
        tensor_name: String,
        cache: Rc<RefCell<Cache<T>>>,
        operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
    ) -> Self {
        let length: usize = shape.iter().product();
        let v = Self::from_cache(shape, tensor_name, cache, operator_queue);
        (0..length).for_each(|x| unsafe {
            *v.data.add(x) = T::default();
        });
        v
    }

    /*
    pub fn from_vec(mut array: Vec<T>, shape: Vec<usize>, tensor_name: String) -> Self {
        let length: usize = shape.iter().product();
        // let data = cache.get(&tensor_name, length);
        let strides = get_strides(&shape);
        Tensor {
            data: array.as_mut_ptr(),
            shape: shape.clone(),
            strides: strides,
            tensor_name: tensor_name,
        }
    } */

    pub fn from_cache(
        shape: Vec<usize>,
        tensor_name: String,
        cache: Rc<RefCell<Cache<T>>>,
        operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
    ) -> Self {
        let length: usize = shape.iter().product();
        let data = cache.borrow_mut().get(&tensor_name, length);
        let strides = get_strides(&shape);
        Tensor {
            data: data,
            shape: shape.clone(),
            strides: strides,
            tensor_name: tensor_name,
            cache: cache.clone(),
            operator_queue: operator_queue.clone(),
        }
    }
    pub fn attention(
        &self,
        tensor2: &Tensor<T>,
        tensor3: &Tensor<T>,
        mut operator: Operator<T>,
        tensor_name: String,
    ) -> Self {
        let output_shape = self.shape.clone();
        let output_tensor = Tensor::from_cache(
            output_shape.clone(),
            tensor_name,
            self.cache.clone(),
            self.operator_queue.clone(),
        );
        let chunks = chunk_attention(
            self.data,
            self.shape.clone(),
            self.strides.clone(),
            tensor2.data,
            tensor2.shape.clone(),
            tensor2.strides.clone(),
            tensor3.data,
            output_tensor.data,
            output_tensor.shape.clone(),
            output_tensor.strides.clone(),
        );
        operator.set_attention_chunk(chunks);
        self.operator_queue.borrow_mut().push(operator);
        output_tensor
        //重点！
    }

    pub fn mapv(
        &self,
        mut operator: Operator<T>,
        tensor_name: String,
        // cache: &mut Cache<T>,
        // operator_queue: &mut Vec<Operator<T>>,
    ) -> Self {
        let output_tensor = Tensor::from_cache(
            self.shape.clone(),
            tensor_name,
            self.cache.clone(),
            self.operator_queue.clone(),
        );
        let chunks = chunk_map(
            self.shape.clone(),
            self.strides.clone(),
            self.data,
            output_tensor.data,
        );
        operator.set_map_chunk(chunks);
        self.operator_queue.borrow_mut().push(operator);
        output_tensor
    }

    pub fn zip_mapv(
        &self,
        b_tensor: &Tensor<T>,
        mut operator: Operator<T>,
        partial_broadcast: bool,
        tensor_name: String,
        // cache: &mut Cache<T>,
        // operator_queue: &mut Vec<Operator<T>>,
    ) -> Self {
        let broadcast_shape = get_broadcast_shape(&self.shape, &b_tensor.shape);
        let a_strides = get_aligned_strides(&self.shape, &broadcast_shape);
        let b_strides = get_aligned_strides(&b_tensor.shape, &broadcast_shape);

        let (output_shape, output_strides) = if partial_broadcast == true {
            let offset = broadcast_shape.len() - self.shape.len();
            let mut output_strides: Vec<usize> = vec![0; offset];
            output_strides.extend(self.strides.iter().cloned());

            (self.shape.clone(), output_strides)
        } else {
            (broadcast_shape.clone(), get_strides(&broadcast_shape))
        };

        let output_tensor = Tensor::from_cache(
            output_shape,
            tensor_name,
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let chunks = chunk_zipmap(
            broadcast_shape,
            self.data,
            a_strides,
            b_tensor.data,
            b_strides,
            output_tensor.data,
            output_strides,
        );
        operator.set_zipmap_chunk(chunks);
        self.operator_queue.borrow_mut().push(operator);
        output_tensor
    }

    pub fn matmul(
        &self,
        tensor2: &Tensor<T>,
        mut operator: Operator<T>,
        params: MatMulParams,
        sequence_length: usize,
        tensor_name: String,
    ) -> Self {
        let single_output_shape = vec![self.shape[0], tensor2.shape[0]];
        let output_shape = if sequence_length == 1 {
            single_output_shape.clone()
        } else {
            vec![
                sequence_length,
                single_output_shape[0],
                single_output_shape[1],
            ]
        };

        let output_tensor = Tensor::from_cache(
            output_shape.clone(),
            tensor_name,
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let chunks = chunk_matmul(self.data, tensor2.data, output_tensor.data, &params);

        if sequence_length == 1 {
            operator.set_zipmap_chunk(chunks);
        } else {
            let sequence_stride: usize = single_output_shape.iter().product();
            let mut expand_chunks = vec![];

            for i in 0..sequence_length {
                let offset = i * sequence_stride;
                for item in &chunks {
                    let mut temp = item.clone();
                    temp.2.ptr = unsafe { temp.2.ptr.add(offset) };
                    expand_chunks.push(temp);
                }
            }
            operator.set_zipmap_chunk(expand_chunks);
        }

        self.operator_queue.borrow_mut().push(operator);
        output_tensor
    }

    pub fn reduce(
        &self,
        sequences: *mut usize,
        sequence_length: usize,
        mut operator: Operator<T>,
        tensor_name: String,
        // cache: &mut Cache<T>,
        // operator_queue: &mut Vec<Operator<T>>,
    ) {
        /*
        let output_shape = vec![sequence_length, self.shape[0]];
        let output_tensor = Tensor::from_cache(
            output_shape.clone(),
            tensor_name,
            self.cache.clone(),
            self.operator_queue.clone(),
        ); */

        let chunks = chunk_reduce(
            self.shape.clone(),
            self.data,
            self.strides.clone(),
            sequences,
            vec![1],
        );
        let mut extended_chunks = vec![];
        for step in (0..self.shape[0] * sequence_length).step_by(self.shape[0]) {
            for (a_ptr, mut b_ptr) in chunks.iter().cloned() {
                unsafe {
                    b_ptr.ptr = b_ptr.ptr.add(step);
                    extended_chunks.push((a_ptr, b_ptr));
                }
            }
        }
        operator.set_reduce_chunk(extended_chunks);
        self.operator_queue.borrow_mut().push(operator);
        // output_tensor
    }

    fn _view(&self, shape: Vec<usize>, strides: Vec<usize>) -> Self {
        // let size: usize = shape.iter().product();
        Tensor {
            data: self.data,
            shape: shape.clone(),
            strides: strides,
            tensor_name: self.tensor_name.clone(),
            cache: self.cache.clone(),
            operator_queue: self.operator_queue.clone(),
        }
    }

    pub fn view(&self, shape: Vec<usize>) -> Self {
        let strides = get_strides(&shape);
        self._view(shape, strides)
    }

    pub fn permute(&self, dims: Vec<usize>) -> Self {
        let shape: Vec<usize> = dims
            .clone()
            .into_iter()
            .map(|index| self.shape[index])
            .collect();
        let strides: Vec<usize> = dims
            .clone()
            .into_iter()
            .map(|index| self.strides[index])
            .collect();
        // let maximum_shape: Vec<usize> = dims.clone().into_iter().map(|index| self.maximum_shape[index]).collect();
        // let tensor = self._view(shape, strides, tensor_name);
        // tensor.set_contiguous(false) ;
        Tensor {
            data: self.data,
            shape: shape,
            strides: strides,
            tensor_name: self.tensor_name.clone(),
            cache: self.cache.clone(),
            operator_queue: self.operator_queue.clone(),
        }
    }

    pub fn transpose(&mut self, index1: usize, index2: usize) -> Self {
        let mut dims: Vec<usize> = (0..self.shape.len()).collect();
        dims.swap(index1, index2);
        // self.set_contiguous(false);
        self.permute(dims)
    }

    /*
    pub fn colmul(&self, tensor2: &Tensor<T>, mut operator: Operator<T>, tensor_name: String) -> Self {
        let output_shape = [self.shape[..2].to_vec(), tensor2.shape[3..4].to_vec()].concat();
        let output_tensor = Tensor::from_cache(
            output_shape.clone(),
            tensor_name,
            self.cache.clone(),
            self.operator_queue.clone(),
        );
        let chunks = chunk_colmul(
            self.data,
            self.shape.clone(),
            self.strides.clone(),
            tensor2.data,
            tensor2.shape.clone(),
            tensor2.strides.clone(),
            output_tensor.data,
            output_tensor.shape.clone(),
            output_tensor.strides.clone(),
        );
        operator.set_zipmap_chunk(chunks);
        self.operator_queue.borrow_mut().push(operator);
        output_tensor
        //重点！
    }


    pub fn vecmul(&self, tensor2: &Tensor<T> ,mut  operator: Operator<T>, tensor_name: String) -> Self {
        let output_shape = [self.shape[..3].to_vec(), tensor2.shape[2..3].to_vec()].concat();
        let output_tensor = Tensor::from_cache(
            output_shape.clone(),
            tensor_name,
            self.cache.clone(),
            self.operator_queue.clone(),
        );
        let chunks = chunk_colmul(
            self.data,
            self.shape.clone(),
            self.strides.clone(),
            tensor2.data,
            tensor2.shape.clone(),
            tensor2.strides.clone(),
            output_tensor.data,
            output_tensor.shape.clone(),
            output_tensor.strides.clone(),
        );
        operator.set_zipmap_chunk(chunks);
        self.operator_queue.borrow_mut().push(operator);
        output_tensor
    }*/

    /*

    pub fn set_contiguous(&mut self, flag: bool) {
        self.is_contiguous = flag;
    }

    pub fn is_contiguous(&self) -> bool {
        self.is_contiguous
    }



    */
}

#[cfg(test)]
mod test {
    use approx::assert_ulps_eq;
    use nom::sequence;
    use num_cpus;
    use std::sync::Arc;
    use std::sync::Barrier;
    use std::thread;

    use super::*;
    /*

    use super::super::super::compiler::map::lookup_rms_map::LookupRMSMap;
    use super::super::super::compiler::map::softmax_map::SoftmaxMap;

    use super::super::super::compiler::mul::mat_mul::MatMulTrait;
    use super::super::super::compiler::zip_map::complex_zip::ComplexZipMap;
    use super::super::super::compiler::zip_map::add_rms_zip::AddRMSZipMap;
    use super::super::super::compiler::zip_map::add_zip::AddZipMap;
    use super::super::super::compiler::zip_map::silu_mul_zip::SiluZipMap;
    use super::super::super::compiler::mul::vec_mul::VecMul;
    use super::super::super::compiler::mul::col_mul::ColMul;
    */

    use super::super::super::compiler::map::rms_map::RMSMap;
    use super::super::super::compiler::mul::attention_mul::AttentionMul;
    use super::super::super::compiler::mul::mat_mul::MatMul;
    use super::super::super::compiler::operator::Operator;
    use super::super::super::compiler::reduce::argmax_reduce::ArgmaxReduce;
    use super::super::super::compiler::zip_map::complex_zip::ComplexZipMap;

    #[test]
    fn test_attention() {
        let head_size = 128;
        let head_num = 64;
        let batch_size = 16;
        let sequence_length = 256;

        let q_shape = vec![batch_size, head_num, head_size];
        let k_shape = vec![sequence_length, batch_size, head_num, head_size];
        let v_shape = vec![sequence_length, batch_size, head_num, head_size];
        let output_shape = q_shape.clone();

        let q_size = q_shape.iter().product();
        let k_size = k_shape.iter().product();
        let v_size = v_shape.iter().product();
        let output_size = output_shape.iter().product();

        let q_data: Vec<f32> = vec![1.0; q_size];
        let k_data: Vec<f32> = vec![1.0; k_size];
        let v_data: Vec<f32> = vec![1.0; v_size];
        let mut output_data: Vec<f32> = vec![0.0; output_size];

        let q_strides = get_strides(&q_shape);
        let k_strides = get_strides(&k_shape);
        let v_strides = get_strides(&v_shape);
        let output_strides = get_strides(&output_shape);

        let mut cache: Cache<f32> = Cache::new(std::collections::HashMap::new());
        let mut operator_queue: Vec<Operator<f32>> = Vec::new();
        let cache_rc = Rc::new(RefCell::new(cache));

        let q_tensor = Tensor::from_cache(
            q_shape.clone(),
            String::from("model.q_tensor.weight"),
            cache_rc.clone(),
            Rc::new(RefCell::new(operator_queue.clone())),
        );
        unsafe {
            q_tensor.data.copy_from(q_data.as_ptr(), q_data.len());
        }

        let k_tensor = Tensor::from_cache(
            k_shape.clone(),
            String::from("model.k_tensor.weight"),
            cache_rc.clone(),
            Rc::new(RefCell::new(operator_queue.clone())),
        );
        unsafe {
            k_tensor.data.copy_from(k_data.as_ptr(), k_data.len());
        }

        let v_tensor = Tensor::from_cache(
            v_shape.clone(),
            String::from("model.v_tensor.weight"),
            cache_rc.clone(),
            Rc::new(RefCell::new(operator_queue.clone())),
        );
        unsafe {
            v_tensor.data.copy_from(v_data.as_ptr(), v_data.len());
        }

        let mut operator = Operator::AttentionMul(AttentionMul::<f32>::new(
            head_size,
            head_num,
            k_strides[2],
            1.0,
            num_cpus::get(),
        ));

        let mut view_k_tensor = k_tensor.permute(vec![1, 2, 0, 3]);
        let mut view_v_tensor = v_tensor.permute(vec![1, 2, 0, 3]);

        let output_tensor = q_tensor.attention(
            &view_k_tensor,
            &view_v_tensor,
            operator,
            String::from("model.output_tensor.weight"),
        );

        let thread_num: usize = num_cpus::get();
        for i in 0..thread_num {
            output_tensor.operator_queue.borrow()[0].run(batch_size, sequence_length, i);
        }

        let result: Vec<f32> = vec![1.0; output_size];
        let output_slice = unsafe { std::slice::from_raw_parts(output_tensor.data, output_size) };
        assert_ulps_eq!(output_slice, &result[..], max_ulps = 4);
    }

    #[test]
    fn test_matmul_batch_size_1() {
        let batch_size = 8;
        let hidden_size = 16;
        let sequence_length = 1;

        let shape1 = vec![batch_size, hidden_size];
        let size1 = shape1.iter().product();
        let data1: Vec<f32> = vec![1.0; size1];

        let shape2 = vec![hidden_size, hidden_size];
        let size2 = shape2.iter().product();
        let data2: Vec<f32> = vec![1.0; size2];

        let output_shape = vec![batch_size, hidden_size];
        let size3 = output_shape.iter().product();
        let mut data3: Vec<f32> = vec![0.0; size3];
        let mut result = vec![0.0; size3];
        for i in 0..hidden_size {
            result[i] = hidden_size as f32;
        }

        let mut cache: Cache<f32> = Cache::new(std::collections::HashMap::new());
        let mut operator_queue: Vec<Operator<f32>> = Vec::new();
        let cache_rc = Rc::new(RefCell::new(cache));

        let tensor1 = Tensor::from_cache(
            shape1.clone(),
            String::from("model.tensor1.weight"),
            cache_rc.clone(),
            Rc::new(RefCell::new(operator_queue.clone())),
        );
        unsafe {
            tensor1.data.copy_from(data1.as_ptr(), data1.len());
        }

        let tensor2 = Tensor::from_cache(
            shape2.clone(),
            String::from("model.tensor2.weight"),
            cache_rc.clone(),
            Rc::new(RefCell::new(operator_queue.clone())),
        );
        unsafe {
            tensor2.data.copy_from(data2.as_ptr(), data2.len());
        }

        let params = MatMulParams {
            a_row: batch_size,
            b_row: hidden_size,
            column: hidden_size,
            a_row_step_macro: 1,
            b_row_step_macro: 1,
            column_step_macro: hidden_size,
            a_row_step_micro: 1,
            b_row_step_micro: 1,
        };
        let thread_num: usize = num_cpus::get();
        let barrier = Barrier::new(thread_num);
        let barrier_arc = Arc::new(barrier);
        let operator = Operator::MatMul(MatMul::<f32>::new(
            batch_size,
            hidden_size,
            hidden_size,
            1,
            1,
            hidden_size,
            1,
            1,
            1,
            thread_num,
            barrier_arc,
        ));

        let output_tensor = tensor1.matmul(
            &tensor2,
            operator,
            params,
            sequence_length,
            String::from("model.tensor3.weight"),
        );

        for i in 0..thread_num {
            output_tensor.operator_queue.borrow()[0].run(1, 0, i);
        }

        let output_slice = unsafe { std::slice::from_raw_parts(output_tensor.data, size3) };
        assert_ulps_eq!(output_slice, &result[..], max_ulps = 4);
    }

    #[test]
    fn test_matmul_sequence() {
        let batch_size = 8;
        let hidden_size = 16;
        let sequence_length = 16;
        let position_index = 8;

        let shape1 = vec![batch_size, hidden_size];
        let size1 = shape1.iter().product();
        let data1: Vec<f32> = vec![1.0; size1];

        let shape2 = vec![hidden_size, hidden_size];
        let size2 = shape2.iter().product();
        let data2: Vec<f32> = vec![1.0; size2];

        let output_shape = vec![sequence_length, batch_size, hidden_size];
        let size3 = output_shape.iter().product();
        let mut data3: Vec<f32> = vec![0.0; size3];

        let mut result = vec![0.0; size3];
        let offset = position_index * batch_size * hidden_size;
        for i in 0..hidden_size {
            result[i + offset] = hidden_size as f32;
        }

        let mut cache: Cache<f32> = Cache::new(std::collections::HashMap::new());
        let mut operator_queue: Vec<Operator<f32>> = Vec::new();
        let cache_rc = Rc::new(RefCell::new(cache));

        let tensor1 = Tensor::from_cache(
            shape1.clone(),
            String::from("model.tensor1.weight"),
            cache_rc.clone(),
            Rc::new(RefCell::new(operator_queue.clone())),
        );
        unsafe {
            tensor1.data.copy_from(data1.as_ptr(), data1.len());
        }

        let tensor2 = Tensor::from_cache(
            shape2.clone(),
            String::from("model.tensor2.weight"),
            cache_rc.clone(),
            Rc::new(RefCell::new(operator_queue.clone())),
        );
        unsafe {
            tensor2.data.copy_from(data2.as_ptr(), data2.len());
        }

        let params = MatMulParams {
            a_row: batch_size,
            b_row: hidden_size,
            column: hidden_size,
            a_row_step_macro: 1,
            b_row_step_macro: 1,
            column_step_macro: hidden_size,
            a_row_step_micro: 1,
            b_row_step_micro: 1,
        };
        let thread_num: usize = num_cpus::get();
        let barrier = Barrier::new(thread_num);
        let barrier_arc = Arc::new(barrier);
        let operator = Operator::MatMul(MatMul::<f32>::new(
            batch_size,
            hidden_size,
            hidden_size,
            1,
            1,
            hidden_size,
            1,
            1,
            sequence_length,
            thread_num,
            barrier_arc,
        ));

        let output_tensor = tensor1.matmul(
            &tensor2,
            operator,
            params,
            sequence_length,
            String::from("model.tensor3.weight"),
        );

        for i in 0..thread_num {
            output_tensor.operator_queue.borrow()[0].run(1, position_index, i);
        }

        let output_slice = unsafe { std::slice::from_raw_parts(output_tensor.data, size3) };
        assert_ulps_eq!(output_slice, &result[..], max_ulps = 4);
    }

    #[test]
    fn test_complex_mul_with_broadcast() {
        let head_size = 34;
        let head_num = 10;
        let batch_size = 10;
        let sequence_length = 10;

        let shape1 = vec![batch_size, head_num, head_size];
        let shape2 = vec![sequence_length, 1, 1, head_size];
        let broadcast_shape = get_broadcast_shape(&shape1, &shape2);

        let length: usize = broadcast_shape.iter().product();
        let input_strides1 = get_aligned_strides(&shape1, &broadcast_shape);
        let input_strides2 = get_aligned_strides(&shape2, &broadcast_shape);
        let output_strides = get_strides(&broadcast_shape);

        let length1: usize = shape1.iter().product();
        let length2: usize = shape2.iter().product();
        let input_data1: Vec<f32> = (1..=34).cycle().take(length1).map(|x| x as f32).collect();
        let input_data2: Vec<f32> = (1..=34).cycle().take(length2).map(|x| x as f32).collect();
        let mut output_data: Vec<f32> = vec![0.0; length];

        let expected: Vec<f32> = vec![
            -3.0, 4.0, -7.0, 24.0, -11.0, 60.0, -15.0, 112.0, -19.0, 180.0, -23.0, 264.0, -27.0,
            364.0, -31.0, 480.0, -35.0, 612.0, -39.0, 760.0, -43.0, 924.0, -47.0, 1104.0, -51.0,
            1300.0, -55.0, 1512.0, -59.0, 1740.0, -63.0, 1984.0, -67.0, 2244.0,
        ];

        let mut cache: Cache<f32> = Cache::new(std::collections::HashMap::new());
        let mut operator_queue: Vec<Operator<f32>> = Vec::new();
        let cache_rc = Rc::new(RefCell::new(cache));

        let tensor1 = Tensor::from_cache(
            shape1.clone(),
            String::from("model.layer.0.hidden_tensor"),
            cache_rc.clone(),
            Rc::new(RefCell::new(operator_queue.clone())),
        );
        unsafe {
            tensor1
                .data
                .copy_from(input_data1.as_ptr(), input_data1.len());
        }

        let tensor2 = Tensor::from_cache(
            shape2.clone(),
            String::from("model.tensor2.weight"),
            cache_rc.clone(),
            Rc::new(RefCell::new(operator_queue.clone())),
        );
        unsafe {
            tensor2
                .data
                .copy_from(input_data2.as_ptr(), input_data2.len());
        }

        let operator = Operator::ComplexZip(ComplexZipMap::<f32>::new(
            head_size,
            head_num,
            batch_size,
            num_cpus::get(),
        ));

        let output_tensor = tensor1.zip_mapv(
            &tensor2,
            operator,
            false,
            String::from("model.layer.0.self_attn.value_tensor"),
        );

        let thread_num: usize = num_cpus::get();
        for i in 0..thread_num {
            output_tensor.operator_queue.borrow()[0].run(batch_size, 1, i);
        }

        let output_slice = unsafe { std::slice::from_raw_parts(output_tensor.data, length) };
        assert_eq!(output_slice[3434..3468], expected);
    }

    #[test]
    fn test_complex_mul_with_partial_broadcast() {
        let head_size = 34;
        let head_num = 10;
        let batch_size = 10;
        let sequence_length = 10;

        let shape1 = vec![batch_size, head_num, head_size];
        let shape2 = vec![sequence_length, 1, 1, head_size];
        // let broadcast_shape = get_broadcast_shape(&shape1, &shape2);

        // let length: usize = broadcast_shape.iter().product();
        // let input_strides1 = get_aligned_strides(&shape1, &broadcast_shape);
        // let input_strides2 = get_aligned_strides(&shape2, &broadcast_shape);
        // let output_strides = get_strides(&broadcast_shape);

        let length1: usize = shape1.iter().product();
        let length2: usize = shape2.iter().product();
        let output_length: usize = length1;
        let input_data1: Vec<f32> = (1..=34).cycle().take(length1).map(|x| x as f32).collect();
        let input_data2: Vec<f32> = (1..=34).cycle().take(length2).map(|x| x as f32).collect();
        // let mut output_data: Vec<f32> = vec![0.0; output_length];

        let expected: Vec<f32> = vec![
            -3.0, 4.0, -7.0, 24.0, -11.0, 60.0, -15.0, 112.0, -19.0, 180.0, -23.0, 264.0, -27.0,
            364.0, -31.0, 480.0, -35.0, 612.0, -39.0, 760.0, -43.0, 924.0, -47.0, 1104.0, -51.0,
            1300.0, -55.0, 1512.0, -59.0, 1740.0, -63.0, 1984.0, -67.0, 2244.0,
        ];

        let mut cache: Cache<f32> = Cache::new(std::collections::HashMap::new());
        let mut operator_queue: Vec<Operator<f32>> = Vec::new();
        let cache_rc = Rc::new(RefCell::new(cache));

        let tensor1 = Tensor::from_cache(
            shape1.clone(),
            String::from("model.layer.0.hidden_tensor"),
            cache_rc.clone(),
            Rc::new(RefCell::new(operator_queue.clone())),
        );
        unsafe {
            tensor1
                .data
                .copy_from(input_data1.as_ptr(), input_data1.len());
        }

        let tensor2 = Tensor::from_cache(
            shape2.clone(),
            String::from("model.tensor2.weight"),
            cache_rc.clone(),
            Rc::new(RefCell::new(operator_queue.clone())),
        );
        unsafe {
            tensor2
                .data
                .copy_from(input_data2.as_ptr(), input_data2.len());
        }

        let operator = Operator::ComplexZip(ComplexZipMap::<f32>::new(
            head_size,
            head_num,
            batch_size,
            num_cpus::get(),
        ));

        let output_tensor = tensor1.zip_mapv(
            &tensor2,
            operator,
            true,
            String::from("model.layer.0.self_attn.value_tensor"),
        );

        let thread_num: usize = num_cpus::get();
        for i in 0..thread_num {
            output_tensor.operator_queue.borrow()[0].run(batch_size, 1, i);
        }

        let output_slice = unsafe { std::slice::from_raw_parts(output_tensor.data, output_length) };
        assert_eq!(output_slice[34..68], expected);
    }

    #[test]
    fn test_reduce() {
        let batch_size = 10;
        let vocab_size = 64;
        let sequence_length = 16;

        let shapes = vec![batch_size, vocab_size];
        let strides = vec![vocab_size, 1];
        let length: usize = shapes.iter().product();

        let input_data: Vec<f32> = (1..=vocab_size)
            .cycle()
            .take(vocab_size * batch_size)
            .map(|x| x as f32)
            .collect();
        let mut output_data: Vec<usize> = vec![0usize; batch_size * sequence_length];

        let mut cache: Cache<f32> = Cache::new(std::collections::HashMap::new());
        let mut operator_queue: Vec<Operator<f32>> = Vec::new();
        let cache_rc = Rc::new(RefCell::new(cache));

        let input_tensor = Tensor::from_cache(
            shapes.clone(),
            String::from("model.input_tensor.weight"),
            cache_rc.clone(),
            Rc::new(RefCell::new(operator_queue.clone())),
        );
        unsafe {
            input_tensor
                .data
                .copy_from(input_data.as_ptr(), input_data.len());
        }

        let mut operator = Operator::ArgmaxReduce(ArgmaxReduce::<f32>::new(
            vocab_size,
            batch_size,
            num_cpus::get(),
        ));
        let output_tensor = input_tensor.reduce(
            output_data.as_mut_ptr(),
            sequence_length,
            operator,
            String::from("model.output_tensor.weight"),
        );

        let thread_num: usize = num_cpus::get();
        for i in 0..thread_num {
            input_tensor.operator_queue.borrow()[0].run(batch_size, 0, i);
        }

        let result: Vec<usize> = vec![63; batch_size];
        assert_eq!(&output_data[..batch_size], &result[..]);
    }

    #[test]
    fn test_permute() {
        let shape = vec![3, 4, 5];
        let tensor = Tensor::<f32>::from_cache(
            shape,
            String::from("modelweight"),
            Rc::new(RefCell::new(Cache::new(std::collections::HashMap::new()))),
            Rc::new(RefCell::new(Vec::new())),
        );

        let size: usize = tensor.shape.iter().product();
        (0..size).for_each(|x| {
            unsafe {
                // *tensor.data.add(x) = (x+1) as f32);
                *tensor.data.add(x) = (x + 1) as f32;
            }
        });

        let m = tensor.permute(vec![1, 2, 0]);
        println!("{:?} {:?}", m.shape, m.strides);
    }

    #[test]
    fn test_rms_map() {
        let batch_size = 10; // 每次批处理 10 个元素
        let hidden_size = 18;
        let shape = vec![batch_size, hidden_size];
        let strides = vec![hidden_size, 1]; // 对应的步长
                                            // let length = shape.iter().product(); // 总元素数量

        let position_index = 0; // 起始位置，根据实际情况可以修改
        let cpu_num = num_cpus::get();
        let length = shape.iter().product();
        // 创建模拟的输入和输出数据
        //let input_data: Vec<f32> = (0..length).map(|x| x as f32).collect();
        let mut input_data: Vec<f32> = (1..=18).cycle().take(180).map(|x| x as f32).collect();

        let mut cache: Cache<f32> = Cache::new(std::collections::HashMap::new());
        let mut operator_queue: Vec<Operator<f32>> = Vec::new();
        let mut output_data: Vec<f32> = vec![0.0; length];
        let input_tensor = Tensor::from_cache(
            shape.clone(),
            String::from("model.input_tensor.weight"),
            Rc::new(RefCell::new(cache)),
            Rc::new(RefCell::new(operator_queue)),
        );

        unsafe {
            input_tensor
                .data
                .copy_from(input_data.as_ptr(), input_data.len());
        }

        let weight = vec![1.0f32; hidden_size];
        let eps = 1e-6;

        let mut operator = RMSMap::new(hidden_size, weight.as_ptr(), eps, cpu_num);

        // let mut output_data: Vec<f32> = vec![0.0; length];
        let result = [
            0.09238425642251968,
            0.18476851284503937,
            0.27715277671813965,
            0.36953702569007874,
            0.4619212746620178,
            0.5543055534362793,
            0.646689772605896,
            0.7390740513801575,
            0.831458330154419,
            0.9238425493240356,
            1.0162267684936523,
            1.1086111068725586,
            1.2009953260421753,
            1.293379545211792,
            1.3857638835906982,
            1.478148102760315,
            1.5705323219299316,
            1.662916660308838,
        ];

        let output_tensor = input_tensor.mapv(
            Operator::RMSMap(operator),
            String::from("model.output_tensor.weight"),
        );
        // 使用 chunk_map 函数创建块
        //let chunks = chunk_map(shape, strides, input_data.as_ptr(), output_data.as_mut_ptr());
        // 使用这些块和长度初始化 ArgmaxMap
        //input_tensor.operator_queue.borrow_mut()[0].set_map_chunk(chunks);
        let thread_num: usize = cpu_num;
        for i in 0..thread_num {
            input_tensor.operator_queue.borrow()[0].run(batch_size, 0, i);
        }

        // 如需打印输出数据，请取消以下注释
        let output_slice = unsafe { std::slice::from_raw_parts(output_tensor.data, length) };
        assert_ulps_eq!(output_slice[18..36], result, max_ulps = 4);
        //println!("{:?}", output_data);
    }

    /*
    #[test]
    fn test_softmax_map() {
        let batch_size = 10;
        let head_num = 10;
        let sequence_size = 18;
        let shapes = vec![batch_size,head_num, sequence_size];
        let strides = get_strides(&shapes);
        let length = shapes.iter().product(); // 总元素数量
        let position_index = 18; // 起始位置，根据实际情况可以修改
        let cpu_num = num_cpus::get();
        let input_data: Vec<f32> = (1..=18).cycle().take(1800).map(|x| x as f32).collect();
        let mut output_data: Vec<f32> = vec![0.0; length];
        let mut cache: Cache<f32> = Cache::new();
        let mut operator_queue: Vec<Operator<f32>> = Vec::new();
        let input_tensor = Tensor::from_cache(
            shapes.clone(),
            String::from("model.layers.0.self_attn.value_tensor"),
            Rc::new(RefCell::new(cache)),
            Rc::new(RefCell::new(operator_queue)),
        );
        unsafe {
            input_tensor
                .data
                .copy_from(input_data.as_ptr(), input_data.len());
        }
        let chunks = chunk_map(shapes, strides, input_data.as_ptr(), output_data.as_mut_ptr());
        let mut operator = Operator::SoftmaxMap(SoftmaxMap::new(head_num, cpu_num));
        input_tensor.mapv(
            operator,
            String::from("model.layers.0.self_attn.value_tensor"),
        );
        let result = [0.0012586231, 0.0017267587, 0.0023690138, 0.0032501512, 0.0044590216, 0.006117522, 0.00839289, 0.011514563, 0.015797319, 0.02167302, 0.02973414, 0.040793534, 0.0559664, 0.0767827, 0.105341434, 0.14452243, 0.19827652, 0.27202395];
        input_tensor.operator_queue.borrow_mut()[0].set_map_chunk(chunks);

        let thread_num: usize = cpu_num;
            // 运行 ArgmaxMap 操作
        for i in 0..thread_num {
            input_tensor.operator_queue.borrow()[0].run(batch_size, position_index, i);
        }
            // 如需打印输出数据，请取消以下注释
        assert_ulps_eq!(output_data[0..18], result, max_ulps=4);
        //println!("{:?}", output_data);

        // println!("{:?}", output);
    }
    #[test]
    fn test_rms_map() {

        let batch_size = 10; // 每次批处理 10 个元素
        let hidden_size = 18;
        let vocab_size = 10;
        let cpu_num = num_cpus::get();

        let shapes = vec![batch_size, hidden_size];
        let strides = vec![hidden_size, 1]; // 对应的步长
        let length = shapes.iter().product(); // 总元素数量
        let sequence_length = 16;
        let position = 0; // 起始位置，根据实际情况可以修改

        // 创建模拟的输入和输出数据
        let input_data: Vec<f32> = (1..=hidden_size).cycle().take(length).map(|x| x as f32).collect();
        let sequences = vec![1; sequence_length];
        let word_embedding: Vec<f32> = (1..=18).cycle().take(vocab_size * hidden_size).map(|x| x as f32).collect();
        let mut cache: Cache<f32> = Cache::new();
        let mut operator_queue: Vec<Operator<f32>> = Vec::new();
        let input_tensor = Tensor::from_cache(
            shapes.clone(),
            String::from("model.layers.0.self_attn.value_tensor"),
            Rc::new(RefCell::new(cache)),
            Rc::new(RefCell::new(operator_queue)),
        );
        unsafe {
            input_tensor
                .data
                .copy_from(input_data.as_ptr(), input_data.len());
        }
        let weight = vec![1.0f32; length];
        let eps = 1e-6;
        let mut output_data: Vec<f32> = vec![0.0; length];

        // 使用 chunk_map 函数创建块
        let chunks = chunk_map(shapes, strides, input_data.as_ptr(), output_data.as_mut_ptr());
        // 使用这些块和长度初始化 ArgmaxMap
        let mut Operator = Operator::LookupRMSMap(LookupRMSMap::new(hidden_size,
                                    weight.as_ptr(),
                                    eps,
                                    cpu_num,
                                    word_embedding.as_ptr(),
                                    sequences.as_ptr(),
                                    hidden_size,
                                    batch_size
                                ));
        input_tensor.mapv(
            Operator,
            String::from("model.layers.0.self_attn.value_tensor"),
        );
        let result = [0.09238425642251968,
        0.18476851284503937,
        0.27715277671813965,
        0.36953702569007874,
        0.4619212746620178,
        0.5543055534362793,
        0.646689772605896,
        0.7390740513801575,
        0.831458330154419,
        0.9238425493240356,
        1.0162267684936523,
        1.1086111068725586,
        1.2009953260421753,
        1.293379545211792,
        1.3857638835906982,
        1.478148102760315,
        1.5705323219299316,
        1.662916660308838];
        input_tensor.operator_queue.borrow_mut()[0].set_map_chunk(chunks);

        let thread_num: usize = cpu_num;
        for i in 0..thread_num {
            input_tensor.operator_queue.borrow()[0].run(batch_size, position, i);
        }

            // 如需打印输出数据，请取消以下注释
        assert_ulps_eq!(output_data[18..36], result, max_ulps=4);
        //println!("{:?}", output_data);
    }
    #[test]
    fn test_col_mul() {
        let head_size = 128;
        let head_num  = 64;
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
        let mut cache: Cache<f32> = Cache::new();
        let mut operator_queue: Vec<Operator<f32>> = Vec::new();
        let cache_rc = Rc::new(RefCell::new(cache));
        let input_tensor = Tensor::from_cache(
            shape1.clone(),
            String::from("model.layers.0.self_attn.value_tensor"),
            cache_rc.clone(),
            Rc::new(RefCell::new(operator_queue.clone())),
        );
        unsafe {
            input_tensor
                .data
                .copy_from(data1.as_ptr(), data1.len());
        }
        let input_tensor2 = Tensor::from_cache(
            shape2.clone(),
            String::from("model.layers.0.self_attn.value_tensor"),
            cache_rc.clone(),
            Rc::new(RefCell::new(operator_queue)),
        );
        unsafe {
            input_tensor2
                .data
                .copy_from(data2.as_ptr(), data2.len());
        }
        let result = vec![sequence_length as f32; size3];

        let chunks = chunk_colmul(
            data1.as_ptr(),
            shape1,
            strides1,
            data2.as_ptr(),
            shape2,
            strides2,
            data3.as_mut_ptr(),
            shape3,
            strides3);

        let thread_num: usize = num_cpus::get();
         let mut operator = Operator::ColMul(ColMul::<f32>::new( head_size, head_num,thread_num));
        input_tensor.colmul(
            &input_tensor2,
            operator,
            String::from("model.layers.0.self_attn.value_tensor"),
        );


        input_tensor.operator_queue.borrow_mut()[0].set_zipmap_chunk(chunks);
        for i in 0..thread_num {
            input_tensor.operator_queue.borrow()[0].run(batch_size, sequence_length,i);
        }
        assert_ulps_eq!(data3[..], result[..], max_ulps=4);
        // println!("{:?}", output);
    }
    #[test]
    fn test_chunk_vec() {

        let head_size = 128;
        let head_num  = 64;
        let batch_size = 16;
        let sequence_length = 256;

        let q_shape = vec![batch_size, head_num, 1, head_size];
        let q_size = q_shape.iter().product();
        let q_data: Vec<f32> = vec![1.0; q_size];
        let q_strides = get_strides(&q_shape);


        let k_shape = vec![batch_size, head_num, sequence_length,  head_size];
        let k_size = k_shape.iter().product();
        let k_data: Vec<f32> = vec![1.0; k_size];
        let k_strides = get_strides(&k_shape);

        let s_shape = vec![batch_size , head_num, 1, sequence_length];
        let s_size = s_shape.iter().product();
        let mut s_data: Vec<f32> = vec![0.0 ; s_size];
        let s_strides = get_strides(&s_shape);
        let mut cache: Cache<f32> = Cache::new();
        let mut operator_queue: Vec<Operator<f32>> = Vec::new();
        let cache_rc = Rc::new(RefCell::new(cache));
        let result = vec![head_size as f32; s_size];
        let input_tensor = Tensor::from_cache(
            q_shape.clone(),
            String::from("model.layers.0.self_attn.value_tensor"),
            cache_rc.clone(),
            Rc::new(RefCell::new(operator_queue.clone())),
        );
        unsafe {
            input_tensor
                .data
                .copy_from(q_data.as_ptr(), q_data.len());
        }
        let input_tensor2 = Tensor::from_cache(
            k_shape.clone(),
            String::from("model.layers.0.self_attn.value_tensor"),
            cache_rc.clone(),
            Rc::new(RefCell::new(operator_queue)),
        );
        unsafe {
            input_tensor2
                .data
                .copy_from(k_data.as_ptr(), k_data.len());
        }
        let chunks = chunk_vecmul(
            q_data.as_ptr(),
            q_shape,
            q_strides,
            k_data.as_ptr(),
            k_shape,
            k_strides,
            s_data.as_mut_ptr(),
            s_shape,
            s_strides);

        let thread_num: usize = num_cpus::get();
        let mut operator = Operator::VecMul(VecMul::<f32>::new( head_size, head_num, sequence_length,thread_num));
        input_tensor.colmul(
            &input_tensor2,
            operator,
            String::from("model.layers.0.self_attn.value_tensor"),
        );
        input_tensor.operator_queue.borrow_mut()[0].set_zipmap_chunk(chunks);
        for i in 0..thread_num {
            input_tensor.operator_queue.borrow_mut()[0].run(batch_size, sequence_length,i);
        }

        assert_ulps_eq!(s_data[..], result[..], max_ulps=4);
    }

    #[test]
    fn test_add_zip() {
        let shapes = vec![10, 18];
        let input_strides1 = vec![18, 1];
        let input_strides2 = vec![18, 1];
        let output_strides = vec![18, 1];

        let length = shapes.iter().product(); // 总元素数量
        let batch_size = 10; // 每次批处理 10 个元素
        let position_size = 0; // 起始位置，根据实际情况可以修改

            // 创建模拟的输入和输出数据
        //let input_data: Vec<f32> = (0..length).map(|x| x as f32).collect();
        let input_data1: Vec<f32> = (0..=17).cycle().take(180).map(|x| x as f32).collect();
        let input_data2:Vec<f32> =vec![1.0;length];
        let results:Vec<f32>=(1..=18).cycle().take(180).map(|x| x as f32).collect();
        //println!("{:?}", input_data2);
        let mut output_data: Vec<f32> = vec![0.0; length];
        let mut cache: Cache<f32> = Cache::new();
        let mut operator_queue: Vec<Operator<f32>> = Vec::new();
        let cache_rc = Rc::new(RefCell::new(cache));
        let input_tensor = Tensor::from_cache(
            shapes.clone(),
            String::from("model.layers.0.self_attn.value_tensor"),
            cache_rc.clone(),
            Rc::new(RefCell::new(operator_queue.clone())),
        );
        unsafe {
            input_tensor
                .data
                .copy_from(input_data1.as_ptr(), input_data1.len());
        }
        let input_tensor2 = Tensor::from_cache(
            shapes.clone(),
            String::from("model.layers.0.self_attn.value_tensor"),
            cache_rc.clone(),
            Rc::new(RefCell::new(operator_queue)),
        );
        unsafe {
            input_tensor2
                .data
                .copy_from(input_data2.as_ptr(), input_data2.len());
        }
            // 使用 chunk_map 函数创建块
        let chunks = chunk_zipmap(shapes,  input_data1.as_ptr(),input_strides1,input_data2.as_ptr(),input_strides2, output_data.as_mut_ptr(),output_strides);
            // 使用这些块和长度初始化 ArgmaxMap
        let thread_num: usize = num_cpus::get();
        let mut operator = Operator::AddZipMap(AddZipMap::new(18, thread_num));
        input_tensor.zip_mapv(
            &input_tensor2,
            operator,
            false,
            String::from("model.layers.0.self_attn.value_tensor"),
        );
        input_tensor.operator_queue.borrow_mut()[0].set_zipmap_chunk(chunks);

        for i in 0..thread_num {
            input_tensor.operator_queue.borrow()[0].run(batch_size, position_size,i);
        }

            // 如需打印输出数据，请取消以下注释
        assert_ulps_eq!(output_data[0..180], results[0..180], max_ulps=4);
        println!("{:?}", output_data);

        // println!("{:?}", output);
    }

    #[test]
    fn test_silu() {
        let batch_size = 10;
        let hidden_size = 19;
        let shapes = vec![batch_size, hidden_size];
        let input_strides1 = get_strides(&shapes);
        //println!("{:?}", input_strides1);
        let input_strides2 = input_strides1.clone();
        let output_strides = input_strides1.clone();
        let length = shapes.iter().product();
        let input_data1: Vec<f32> = vec![2.1671206951141357,
        1.4490455389022827,
        -2.002431631088257,
        0.5662149786949158,
        0.3909946382045746,
        0.9437483549118042,
        -0.37030690908432007,
        0.7542704939842224,
        0.5875813961029053,
        1.6026240587234497,
        2.2485475540161133,
        -0.6622593402862549,
        -0.0015666020335629582,
        -0.5069465041160583,
        -0.37254711985588074,
        0.4420417249202728,
        -0.9305257201194763,
        0.5145581364631653,
        0.6260590553283691
        ].repeat(10);
        let input_data2: [f32; 190] = [1.0; 190];
        let mut output_data: Vec<f32> = vec![0.0; length];
        let mut cache: Cache<f32> = Cache::new();
        let mut operator_queue: Vec<Operator<f32>> = Vec::new();
        let cache_rc = Rc::new(RefCell::new(cache));
        let input_tensor = Tensor::from_cache(
            shapes.clone(),
            String::from("model.layers.0.self_attn.value_tensor"),
            cache_rc.clone(),
            Rc::new(RefCell::new(operator_queue.clone())),
        );
        unsafe {
            input_tensor
                .data
                .copy_from(input_data1.as_ptr(), input_data1.len());
        }
        let input_tensor2 = Tensor::from_cache(
            shapes.clone(),
            String::from("model.layers.0.self_attn.value_tensor"),
            cache_rc.clone(),
            Rc::new(RefCell::new(operator_queue)),
        );
        unsafe {
            input_tensor2
                .data
                .copy_from(input_data2.as_ptr(), input_data2.len());
        }
        let chunks = chunk_zipmap(shapes, input_data1.as_ptr(), input_strides1, input_data2.as_ptr(), input_strides2, output_data.as_mut_ptr(), output_strides);
        let thread_num: usize = num_cpus::get();
        let mut operator =Operator::SiluMulZipMap(SiluZipMap::new(hidden_size, thread_num));
        input_tensor.zip_mapv(
            &input_tensor2,
            operator,
            false,
            String::from("model.layers.0.self_attn.value_tensor"),
        );
        input_tensor.operator_queue.borrow_mut()[0].set_zipmap_chunk(chunks);

        for i in 0..thread_num {
            input_tensor.operator_queue.borrow_mut()[0].run(batch_size ,1usize,i);
        }
        let result = [1.9444659948349,
        1.1735117435455322,
        -0.23818494379520416,
        0.36118248105049133,
        0.23323695361614227,
        0.6793630719184875,
        -0.15125809609889984,
        0.5129857659339905,
        0.3777032196521759,
        1.3339999914169312,
        2.033867835998535,
        -0.22532200813293457,
        -0.0007826874498277903,
        -0.1905660629272461,
        -0.15197153389453888,
        0.269090861082077,
        -0.2631694972515106,
        0.32204875349998474,
        0.4079371392726898].repeat(10);
        //println!("{:?}",result.len());
        assert_ulps_eq!(output_data[..], result, max_ulps=4);
        // println!("{:?}", output);
    }*/
}
