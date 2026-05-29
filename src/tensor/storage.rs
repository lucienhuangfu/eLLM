use std::ops::{AddAssign, Neg, Sub};

use crate::num_traits::NegInfinity;
use crate::mem_mgr::allocator::AlignedBox;
use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::operators::operator::Operator;
use crate::tensor::get_strides;

use super::GlobalOperatorQueue;

#[derive(Clone)]
pub struct Tensor<T>
where
    T: Copy + PartialOrd,
{
    pub data: *mut T,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub tensor_name: String,
}

pub(super) fn leaked_aligned_ptr<T: Copy>(length: usize, value: T) -> *mut T {
    let buffer = AlignedBox::allocate_init(length, value);
    let ptr = buffer.as_mut_ptr();
    std::mem::forget(buffer);
    ptr
}

impl<T> Tensor<T>
where
    T: Copy
        + PartialOrd
        + Default
        + Sub<Output = T>
        + Neg<Output = T>
        + NegInfinity
        + AddAssign
        + GlobalMemPool
        + GlobalOperatorQueue,
{
    pub(super) fn from_pool(shape: Vec<usize>, tensor_name: String) -> Self {
        let data = T::with_global(|pool| pool.get(&tensor_name, &shape));
        let strides = get_strides(&shape);
        Tensor {
            data,
            shape,
            strides,
            tensor_name,
        }
    }

    #[inline]
    pub(super) fn enqueue(operator: Operator<T>) {
        T::with_operator_queue(|queue| queue.push(operator));
    }

    #[inline]
    pub(super) fn output_tensor(shape: Vec<usize>, scope_name: &str) -> Self {
        Self::from_mem_pool(shape, format!("{}.output", scope_name))
    }

    #[inline]
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    #[inline]
    pub fn row_count(&self) -> usize {
        let last_dim = self.last_dim();
        self.element_count() / last_dim
    }

    #[inline]
    pub fn last_dim(&self) -> usize {
        *self
            .shape
            .last()
            .expect("Tensor shape must have at least one dimension")
    }

    pub fn from_mem_pool(shape: Vec<usize>, tensor_name: String) -> Self {
        Self::from_pool(shape, tensor_name)
    }

    pub fn from_vec(shape: Vec<usize>, data: Vec<T>, tensor_name: String) -> Self {
        let length: usize = shape.iter().product();
        assert!(
            data.len() == length,
            "Tensor::from_vec length mismatch: shape product {} != data len {}",
            length,
            data.len()
        );
        let v = Self::from_mem_pool(shape, tensor_name);
        unsafe {
            v.data.copy_from_nonoverlapping(data.as_ptr(), data.len());
        }
        v
    }

    pub fn permute(&self, dims: Vec<usize>) -> Self {
        assert_eq!(
            dims.len(),
            self.shape.len(),
            "Tensor::permute rank mismatch: dims {:?}, shape {:?}",
            dims,
            self.shape
        );

        let shape: Vec<usize> = dims.iter().map(|&index| self.shape[index]).collect();
        let strides: Vec<usize> = dims.iter().map(|&index| self.strides[index]).collect();
        Tensor {
            data: self.data,
            shape,
            strides,
            tensor_name: self.tensor_name.clone(),
        }
    }

    pub fn transpose(&mut self, index1: usize, index2: usize) -> Self {
        let mut dims: Vec<usize> = (0..self.shape.len()).collect();
        dims.swap(index1, index2);
        // self.set_contiguous(false);
        self.permute(dims)
    }

    pub fn view(&self, shape: Vec<usize>) -> Self {
        assert_eq!(
            self.element_count(),
            shape.iter().product::<usize>(),
            "Tensor::view element count mismatch: source {:?}, target {:?}",
            self.shape,
            shape
        );
        let strides = get_strides(&shape);
        self._view(shape, strides)
    }

    pub fn zeros(shape: Vec<usize>, tensor_name: String) -> Self {
        Self::from_mem_pool(shape, tensor_name)
    }

    fn _view(&self, shape: Vec<usize>, strides: Vec<usize>) -> Self {
        Tensor {
            data: self.data,
            shape,
            strides,
            tensor_name: self.tensor_name.clone(),
        }
    }
}

unsafe impl<T: Copy + Default + Send + Sync + PartialOrd> Send for Tensor<T> {}
unsafe impl<T: Copy + Default + Send + Sync + PartialOrd> Sync for Tensor<T> {}
