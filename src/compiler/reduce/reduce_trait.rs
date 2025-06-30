// use std::ptr;

pub trait ReduceTrait<T>  
// where T: Copy + PartialOrd,
{
    fn compute(&self, 
            input_ptr: *const T, 
            output_ptr: *mut usize,
            length: usize
    );
}

//unsafe impl Send for MapTrait {}
//unsafe impl Sync for MapTrait {}















