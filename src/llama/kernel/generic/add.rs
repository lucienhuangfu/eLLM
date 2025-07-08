use std::ptr;
use std::ops::Add;

pub fn add<T>(input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T, length: usize) 
where
    T: Copy + Add<Output = T> ,
{
    // println!("code add");
    unsafe {
        for (input1, input2, output) in (0..length).map(|x| (input_ptr1.add(x), input_ptr2.add(x), output_ptr.add(x))) {
            let y = *input1 + *input2;
            ptr::write(output, y);
        } 
    }
}


#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;
    use super::*;

    #[test]
    fn test_add() {
        let v1 = [1.0; 19];
        let v2 = [2.0; 19];
        let mut v3 = [0.0; 19];
        let result = [3.0; 19];
        add(v1.as_ptr(), v2.as_ptr(), v3.as_mut_ptr(), v1.len());
        assert_ulps_eq!(v3[..], result[..], max_ulps=4);
    }
}