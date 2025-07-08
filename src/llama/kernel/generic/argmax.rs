
pub fn argmax<T>(input_ptr: *const T, output_ptr: *mut usize, length: usize) 
where
    T: Copy + PartialOrd,
{
    unsafe {          
        *output_ptr = (0..length).map(|x| (x, *(input_ptr.add(x)))).max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(index, _)| index).unwrap();
    }
} 

#[cfg(test)]
mod test {
    use super::*;
    
    #[test]
    fn test_argmax() {
        let v: Vec<f32> = vec![2.0, 3.0, 1.0, 5.0, 1.0, 4.0];
        let mut index: usize = 0;
        let result: usize = 3;
        argmax(v.as_ptr(), &mut index, v.len());
    
        assert_eq!(result, index);
    }
}