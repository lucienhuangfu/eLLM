use std::ptr;
use std::ops::{Add, Mul, Sub};
pub fn complex_mul<T>(input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T, length: usize) 
where
    T: Copy + Add<Output = T>  + Sub<Output = T> + Mul<Output = T>,
{//f16
    for i in (0..length).step_by(2) {
        unsafe {
            let a = input_ptr1.add(i).read();
            let b = input_ptr1.add(i + 1).read();
            let c = input_ptr2.add(i).read();
            let d = input_ptr2.add(i + 1).read();

            let e = a*c - b*d;
            let f = a*d + b*c;
            output_ptr.add(i).write(e);
            output_ptr.add(i + 1).write(f);
        }
    }
}

#[cfg(test)]
mod tests{

    use super::*;
    // use std::f16;
    //use crate::kernel::asmsimd::*;

    #[test]
    fn test_complexmul(){
        //each complex number is represented by two f16 numbers that is next to each other in memory
        //34 f16 numbers represent 17 complex numbers
        let input1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0];
        let input2 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0];
        //f32 to f16
        let input1: Vec<f32> = input1.into_iter().map(|x| x).collect();
        let input2: Vec<f32> = input2.into_iter().map(|x| x).collect();
        let mut output = vec![0.0; input1.len()];
        complex_mul(input1.as_ptr(), input2.as_ptr(), output.as_mut_ptr(), input1.len());
        let expected = vec![-3.0, 4.0, -7.0, 24.0, -11.0, 60.0, -15.0, 112.0, -19.0, 180.0, -23.0, 264.0, -27.0, 364.0, -31.0, 480.0, -35.0, 612.0, -39.0, 760.0, -43.0, 924.0, -47.0, 1104.0, -51.0, 1300.0, -55.0, 1512.0, -59.0, 1740.0, -63.0, 1984.0, -67.0, 2244.0];
        let expected: Vec<f32> = expected.into_iter().map(|x| x).collect();
        //println!("{:?}", output);
        //println!("{:?}", expected);
        assert_eq!(output, expected);

    }
}