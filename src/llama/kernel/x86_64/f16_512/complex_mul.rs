use std::arch::x86_64::{_mm512_fmul_pch, _mm512_load_ph, _mm512_store_ph};
use std::f16;
#[inline(always)]
pub fn complex_mul(
    input_ptr1: *const f16,
    input_ptr2: *const f16,
    output_ptr: *mut f16,
    length: usize,
) {
    unsafe {
        for (ptr1, ptr2, ptr3) in (0..length)
            .step_by(32)
            .map(|x| (input_ptr1.add(x), input_ptr2.add(x), output_ptr.add(x)))
        {
            let x = _mm512_load_ph(ptr1); //load 32 f16 numbers from ptr1 that represent 16 complex numbers
            let y = _mm512_load_ph(ptr2); //load 32 f16 numbers from ptr2 that represent 16 complex numbers
            let z = _mm512_fmul_pch(x, y);
            _mm512_store_ph(ptr3, z);
        }

        /*
        //complex multiplication using _mm512_cmul_ph
        let rem = length % 32;
        let length2 = length - rem;
        if rem != length {
            for (ptr1, ptr2, ptr3) in (0..length2).step_by(32).map(|x| (input_ptr1.add(x), input_ptr2.add(x), output_ptr.add(x))) {
                let x = _mm512_load_ph(ptr1);//load 32 f16 numbers from ptr1 that represent 16 complex numbers
                let y = _mm512_load_ph(ptr2);//load 32 f16 numbers from ptr2 that represent 16 complex numbers
                let z = _mm512_fmul_pch(x, y);
                _mm512_store_ph(ptr3, z);
            }
        }

        if rem != 0 {
            for i in (length2..length).step_by(2) {
                let a = input_ptr1.add(i).read();
                let b = input_ptr1.add(i + 1).read();
                let c = input_ptr2.add(i).read();
                let d = input_ptr2.add(i + 1).read();

                let x = a * c - b * d;
                let y = a * d + b * c;
                output_ptr.add(i).write(x);
                output_ptr.add(i + 1).write(y);

            }
        } */
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::allocator::allocate_init;
    use std::ptr;

    #[test]
    fn test_complexmul() {
        //each complex number is represented by two f16 numbers that is next to each other in memory
        //34 f16 numbers represent 16 complex numbers
        /*
        let input1: Vec<f16> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        ]; */

        let length = 32;
        let input1 = allocate_init::<f16>(length, 0.0);
        for i in 1..=length {
            unsafe {
                ptr::write(input1.wrapping_add(i - 1), i as f16);
            }
        }

        let input2 = allocate_init::<f16>(length, 0.0);
        for i in 1..=length {
            unsafe {
                ptr::write(input2.wrapping_add(i - 1), i as f16);
            }
        }

        let output = allocate_init::<f16>(length, 0.0);

        complex_mul(input1, input2, output, length);
        let expected: Vec<f16> = vec![
            1.0 * 1.0 - 2.0 * 2.0,
            1.0 * 2.0 + 2.0 * 1.0,
            3.0 * 3.0 - 4.0 * 4.0,
            3.0 * 4.0 + 4.0 * 3.0,
            5.0 * 5.0 - 6.0 * 6.0,
            5.0 * 6.0 + 6.0 * 5.0,
            7.0 * 7.0 - 8.0 * 8.0,
            7.0 * 8.0 + 8.0 * 7.0,
            9.0 * 9.0 - 10.0 * 10.0,
            9.0 * 10.0 + 10.0 * 9.0,
            11.0 * 11.0 - 12.0 * 12.0,
            11.0 * 12.0 + 12.0 * 11.0,
            13.0 * 13.0 - 14.0 * 14.0,
            13.0 * 14.0 + 14.0 * 13.0,
            15.0 * 15.0 - 16.0 * 16.0,
            15.0 * 16.0 + 16.0 * 15.0,
            17.0 * 17.0 - 18.0 * 18.0,
            17.0 * 18.0 + 18.0 * 17.0,
            19.0 * 19.0 - 20.0 * 20.0,
            19.0 * 20.0 + 20.0 * 19.0,
            21.0 * 21.0 - 22.0 * 22.0,
            21.0 * 22.0 + 22.0 * 21.0,
            23.0 * 23.0 - 24.0 * 24.0,
            23.0 * 24.0 + 24.0 * 23.0,
            25.0 * 25.0 - 26.0 * 26.0,
            25.0 * 26.0 + 26.0 * 25.0,
            27.0 * 27.0 - 28.0 * 28.0,
            27.0 * 28.0 + 28.0 * 27.0,
            29.0 * 29.0 - 30.0 * 30.0,
            29.0 * 30.0 + 30.0 * 29.0,
            31.0 * 31.0 - 32.0 * 32.0,
            31.0 * 32.0 + 32.0 * 31.0,
        ];

        println!("{:?}", output);
        println!("{:?}", expected);

        for j in 0..length {
            unsafe {

                // assert!(f16::abs(output.wrapping_add(j).read() - expected[j]) < 1e-6);
            }
        }
    }
}
