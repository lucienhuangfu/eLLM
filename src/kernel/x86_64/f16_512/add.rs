use std::arch::x86_64::{_mm512_add_ph, _mm512_load_ph, _mm512_store_ph};
use std::f16;
#[inline(always)]
pub fn add(input_ptr1: *const f16, input_ptr2: *const f16, output_ptr: *mut f16, length: usize) {
    // println!("add");
    unsafe {
        // let mut i = 0;
        for i in (0..length).step_by(32) {
            let a = _mm512_load_ph(input_ptr1.add(i));
            let b = _mm512_load_ph(input_ptr2.add(i));
            let c = _mm512_add_ph(a, b);
            // _mm512_storeu_ph(output_ptr.add(i) as *mut _, c);
            _mm512_store_ph(output_ptr.add(i), c);
            // i += 32; // Each __m512i register can hold 32 f16 values
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;
    use std::alloc::{alloc, dealloc, Layout};
    use std::ptr;

    use super::*;

    #[test]
    fn test_add_() {
        let layout = Layout::from_size_align(128 * std::mem::size_of::<f16>(), 64).unwrap();
        unsafe {
            let a_ptr = alloc(layout) as *mut f16;
            let b_ptr = alloc(layout) as *mut f16;
            let c_ptr = alloc(layout) as *mut f16;

            for i in 0..128 {
                ptr::write(a_ptr.add(i), 1.2);
                ptr::write(b_ptr.add(i), 2.6);
                ptr::write(c_ptr.add(i), 0.0);
            }

            add(a_ptr, b_ptr, c_ptr, 128);

            for i in 0..128 {
                // assert_ulps_eq!(*c_ptr.add(i) as f32, 3.8, max_ulps = 4);
                println!("{}, {}", *c_ptr.add(i) as f32, 3.8);
                assert!(f16::abs(*c_ptr.add(i) - 3.8) < 1e-6);
            }

            dealloc(a_ptr as *mut u8, layout);
            dealloc(b_ptr as *mut u8, layout);
            dealloc(c_ptr as *mut u8, layout);
        }
    }

    /*
    #[test]
    fn test_add() {
        let a: Vec<f16> = vec![1.2; 128];
        let b: Vec<f16> = vec![2.6; 128];
        let mut c: Vec<f16> = vec![0.0; 128];
        let result: Vec<f16> = vec![3.8; 128];
        add(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), a.len());
        // println!("{:?}", c);
        for i in 0..128 {
            assert_ulps_eq!(c[i] as f32, result[i] as f32, max_ulps=4);
        }
    } */
}
