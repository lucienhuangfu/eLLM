use std::arch::x86_64::{
    __m512h, _mm512_cmp_ph_mask, _mm512_loadu_ph, _mm512_max_ph, _mm512_set1_ph, _mm512_storeu_ph,
};

use std::f16;

#[inline(always)]
pub fn argmax(input_ptr: *const f16, output_ptr: *mut usize, length: usize) {
    unsafe {
        let mut max: f16 = f16::MIN; // the max value
        let mut idx: usize = 0; // the index of the max value
        let n_mod_128: usize = length & 127;
        let boundary: usize = length - n_mod_128;

        // Register_00: store the max value
        let mut p: __m512h = _mm512_set1_ph(f16::MIN);

        if boundary == 0 {
            for i in 0..length {
                if (*input_ptr.add(i)) > max {
                    max = *input_ptr.add(i);
                    idx = i;
                }
            }
            *output_ptr = idx;
            return;
        }

        for i in (0..boundary).step_by(128) {
            let y1 = _mm512_loadu_ph(input_ptr.add(i) as *const _);

            let y2 = _mm512_loadu_ph(input_ptr.add(i + 32) as *const _);
            let y3 = _mm512_loadu_ph(input_ptr.add(i + 64) as *const _);
            let y4 = _mm512_loadu_ph(input_ptr.add(i + 96) as *const _);

            let y1 = _mm512_max_ph(y1, y2);
            let y3 = _mm512_max_ph(y3, y4);
            let y1 = _mm512_max_ph(y1, y3);

            let mask = _mm512_cmp_ph_mask::<0x01>(p, y1);

            if mask != 0 {
                idx = i;
                for j in i..i + 128 {
                    max = if *input_ptr.add(j) > max {
                        *input_ptr.add(j)
                    } else {
                        max
                    }
                }
                p = _mm512_set1_ph(max);
            }
        }

        // Find the max value in the max block
        for i in idx..idx + 128 {
            if *input_ptr.add(i) == max {
                idx = i;
                break;
            }
        }

        // Find the max over the remainder of the array
        for i in boundary..length {
            if *input_ptr.add(i) > max {
                max = *input_ptr.add(i);
                idx = i;
            }
        }
        *output_ptr = idx;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use std::mem;
    use std::ptr;
    // use std::f16;

    #[test]
    fn test_argmax() {
        //let v randomly initialized
        let mut rng = rand::thread_rng();
        let a = rng.gen_range(0.0f32..10.0f32);

        let v: Vec<f16> = (0..1000).map(|x| x as f16).collect();
        // let v: Vec<f16> = v.into_iter().map(|x| x as f16).collect();
        let vptr = v.as_ptr();
        let mut output_ptr: usize = 0;
        // ptr::null();
        argmax(vptr, &mut output_ptr, v.len());

        //find the max value and its index
        let mut max = f16::MIN;
        let mut idx: usize = 0;
        for i in 0..v.len() {
            if v[i] > max {
                max = v[i];
                idx = i;
            }
        }
        let result = idx;
        // unsafe {vptr.add(idx)};
        println!("idx: {:?}, {:?}", output_ptr, result);
        assert_eq!(result, output_ptr);
    }
}
