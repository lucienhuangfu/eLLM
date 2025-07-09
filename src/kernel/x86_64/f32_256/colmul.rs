use std::arch::x86_64::*;
use std::ptr;
use super::utils::hsum256_ps_avx;



#[inline]
unsafe fn insert_float_at_dynamic_position(vec: &mut __m256, value: f32, position: usize) {
    let val_vec = _mm256_set1_ps(value);
    let mut mask: [i32; 8] = [0; 8];
    mask[position] = -1; // Insert value at dynamic position.
    let mask_vec = _mm256_loadu_si256(mask.as_ptr() as *const __m256i);
    _mm256_maskstore_ps(vec as *mut __m256 as *mut f32, mask_vec, val_vec);
}

pub fn colmul(input_ptr1: *const f32, input_ptr2: *const f32, output_ptr: *mut f32, row_size: usize, col_size: usize) {
    println!("add avx 256");

    unsafe {
        let rem = row_size % 8;
        let length2 = row_size - rem;
        let mut storage_vecs: [__m256; 4] = [_mm256_setzero_ps(); 4]; // Only 4 vectors for storage
        let mut acc_vecs: [__m256; 4] = [_mm256_setzero_ps(); 4];     // 4 vectors for accumulation

        for (x, (ptr1, ptr2)) in (0..col_size).map(|x| (input_ptr1, input_ptr2.add(row_size))).enumerate() {
            let mut sum1 = 0.0f32;
            let position = x % 8;

            if rem != row_size {
                for j in (0..length2).step_by(8) {
                    let x1 = _mm256_loadu_ps(ptr1.add(j));
                    let x2 = _mm256_loadu_ps(ptr2.add(j));
                    let prod = _mm256_mul_ps(x1, x2);
                    sum1 += hsum256_ps_avx(prod);
                }
            }

            let mut sum2 = 0.0f32;
            if rem != 0 {
                for j in length2..row_size {
                    sum2 += *ptr1.add(j) * *ptr2.add(j);
                }
            }

            let sum = sum1 + sum2;

            // Insert the sum into the corresponding acc_vecs vector.
            insert_float_at_dynamic_position(&mut acc_vecs[(x / 8) % 4], sum, position);

            // When position reaches 7, accumulate into storage_vecs and reset acc_vecs
            if position == 7 {
                storage_vecs[(x / 8) % 4] = _mm256_add_ps(storage_vecs[(x / 8) % 4], acc_vecs[(x / 8) % 4]);
                acc_vecs[(x / 8) % 4] = _mm256_setzero_ps(); // Reset acc_vecs for the next batch
            }
        }

        // Store the result into the output vector, using only the 8 registers.
        for i in 0..4 {
            _mm256_storeu_ps(output_ptr.add(8 * i), storage_vecs[i]);
        }
    }
}

pub fn colmul_reg(input_ptr1: *const f32, input_ptr2: *const f32, output_ptr: *mut f32, row_size: usize, col_size: usize) {
    println!("add avx 256");

    unsafe {
        let rem = row_size % 8;
        let length2 = row_size - rem;
        let mut storage_vecs: [__m256; 4] = [_mm256_setzero_ps(); 4]; // Only 4 vectors for storage
        let mut acc_vecs: [__m256; 4] = [_mm256_setzero_ps(); 4];     // 4 vectors for accumulation
        let mut temp = _mm256_setzero_ps();

        for (x, (ptr1, ptr2)) in (0..col_size).map(|x| (input_ptr1, input_ptr2.add(row_size))).enumerate() {
            let mut sum1 = 0.0f32;
            let position = x % 8;

            if rem != row_size {
                for j in (0..length2).step_by(8) {
                    let x1 = _mm256_loadu_ps(ptr1.add(j));
                    let x2 = _mm256_loadu_ps(ptr2.add(j));
                    let prod = _mm256_mul_ps(x1, x2);
                    sum1 += hsum256_ps_avx(prod);
                }
            }

            let mut sum2 = 0.0f32;
            if rem != 0 {
                for j in length2..row_size {
                    sum2 += *ptr1.add(j) * *ptr2.add(j);
                }
            }

            let sum = sum1 + sum2;
            insert_float_at_dynamic_position(&mut temp, sum, position);

            if position == 7 {
                storage_vecs[(x / 8) % 4] = temp;
                temp = _mm256_setzero_ps();
            }
        }

        // Store the result into the output vector, using only the 8 registers.
        for i in 0..4 {
            _mm256_storeu_ps(output_ptr.add(8 * i), storage_vecs[i]);
        }
    }
}


#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;
    use super::*;

    #[test]
    fn test_mul() {
        let head_size = 32;//row
        // let head_size = 8;//row

        let head_num  = 64;
        let batch_size = 16;
        let sequence_length = 256;//col
        // let sequence_length = 8;//col

        let hidden_size = 8192;
        
        let shapes = vec![batch_size, hidden_size];
        // let length = shapes.iter().product();
        let data1: Vec<f32> = vec![1.0; sequence_length];
        let data2: Vec<f32> = vec![1.0; sequence_length * head_size];
        let mut output: Vec<f32> = vec![0.0; head_size];
        let result: Vec<f32> = vec![sequence_length as f32; head_size];
        colmul_reg(data1.as_ptr(), data2.as_ptr(), output.as_mut_ptr(),sequence_length,head_size );
        println!("result:{:?}",output);
        assert_ulps_eq!(output[..], result[..], max_ulps=4);
        
    }
    
}