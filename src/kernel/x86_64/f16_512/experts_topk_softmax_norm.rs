use crate::kernel::generic::exp::Exp;
use crate::kernel::x86_64::f16_512::activation::exp512;
use std::ops::{AddAssign, Div, Sub};
use std::ptr;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub fn experts_topk_softmax_norm(
    input_ptr: *const f16,
    // [num_experts]
    experts_indicator_ptr: *mut bool,
    // token_size = sequence_chunk_size * batch_size
    // [num_experts, token_size]
    output_indices_ptr: *mut bool,
    // [num_experts, token_size]
    output_values_ptr: *mut f16,
    index_token: usize,
    num_token: usize,
    num_experts: usize,
    num_topk: usize,
) {
   
}


fn get_topk(input_ptr: *const f16) -> (__m512i, __m512) {
    unsafe {


        (topk_indices, topk_values)
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_ulps_eq;
    use std::arch::x86_64::*;

    
}
