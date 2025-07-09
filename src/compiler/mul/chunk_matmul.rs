use crate::init::{send_sync_ptr::{ConstPtr, MutPtr}, matmul_params::MatMulParams};


pub fn chunk_matmul<T>(
    data1: *const T,
    data2: *const T,
    data3: *mut T,
    params: &MatMulParams,
) -> Vec<(ConstPtr<T>, ConstPtr<T>, MutPtr<T>)> {

    
    // Ensure dimensions are divisible by step sizes
    assert!(
        params.a_row % params.a_row_step_macro == 0,
        "a_row is not divisible by a_row_step_macro"
    );
    assert!(
        params.b_row % params.b_row_step_macro == 0,
        "b_row is not divisible by b_row_step_macro"
    );

    // Create the work list
    let num_tasks = (params.a_row / params.a_row_step_macro)
        * (params.b_row / params.b_row_step_macro);
    let mut tasks: Vec<_> = Vec::with_capacity(num_tasks);

    // Base pointers
    let base_a_ptr: *const T = data1;
    let base_b_ptr: *const T = data2;
    let base_c_ptr: *mut T = data3;

    // Dimensions
    let m_total = params.a_row;
    let n_total = params.b_row;
    let k_total = params.column;

    for i in (0..m_total).step_by(params.a_row_step_macro) {
        let m = std::cmp::min(params.a_row_step_macro, m_total - i);
        for j in (0..n_total).step_by(params.b_row_step_macro) {
            let n = std::cmp::min(params.b_row_step_macro, n_total - j);

            // Compute offsets
            let a_offset = i * k_total;
            let b_offset = j;
            let c_offset = i * n_total + j;

            let a_macro_ptr = unsafe { base_a_ptr.add(a_offset) };
            let b_macro_ptr = unsafe { base_b_ptr.add(b_offset) };
            let c_macro_ptr = unsafe { base_c_ptr.add(c_offset) };

            tasks.push(
                (
                    ConstPtr {ptr: a_macro_ptr},
                    ConstPtr {ptr: b_macro_ptr},
                    MutPtr {ptr: c_macro_ptr}
                ));
        }
    }
    tasks
}


#[cfg(test)]
mod test {
    use approx::assert_ulps_eq;
    use super::*;
    use crate::kernel;

    #[test]
    fn test_chunk_mat() {
        // Define dimensions
        let hidden_size = 16;
        let batch_size = 8;

        // Initialize input matrices with ones
        let size1 = batch_size * hidden_size;
        let data1: Vec<f32> = vec![1.0; size1];

        let size2 = hidden_size * hidden_size;
        let data2: Vec<f32> = vec![1.0; size2];

        // Initialize output matrix with zeros
        let size3 = batch_size * hidden_size;
        let mut data3: Vec<f32> = vec![0.0; size3];

        // Expected result: all elements should be equal to hidden_size
        let result = vec![hidden_size as f32; size3];

        // Define MatMulParams
        let param = MatMulParams {
            a_row: batch_size,
            b_row: hidden_size,
            column: hidden_size,
            a_row_step_macro: 4,
            b_row_step_macro: 8,
            column_step_macro: hidden_size, // Set to cover the full k dimension
            a_row_step_micro: 2,
            b_row_step_micro: 2,
            // These parameters are used by matmul_block
            // column_step_micro: 0, // Not used in this context
        };

        // Execute matmul_block over the blocks
        for i in (0..batch_size).step_by(param.a_row_step_macro) {
            for j in (0..hidden_size).step_by(param.b_row_step_macro) {
                // Pointers to the current blocks
                let a_ptr = data1.as_ptr().wrapping_add(i * param.column);
                let b_ptr = data2.as_ptr().wrapping_add(j);
                let c_ptr = data3.as_mut_ptr().wrapping_add(i * param.b_row + j);

                // Update param for the current block if necessary
                let mut block_param = param.clone();

                // Adjust step sizes if we're at the edge and remaining size is smaller
                block_param.a_row_step_micro = std::cmp::min(param.a_row_step_macro, batch_size - i);
                block_param.b_row_step_micro = std::cmp::min(param.b_row_step_macro, hidden_size - j);

                // Call matmul_block
                kernel::generic::matmul_block::matmul_block(a_ptr, b_ptr, c_ptr, &block_param);
            }
        }

        // Assert that the computed data3 matches the expected result
        assert_ulps_eq!(data3[..], result[..], max_ulps = 4);
    }
}

