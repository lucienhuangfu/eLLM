pub fn get_strides(shape: &Vec<usize>) -> Vec<usize> {
    let len = shape.len();
    let mut strides: Vec<usize> = vec![0; len];
    if shape.last().copied().unwrap() > 1 {
        if let Some(last) = strides.last_mut() {
            *last = 1;
        }
    }
    for k in 0..(len - 1) {
        // let product = ((k+1)..len).map(|x| shape[x]).product();
        let product = shape[(k + 1)..len].iter().product();
        strides[k] = product;
    }
    strides
}

// 计算对齐的步幅
pub fn get_aligned_strides(shape: &Vec<usize>, broadcast_shape: &Vec<usize>) -> Vec<usize> {
    let mut strides = vec![0; broadcast_shape.len()];
    if shape.len() > broadcast_shape.len() {
        panic!("Shape length cannot be greater than broadcast shape length");
    }
    let start = broadcast_shape.len().checked_sub(shape.len()).unwrap_or(0);
    let original_strides = get_strides(shape);

    for (i, &stride) in original_strides.iter().enumerate() {
        strides[start + i] = stride;
    }

    for i in 0..broadcast_shape.len() {
        if broadcast_shape[i] != shape.get(i.checked_sub(start).unwrap_or(0)).copied().unwrap_or(1) {
            strides[i] = 0;
        }
    }

    strides
}

pub fn get_broadcast_shape(a_shape: &Vec<usize>, b_shape: &Vec<usize>) -> Vec<usize> {
    let mut result_shape = Vec::new();
    let max_len = std::cmp::max(a_shape.len(), b_shape.len());

    for i in 0..max_len {
        let a_dim = if i < a_shape.len() { a_shape[a_shape.len() - 1 - i] } else { 1 };
        let b_dim = if i < b_shape.len() { b_shape[b_shape.len() - 1 - i] } else { 1 };

        if a_dim != 1 && b_dim != 1 && a_dim != b_dim {
            panic!("Shapes cannot be broadcasted: {:?} and {:?}", a_shape, b_shape);
        }

        result_shape.push(std::cmp::max(a_dim, b_dim));
    }

    result_shape.reverse();
    result_shape
}

/*
pub fn iterate_with_broadcast<F>(
    a_data: *const T,
    b_data: *const T,
    broadcast_shape: Vec<usize>, 
    a_shape: Vec<usize>, 
    b_shape: Vec<usize>, 
    mut f: F,
) where F: FnMut(*const T, *const T) {
    let a_strides = Self::get_aligned_strides(&a_shape, &broadcast_shape);
    let b_strides = Self::get_aligned_strides(&b_shape, &broadcast_shape);

    let mut a_index = 0;
    let mut b_index = 0;

    for i in 0..broadcast_shape.iter().product::<usize>() {
        f(unsafe { a_data.add(a_index) }, unsafe { b_data.add(b_index) });

        let mut carry = 1;
        for (dim, (a_stride, b_stride)) in broadcast_shape.iter().zip(a_strides.iter().zip(b_strides.iter())).rev() {
            if carry == 0 {
                break;
            }

            a_index += a_stride * carry;
            b_index += b_stride * carry;

            carry = (i + 1) % dim;
        }
    }
}
 */



#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_get_strides() {
        // 3 * 4
        let high = 2;
        let row = 3;
        let col = 4;
        let shape2 = vec![row, col];
        let strides2: Vec<usize> = vec![col, 1];
        let shape2 = get_strides(&shape2);
        assert_eq!(strides2, shape2);

        let shape3 = vec![high, row, col];
        let strides3: Vec<usize> = vec![row * col, col, 1];
        let shape3 = get_strides(&shape3);
        assert_eq!(strides3, shape3);

        let row = 1;
        let col = 4;
        let shape2 = vec![row, col];
        let strides2: Vec<usize> = vec![4, 1];
        let shape2 = get_strides(&shape2);
        assert_eq!(strides2, shape2);
        // println!("broadcast_shape {:?}", broadcast_shape);
    }

    #[test]
    fn test_broadcast() {
        let result_shape = get_broadcast_shape(&vec![512, 1024], &vec![512, 1024]);
        assert_eq!(result_shape, vec![512, 1024]);

        let result_shape = get_broadcast_shape(&vec![1, 1024], &vec![512, 1024]);
        assert_eq!(result_shape, vec![512, 1024]);

                /* 
        vec![
            config.max_position_embeddings,
            1,
            1,
            config.attention_head_size,
        ]*/

        let sequence_length = 8;
        let batch_size = 4;
        let head_num = 64;
        let head_size = 128;

        let result_shape = get_broadcast_shape(&vec![batch_size, head_num, head_size], &vec![sequence_length, 1, 1, head_size]);
        assert_eq!(result_shape, vec![8, 4, 64, 128]);
        println!("{:?}", result_shape);
    }

    #[test]
    fn test_get_aligned_strides() {
        let shape = vec![3, 4];
        let broadcast_shape = vec![2, 3, 4];
        let aligned_strides = get_aligned_strides(&shape, &broadcast_shape);
        assert_eq!(aligned_strides, vec![0, 4, 1]);

        let shape = vec![1, 4];
        let broadcast_shape = vec![2, 3, 4];
        let aligned_strides = get_aligned_strides(&shape, &broadcast_shape);
        assert_eq!(aligned_strides, vec![0, 0, 1]);

        let shape = vec![2, 3, 4];
        let broadcast_shape = vec![2, 3, 4];
        let aligned_strides = get_aligned_strides(&shape, &broadcast_shape);
        assert_eq!(aligned_strides, vec![12, 4, 1]);
    }
}
