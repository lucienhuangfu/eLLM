use crate::num_traits::NegInfinity;

#[derive(Clone)]
pub(super) struct AttentionScratch<T> {
    running_max_pool: Box<[T]>,
    running_max_stride: usize,
    running_denom_pool: Box<[T]>,
    running_denom_stride: usize,
    scores_pool: Box<[T]>,
    scores_stride: usize,
}

pub(super) struct AttentionScratchSlice<'a, T> {
    pub(super) running_max: &'a mut [T],
    pub(super) running_denom: &'a mut [T],
    pub(super) scores: &'a mut [T],
}

impl<T> AttentionScratch<T>
where
    T: Copy + Default,
{
    pub(super) fn new(thread_num: usize, row_step: usize, col_step: usize) -> Self {
        let running_max_stride = row_step.max(1);
        let running_denom_stride = row_step.max(1);
        let scores_stride = col_step.max(1);

        Self {
            running_max_pool: vec![T::default(); thread_num * running_max_stride]
                .into_boxed_slice(),
            running_max_stride,
            running_denom_pool: vec![T::default(); thread_num * running_denom_stride]
                .into_boxed_slice(),
            running_denom_stride,
            scores_pool: vec![T::default(); thread_num * scores_stride].into_boxed_slice(),
            scores_stride,
        }
    }

    #[inline(always)]
    pub(super) fn thread_buffers(
        &self,
        thread_id: usize,
        row_count: usize,
        col_count: usize,
    ) -> AttentionScratchSlice<'_, T> {
        unsafe {
            let running_max = std::slice::from_raw_parts_mut(
                self.running_max_pool
                    .as_ptr()
                    .add(thread_id * self.running_max_stride) as *mut T,
                row_count,
            );
            let running_denom = std::slice::from_raw_parts_mut(
                self.running_denom_pool
                    .as_ptr()
                    .add(thread_id * self.running_denom_stride) as *mut T,
                row_count,
            );
            let scores = std::slice::from_raw_parts_mut(
                self.scores_pool
                    .as_ptr()
                    .add(thread_id * self.scores_stride) as *mut T,
                col_count,
            );
            AttentionScratchSlice {
                running_max,
                running_denom,
                scores,
            }
        }
    }
}

impl<T> AttentionScratchSlice<'_, T>
where
    T: Copy + Default + NegInfinity,
{
    #[inline(always)]
    pub(super) fn clear(&mut self) {
        for value in self.running_max.iter_mut() {
            *value = T::neg_infinity();
        }
        for value in self.running_denom.iter_mut() {
            *value = T::default();
        }
        for value in self.scores.iter_mut() {
            *value = T::default();
        }
    }
}
