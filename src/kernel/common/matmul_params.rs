#[derive(Clone, Copy)]
pub struct MatMulParams {
    pub a_row_step_macro: usize,
    pub b_row_step_macro: usize,
    pub column_step_macro: usize,
    pub a_row_step_micro: usize,
    pub b_row_step_micro: usize,
}

impl MatMulParams {
    #[inline(always)]
    pub fn mb(self) -> usize {
        self.a_row_step_macro.max(1)
    }

    #[inline(always)]
    pub fn nb(self) -> usize {
        self.b_row_step_macro.max(1)
    }

    #[inline(always)]
    pub fn kc(self) -> usize {
        self.column_step_macro.max(1)
    }

    #[inline(always)]
    pub fn mr(self) -> usize {
        self.a_row_step_micro.max(1)
    }

    #[inline(always)]
    pub fn nr(self) -> usize {
        self.b_row_step_micro.max(1)
    }
}
