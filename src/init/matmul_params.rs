#[derive(Clone, Copy)]
pub struct MatMulParams {
    pub a_row: usize,
    pub b_row: usize,
    pub column: usize,
    pub a_row_step_macro: usize,
    pub b_row_step_macro: usize,
    pub column_step_macro: usize,
    pub a_row_step_micro: usize,
    pub b_row_step_micro: usize,
}