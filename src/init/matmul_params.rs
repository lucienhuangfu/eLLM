#[derive(Clone, Copy)]
pub struct MatmulParams {
    pub a_row_step_macro: usize,
    pub b_row_step_macro: usize,
    pub column_step_macro: usize,
    pub a_row_step_micro: usize,
    pub b_row_step_micro: usize,
}