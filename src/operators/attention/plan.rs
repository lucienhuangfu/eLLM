#[derive(Copy, Clone)]
pub(super) struct RowVisitPlan {
    pub(super) main: Option<(usize, usize)>,
    pub(super) tail: Option<(usize, usize)>,
}
