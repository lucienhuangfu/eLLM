pub trait ExpertsDownTrait<T> {
    fn compute1(&self, a_tile: *const T, b_panel: *const T, acc: *mut T);

    fn compute2(&self, out_row: *mut T, acc_row: *const T, factor: *const T, len: usize);
}

pub trait ExpertsSiluTrait<T> {
    fn compute1(
        &self,
        a_tile: *const T,
        gate_panel: *const T,
        up_panel: *const T,
        gate_acc: *mut T,
        up_acc: *mut T,
        kc: usize,
    );

    fn compute2(&self, gate_row: *const T, up_row: *const T, c_row: *mut T);
}

pub trait MoeMergeTrait<T> {
    fn merge_add(&self, out_row: *mut T, add_row: *const T, len: usize);
}
