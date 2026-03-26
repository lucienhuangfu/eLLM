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

#[derive(Clone, Copy)]
pub struct MatMulSigmoidParams {
    pub matmul: MatMulParams,
    pub m_max: usize,
    pub n_max: usize,
    pub k_max: usize,
}

impl MatMulSigmoidParams {
    #[inline(always)]
    pub const fn new(matmul: MatMulParams, m_max: usize, n_max: usize, k_max: usize) -> Self {
        Self {
            matmul,
            m_max,
            n_max,
            k_max,
        }
    }

    #[inline(always)]
    pub fn mb(self) -> usize {
        self.matmul.mb()
    }

    #[inline(always)]
    pub fn nb(self) -> usize {
        self.matmul.nb()
    }

    #[inline(always)]
    pub fn kc(self) -> usize {
        self.matmul.kc()
    }

    #[inline(always)]
    pub fn mr(self) -> usize {
        self.matmul.mr()
    }

    #[inline(always)]
    pub fn nr(self) -> usize {
        self.matmul.nr()
    }

    #[inline(always)]
    pub fn padded_m(self, m_run: usize) -> usize {
        let mr = self.mr();
        ((m_run + mr - 1) / mr) * mr
    }
}
