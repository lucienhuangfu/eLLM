use crate::llama::Transformer;
use pyo3::{exceptions::PyRuntimeError, prelude::*};

#[pyclass]
pub struct Transformer32 {
    inner: Transformer<f32>,
}

#[pymethods]
impl Transformer32 {
    #[new]
    fn new(config: crate::init::config::Config) -> Self {
        Self {
            inner: Transformer::new(config),
        }
    }

    fn forward(&self, sequences: Vec<Vec<i32>>, start_pos: usize) -> PyResult<Vec<Vec<f32>>> {
        self.inner
            .forward(sequences, start_pos)
            .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
    }
}
