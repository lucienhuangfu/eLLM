use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use numpy::{PyArray1, PyArray2};
use crate::init::config::Config;

/// Python 绑定的 Config 结构
#[pyclass(name = "Config")]
#[derive(Clone)]
pub struct PyConfig {
    pub inner: Config,
}

#[pymethods]
impl PyConfig {
    #[new]
    #[pyo3(signature = (
        hidden_size=4096,
        vocab_size=32000,
        num_hidden_layers=32,
        num_attention_heads=32,
        max_position_embeddings=2048,
        intermediate_size=11008,
        rms_norm_eps=1e-6,
        **kwargs
    ))]
    fn new(
        hidden_size: usize,
        vocab_size: usize,
        num_hidden_layers: usize,
        num_attention_heads: usize,
        max_position_embeddings: usize,
        intermediate_size: usize,
        rms_norm_eps: f32,
        kwargs: Option<&PyDict>,
    ) -> PyResult<Self> {
        let mut config = Config::new();
        
        // 设置基本参数
        config.hidden_size = hidden_size;
        config.vocab_size = vocab_size;
        config.num_hidden_layers = num_hidden_layers;
        config.num_attention_heads = num_attention_heads;
        config.max_position_embeddings = max_position_embeddings;
        config.intermediate_size = intermediate_size;
        config.rms_norm_eps = rms_norm_eps;
        config.attention_head_size = hidden_size / num_attention_heads;
        
        // 处理额外的关键字参数
        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs.iter() {
                let key_str = key.extract::<String>()?;
                match key_str.as_str() {
                    "bos_token_id" => config.bos_token_id = value.extract::<usize>()?,
                    "eos_token_id" => config.eos_token_id = value.extract::<usize>()?,
                    "batch_size" => config.batch_size = value.extract::<usize>()?,
                    "num_key_value_heads" => config.num_key_value_heads = value.extract::<usize>()?,
                    _ => {
                        // 忽略未知参数或记录警告
                        eprintln!("Warning: Unknown config parameter: {}", key_str);
                    }
                }
            }
        }
        
        Ok(PyConfig { inner: config })
    }

    #[classmethod]
    fn from_json_file(_cls: &PyType, path: String) -> PyResult<Self> {
        let config = Config::from_file(&path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("Failed to load config from {}: {}", path, e)
            ))?;
        Ok(PyConfig { inner: config })
    }

    fn to_dict(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("hidden_size", self.inner.hidden_size)?;
            dict.set_item("vocab_size", self.inner.vocab_size)?;
            dict.set_item("num_hidden_layers", self.inner.num_hidden_layers)?;
            dict.set_item("num_attention_heads", self.inner.num_attention_heads)?;
            dict.set_item("num_key_value_heads", self.inner.num_key_value_heads)?;
            dict.set_item("max_position_embeddings", self.inner.max_position_embeddings)?;
            dict.set_item("intermediate_size", self.inner.intermediate_size)?;
            dict.set_item("rms_norm_eps", self.inner.rms_norm_eps)?;
            dict.set_item("attention_head_size", self.inner.attention_head_size)?;
            dict.set_item("batch_size", self.inner.batch_size)?;
            dict.set_item("bos_token_id", self.inner.bos_token_id)?;
            dict.set_item("eos_token_id", self.inner.eos_token_id)?;
            Ok(dict.into())
        })
    }

    // 属性访问器
    #[getter]
    fn hidden_size(&self) -> usize { self.inner.hidden_size }
    
    #[getter]
    fn vocab_size(&self) -> usize { self.inner.vocab_size }
    
    #[getter]
    fn num_hidden_layers(&self) -> usize { self.inner.num_hidden_layers }
    
    #[getter]
    fn num_attention_heads(&self) -> usize { self.inner.num_attention_heads }
    
    #[getter]
    fn attention_head_size(&self) -> usize { self.inner.attention_head_size }

    fn __repr__(&self) -> String {
        format!(
            "Config(hidden_size={}, vocab_size={}, num_layers={}, num_heads={})",
            self.inner.hidden_size,
            self.inner.vocab_size,
            self.inner.num_hidden_layers,
            self.inner.num_attention_heads
        )
    }
}

// 为 Config 添加从文件加载的方法
impl Config {
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        use std::fs::File;
        use std::io::BufReader;
        
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let config: Config = serde_json::from_reader(reader)?;
        Ok(config)
    }
}
