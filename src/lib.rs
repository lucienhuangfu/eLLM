#![feature(test)]
#![feature(f16)]
#![feature(duration_millis_float)]
#![feature(sync_unsafe_cell)]
#![feature(stdarch_x86_avx512)]
#![feature(stdarch_x86_avx512_f16)]
#![feature(stdarch_x86_avx512_bf16)]
#![feature(avx512_target_feature)]
#![feature(specialization)]
#![allow(incomplete_features)]
#![allow(unused_parens)]

pub mod init;
pub mod memory;
pub mod kernel;
pub mod compiler;
pub mod ptensor;
pub mod llama;

// Python 绑定模块
#[cfg(feature = "python")]
pub mod python_bindings;

use pyo3::prelude::*;

/// Python 模块定义
#[pymodule]
fn ellm(_py: Python, m: &PyModule) -> PyResult<()> {
    // 添加配置类
    m.add_class::<python_bindings::PyConfig>()?;
    
    // 添加 Transformer 类
    m.add_class::<python_bindings::PyTransformer>()?;
    
    // 添加分词器类
    m.add_class::<python_bindings::PyTokenizer>()?;
    
    // 添加版本信息
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    Ok(())
}

/*
pub mod runtime;
pub mod serving;
*/

/* 
#[macro_use]
extern crate log;
#[macro_use]
extern crate approx;*/