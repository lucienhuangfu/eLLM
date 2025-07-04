#![feature(test)]
#![feature(f16)]
#![feature(duration_millis_float)]
#![feature(sync_unsafe_cell)]
#![feature(stdarch_x86_avx512)]
#![feature(stdarch_x86_avx512_f16)]
#![feature(avx512_target_feature)]
#![feature(specialization)]
#![allow(incomplete_features)]
#![allow(unused_parens)]
// #![feature(trait_upcasting)]
// #![feature(asm_const)]
pub mod init;
pub mod memory;
pub mod kernel;
pub mod compiler;
pub mod ptensor;
pub mod llama;
// pub mod runtime;
/*
pub mod serving;
*/

/* 
#[macro_use]
extern crate log;
#[macro_use]
extern crate approx;*/