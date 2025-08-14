#[cfg(feature = "python")]
pub mod python_bindings {
    pub mod transformer32;
    pub use transformer32::Transformer32;
}