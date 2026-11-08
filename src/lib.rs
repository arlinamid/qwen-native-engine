//! Qwen3.5-0.8B in the browser: Rust owns tokenizing, the generation loop, grammar-constrained
//! decoding and the agent loop; onnxruntime-web runs the model on WebGPU.

pub mod grammar;
pub mod prompt;
pub mod sampling;
pub mod schema;
pub mod toolcall;

#[cfg(target_arch = "wasm32")]
mod engine;
#[cfg(target_arch = "wasm32")]
mod ortjs;
