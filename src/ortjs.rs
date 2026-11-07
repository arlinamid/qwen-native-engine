//! Minimal direct bindings to onnxruntime-web (global `ort`).
//!
//! The `ort-web` crate does not expose `preferredOutputLocation`, which is what keeps the
//! KV-cache on the GPU between steps, so the runtime is bound directly here.

use js_sys::{Array, ArrayBuffer, Object, Reflect};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ort)]
    pub type InferenceSession;

    #[wasm_bindgen(catch, js_namespace = ort, static_method_of = InferenceSession, js_name = create)]
    async fn create_raw(model: &JsValue, options: &JsValue) -> Result<JsValue, JsValue>;

    #[wasm_bindgen(catch, structural, method, js_name = run)]
    async fn run_raw(this: &InferenceSession, feeds: &Object) -> Result<JsValue, JsValue>;

    #[wasm_bindgen(structural, method, getter, js_name = inputMetadata)]
    pub fn input_metadata(this: &InferenceSession) -> Array;

    #[wasm_bindgen(structural, method, getter, js_name = outputNames)]
    pub fn output_names(this: &InferenceSession) -> Array;

    #[wasm_bindgen(js_namespace = ort)]
    #[derive(Clone)]
    pub type Tensor;

    #[wasm_bindgen(catch, constructor, js_namespace = ort)]
    pub fn new(dtype: &str, data: &JsValue, dims: &Array) -> Result<Tensor, JsValue>;

    #[wasm_bindgen(structural, method, getter)]
    pub fn dims(this: &Tensor) -> Array;

    #[wasm_bindgen(structural, method, getter)]
    pub fn location(this: &Tensor) -> String;

    #[wasm_bindgen(structural, method, getter, js_name = type)]
    pub fn dtype(this: &Tensor) -> String;

    #[wasm_bindgen(structural, method, getter, js_name = gpuBuffer)]
    pub fn gpu_buffer(this: &Tensor) -> JsValue;

    #[wasm_bindgen(catch, structural, method, js_name = getData)]
    pub async fn get_data(this: &Tensor) -> Result<JsValue, JsValue>;

    #[wasm_bindgen(structural, method)]
    pub fn dispose(this: &Tensor);
}

#[wasm_bindgen(inline_js = r#"
let staging = null;
export function nowMs() { return performance.now(); }
// Copies one byte range of a GPU buffer to the CPU. A full-tensor read of the logits would move
// hundreds of kB per prompt token; the generation loop only ever needs the last row.
export async function readBufferRange(buffer, offset, size) {
  const device = await globalThis.ort.env.webgpu.device;
  if (!staging || staging.size < size) {
    staging?.destroy();
    staging = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  }
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, offset, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ, 0, size);
  const copy = staging.getMappedRange(0, size).slice(0);
  staging.unmap();
  return copy;
}
"#)]
extern "C" {
    #[wasm_bindgen(js_name = nowMs)]
    pub fn now_ms() -> f64;

    #[wasm_bindgen(catch, js_name = readBufferRange)]
    async fn read_buffer_range(buffer: &JsValue, offset: f64, size: f64) -> Result<JsValue, JsValue>;
}

impl InferenceSession {
    pub async fn create(model: &JsValue, options: &JsValue) -> Result<InferenceSession, JsValue> {
        Ok(Self::create_raw(model, options).await?.unchecked_into())
    }

    pub async fn run(&self, feeds: &Object) -> Result<JsValue, JsValue> {
        self.run_raw(feeds).await
    }
}

pub fn get(obj: &JsValue, key: &str) -> Result<JsValue, JsValue> {
    Reflect::get(obj, &JsValue::from_str(key))
}

/// Reads `size` bytes at `offset` of a GPU-located tensor into a fresh `ArrayBuffer`.
pub async fn read_gpu_range(tensor: &Tensor, offset: f64, size: f64) -> Result<ArrayBuffer, JsValue> {
    Ok(read_buffer_range(&tensor.gpu_buffer(), offset, size).await?.unchecked_into())
}
