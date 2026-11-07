// Runs the engine off the main thread. Each load uses a fresh worker, so that terminating it
// releases the session and its GPU memory.
import * as ort from '/vendor/ort/ort.webgpu.bundle.min.mjs';
import init, { Engine } from '/pkg/qwen_wasm.js';
import { runTool } from './tools.js';

globalThis.ort = ort;
ort.env.wasm.wasmPaths = '/vendor/ort/';
ort.env.wasm.numThreads = 1;
ort.env.logLevel = 'warning';

const TOKENIZER = '/models/qwen3.5-0.8b/tokenizer.json';
let engine = null;

async function gpuInfo() {
  if (!navigator.gpu) return { available: false };
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return { available: false };
  return {
    available: true,
    vendor: adapter.info?.vendor,
    architecture: adapter.info?.architecture,
    description: adapter.info?.description,
    shaderF16: adapter.features.has('shader-f16'),
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
  };
}

const commands = {
  async load(options) {
    // Threads speed up a CPU prefill session; they need cross-origin isolation (see serve.mjs).
    ort.env.wasm.numThreads = options?.wasm_threads ?? 1;
    const t0 = performance.now();
    await init();
    const wasmMs = performance.now() - t0;
    const tokenizerBytes = new Uint8Array(await (await fetch(TOKENIZER)).arrayBuffer());
    const t1 = performance.now();
    engine = await Engine.load(tokenizerBytes, options);
    const sessionMs = performance.now() - t1;
    return {
      ...engine.info(),
      gpu: await gpuInfo(),
      crossOriginIsolated: globalThis.crossOriginIsolated,
      wasmMs,
      sessionMs,
    };
  },
  prepareGrammar: () => engine.prepare_grammar(),
  clearPrefixCache: () => engine.clear_prefix_cache(),
  generate: ({ prompt, options }, post) =>
    engine.generate(prompt, options, (piece) => post({ type: 'token', piece })),
  agent({ request, mocks }, post) {
    const queue = mocks ? [...mocks] : null;
    const execute = (name, argsJson) => {
      if (queue) return queue.shift() ?? '{"error":"no result available"}';
      return runTool(name, JSON.parse(argsJson));
    };
    return engine.agent(request, execute, (event) => post(event));
  },
};

self.onmessage = async ({ data: { id, command, args } }) => {
  const post = (event) => self.postMessage({ id, event });
  try {
    self.postMessage({ id, result: await commands[command](args, post) });
  } catch (error) {
    self.postMessage({ id, error: String(error?.stack ?? error?.message ?? error) });
  }
};
