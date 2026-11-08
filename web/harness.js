// Main-thread side: worker lifecycle, benchmarks and the tool-calling evaluation.
import { SYSTEM_PROMPT, TOOLS } from './tools.js';

let worker = null;
let nextId = 1;
const pending = new Map();

function call(command, args, onEvent) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject, onEvent });
    worker.postMessage({ id, command, args });
  });
}

function startWorker() {
  worker?.terminate();
  for (const { reject } of pending.values()) reject(new Error('worker restarted'));
  pending.clear();
  worker = new Worker('./worker.js', { type: 'module' });
  worker.onmessage = ({ data }) => {
    const job = pending.get(data.id);
    if (!job) return;
    if (data.event) {
      job.onEvent?.(data.event);
      return;
    }
    pending.delete(data.id);
    if (data.error) job.reject(new Error(data.error));
    else job.resolve(data.result);
  };
  worker.onerror = (event) => {
    for (const { reject } of pending.values()) reject(new Error(event.message));
    pending.clear();
  };
}

// Prompt processing on 4 CPU threads, generation on WebGPU: measured 4x faster prompts than
// WebGPU alone on the reference iGPU. Pass prefill_device: undefined to turn it off.
const LOAD_DEFAULTS = { dtype: 'q4f16', prefill_device: 'wasm', wasm_threads: 4 };

export const load = (options = {}) => {
  startWorker();
  return call('load', { ...LOAD_DEFAULTS, ...options });
};
export const generate = (prompt, options = {}, onEvent) => call('generate', { prompt, options }, onEvent);
export const agent = (request, onEvent, mocks) =>
  call('agent', { request: { tools: TOOLS, system: SYSTEM_PROMPT, ...request }, mocks }, onEvent);
export const prepareGrammar = () => call('prepareGrammar');
export const clearPrefixCache = () => call('clearPrefixCache');

export async function saveResult(name, data) {
  await fetch(`/results/${name}.json`, { method: 'PUT', body: JSON.stringify(data, null, 1) });
}

const chat = (user) => `<|im_start|>user\n${user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;
const SHORT = chat('Explain in simple terms why the sky is blue.');
const LONG = chat(`${'The history of computing spans mechanical calculators, vacuum tubes, transistors, integrated circuits and the networked machines of today. '.repeat(24)}\nSummarize the text above.`);
const GREEDY = { temperature: 0 };

/** Speed of one memory layout: short and long prompts, each generating 64 tokens. */
export async function benchLayout({ kv_location = 'gpu-buffer', logits_location = 'gpu-buffer', device = 'webgpu', prefill_device, wasm_threads, dtype = 'q4f16', prefill_chunk, tokens = 64 }) {
  const loaded = await load({ device, prefill_device, wasm_threads, dtype, kv_location, logits_location });
  const runs = [];
  const run = async (name, prompt, max_new_tokens) => {
    const { text, stats } = await generate(prompt, { max_new_tokens, prefill_chunk, sampling: GREEDY });
    runs.push({ name, text, ...stats });
  };
  await run('warmup', SHORT, 4);
  await run('short', SHORT, tokens);
  await run('long', LONG, tokens);
  return { kv_location, logits_location, device, prefill_device, wasm_threads, dtype, prefill_chunk, load: loaded, runs };
}

/** Time to first token of a tool prompt, with and without the KV-cache snapshot. */
export async function benchPrefix() {
  const generation = { max_new_tokens: 48, sampling: GREEDY };
  const once = (prefix_cache) => agent({ user: 'What is the weather in Paris?', max_steps: 1, prefix_cache, generation });
  const without = await once(false);
  const miss = await once(true);
  const hit = await once(true);
  const pick = ({ steps: [step] }) => ({ text: step.text, ...step.stats });
  return { without: pick(without), first_with_cache: pick(miss), hit: pick(hit) };
}

function matches(call, expected) {
  if (call.name !== expected.name) return false;
  const args = call.arguments ?? {};
  for (const [key, value] of Object.entries(expected.args ?? {})) {
    const actual = key === 'expression' ? String(args[key]).replace(/\s+/g, '') : args[key];
    if (actual !== value) return false;
  }
  for (const [key, part] of Object.entries(expected.contains ?? {})) {
    if (!String(args[key] ?? '').toLowerCase().includes(String(part).toLowerCase())) return false;
  }
  return true;
}

/**
 * A case passes only if the exact sequence of expected calls was made and executed, with no
 * rejected call and no malformed output on the way.
 */
export function scoreCase(testCase, outcome) {
  const calls = outcome.calls;
  const expected = testCase.expected;
  const toolsRight = calls.length === expected.length && expected.every((e, i) => calls[i].name === e.name);
  const clean = outcome.parse_errors === 0 && calls.every((c) => !c.error);
  const argumentsRight = toolsRight && expected.every((e, i) => matches(calls[i], e));
  const completed = outcome.finished === true && typeof outcome.answer === 'string' && outcome.answer.trim().length > 0;
  const answerChecked = Array.isArray(testCase.answer_contains) && testCase.answer_contains.length > 0;
  const answerRight = answerChecked && testCase.answer_contains.every(text => outcome.answer?.toLowerCase().includes(String(text).toLowerCase()));
  // A tool-routing-only case is not an end-to-end pass without an answer oracle.
  const routingPass = clean && argumentsRight && completed;
  const pass = !routingPass ? false : answerChecked ? answerRight : null;
  return { toolsRight, argumentsRight, completed, answerChecked, answerRight, routingPass, pass };
}

export function summarize(rows) {
  const group = (filter) => {
    const part = rows.filter(filter);
    const count = (key) => part.filter((r) => r[key]).length;
    return { n: part.length, tools_right: count('toolsRight'), routing_pass: count('routingPass'), answer_checked: count('answerChecked'), pending_review: part.filter(r => r.pass === null).length, pass: count('pass') };
  };
  const invalid = rows.reduce((sum, r) => sum + r.parseErrors + r.invalidCalls, 0);
  return {
    all: group(() => true),
    single: group((r) => r.kind === 'single'),
    none: group((r) => r.kind === 'none'),
    chain: group((r) => r.kind === 'chain'),
    en: group((r) => r.lang === 'en'),
    hu: group((r) => r.lang === 'hu'),
    invalid_outputs: invalid,
    total_tokens: rows.reduce((sum, r) => sum + r.tokens, 0),
    seconds: rows.reduce((sum, r) => sum + r.ms, 0) / 1000,
  };
}

/** Runs every case greedily with mock tool results. `onRow` reports progress. */
export async function runEval({ mode = 'lazy', thinking = false, max_new_tokens = 160, ids, onRow } = {}) {
  let cases = await (await fetch('./eval_cases.json')).json();
  if (ids) cases = cases.filter((c) => ids.includes(c.id));
  const rows = [];
  for (const testCase of cases) {
    const t0 = performance.now();
    const outcome = await agent(
      { user: testCase.user, mode, thinking, max_steps: testCase.expected.length + 2, generation: { max_new_tokens, sampling: GREEDY } },
      undefined,
      testCase.mock,
    );
    const row = {
      id: testCase.id,
      kind: testCase.kind,
      lang: testCase.lang,
      user: testCase.user,
      ...scoreCase(testCase, outcome),
      calls: outcome.calls,
      answer: outcome.answer,
      parseErrors: outcome.parse_errors,
      invalidCalls: outcome.calls.filter((c) => c.error).length,
      tokens: outcome.steps.reduce((sum, s) => sum + s.stats.new_tokens, 0),
      ms: performance.now() - t0,
    };
    rows.push(row);
    onRow?.(row);
  }
  return { mode, thinking, summary: summarize(rows), rows };
}
