//! Checks the Rust prompt renderer and tokenizer against Hugging Face's output, and the grammar
//! against the real vocabulary.

use qwen_wasm::{
    grammar::ToolGrammar,
    prompt::{Message, render},
    sampling::{Sampler, SamplingOptions},
    toolcall::{parse_output, validate_call},
};
use serde::Deserialize;
use serde_json::{Value, json};
use tokenizers::Tokenizer;

const TOKENIZER: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/qwen3.5-0.8b/tokenizer.json");
const VOCAB: usize = 248_320;

#[derive(Deserialize)]
struct Case {
    name: String,
    messages: Vec<Message>,
    tools: Vec<Value>,
    enable_thinking: bool,
    text: String,
    ids: Vec<u32>,
}

fn cases() -> Vec<Case> {
    serde_json::from_str(include_str!("golden.json")).unwrap()
}

#[test]
fn prompt_matches_hugging_face_template() {
    for case in cases() {
        let got = render(&case.messages, &case.tools, true, case.enable_thinking);
        assert_eq!(got, case.text, "case {}", case.name);
    }
}

#[test]
fn tokenizer_matches_hugging_face_ids() {
    let tokenizer = Tokenizer::from_file(TOKENIZER).unwrap();
    for case in cases() {
        let ids = tokenizer.encode(case.text.as_str(), false).unwrap().get_ids().to_vec();
        assert_eq!(ids, case.ids, "case {}", case.name);
    }
}

fn tools() -> Vec<Value> {
    vec![
        json!({"type":"function","function":{"name":"get_weather","parameters":{"type":"object",
            "properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},
            "required":["city"]}}}),
        json!({"type":"function","function":{"name":"set_timer","parameters":{"type":"object",
            "properties":{"minutes":{"type":"number"},"label":{"type":"string"}},"required":["minutes"]}}}),
        json!({"type":"function","function":{"name":"now","parameters":{"type":"object","properties":{}}}}),
    ]
}

/// Drives the grammar with noise instead of a model: however bad the logits, the result must
/// parse and validate. That is the guarantee the agent relies on.
#[test]
fn grammar_forces_valid_calls_from_random_logits() {
    let tokenizer = Tokenizer::from_file(TOKENIZER).unwrap();
    let grammar = ToolGrammar::new(&tokenizer).unwrap();
    let tools = tools();
    let mut seen = std::collections::HashSet::new();

    for seed in 1..=60u64 {
        let mut constraint = grammar.constraint(&tools).unwrap();
        let mut sampler = Sampler::new(SamplingOptions { temperature: 1.0, top_k: 8, top_p: 1.0, seed, ..Default::default() });
        // Favouring the low token ids (punctuation, bytes) makes random strings close soon enough.
        let logits: Vec<f32> = (0..VOCAB).map(|i| if i < 512 { 2.0 } else { 0.0 }).collect();
        let mut out = Vec::new();
        let mut finished = false;

        for _ in 0..500 {
            let step = constraint.compute_mask().unwrap();
            if step.is_stop() {
                finished = true;
                break;
            }
            let mask = step.sample_mask.clone().expect("mask or stop");
            let token = sampler.sample(&logits, Some(&mask));
            assert!(mask.is_allowed(token), "sampled a masked token");
            out.push(token);
            if constraint.commit_token(Some(token)).unwrap().stop {
                finished = true;
                break;
            }
        }
        if !finished {
            continue; // A long random string, not a grammar failure.
        }

        let text = format!("<tool_call>{}", tokenizer.decode(&out, false).unwrap());
        let parsed = parse_output(&text, &tools);
        assert!(parsed.errors.is_empty(), "seed {seed}: {text:?} -> {:?}", parsed.errors);
        assert_eq!(parsed.tool_calls.len(), 1, "seed {seed}: {text:?}");
        validate_call(&parsed.tool_calls[0], &tools).unwrap_or_else(|e| panic!("seed {seed}: {text:?}: {e}"));
        seen.insert(parsed.tool_calls[0].name.clone());
    }
    assert!(!seen.is_empty(), "no random walk reached the end of the grammar");
    println!("tools reached by random walks: {seen:?}");
}

fn walk(grammar: &ToolGrammar, tokenizer: &Tokenizer, tools: &[Value], text: &str) -> Result<(), String> {
    let mut constraint = grammar.constraint(tools).unwrap();
    for id in tokenizer.encode(text, false).unwrap().get_ids() {
        let step = constraint.compute_mask().unwrap();
        let mask = step.sample_mask.clone().ok_or("grammar ended early")?;
        if !mask.is_allowed(*id) {
            return Err(format!("rejected token {id}: {:?}", tokenizer.decode(&[*id], false)));
        }
        constraint.commit_token(Some(*id)).unwrap();
    }
    Ok(())
}

#[test]
fn grammar_accepts_real_calls_token_by_token() {
    let tokenizer = Tokenizer::from_file(TOKENIZER).unwrap();
    let grammar = ToolGrammar::new(&tokenizer).unwrap();
    let tools = tools();
    for text in [
        "\n<function=get_weather>\n<parameter=city>\nGyőr\n</parameter>\n<parameter=unit>\ncelsius\n</parameter>\n</function>\n</tool_call>",
        "\n<function=get_weather>\n<parameter=city>\nNew York\n</parameter>\n</function>\n</tool_call>",
        "\n<function=set_timer>\n<parameter=minutes>\n2.5\n</parameter>\n<parameter=label>\ntwo\nlines\n</parameter>\n</function>\n</tool_call>",
        "\n<function=set_timer>\n<parameter=minutes>\n0\n</parameter>\n</function>\n</tool_call>",
        "\n<function=set_timer>\n<parameter=minutes>\n-0.5\n</parameter>\n</function>\n</tool_call>",
        "\n<function=set_timer>\n<parameter=minutes>\n1e3\n</parameter>\n</function>\n</tool_call>",
        "\n<function=now>\n</function>\n</tool_call>",
    ] {
        walk(&grammar, &tokenizer, &tools, text).unwrap_or_else(|e| panic!("{text:?}: {e}"));
        let parsed = parse_output(&format!("<tool_call>{text}"), &tools);
        validate_call(&parsed.tool_calls[0], &tools).unwrap();
    }
}

#[test]
fn grammar_rejects_invalid_calls() {
    let tokenizer = Tokenizer::from_file(TOKENIZER).unwrap();
    let grammar = ToolGrammar::new(&tokenizer).unwrap();
    let tools = tools();
    for text in [
        "\n<function=delete_everything>",
        "\n<function=get_weather>\n</function>",
        "\n<function=get_weather>\n<parameter=city>\nGyőr\n</parameter>\n<parameter=unit>\nkelvin",
        "\n<function=set_timer>\n<parameter=minutes>\nsoon",
        // Leading zeros are not JSON and would fail to parse.
        "\n<function=set_timer>\n<parameter=minutes>\n01\n</parameter>",
        "\n<function=set_timer>\n<parameter=minutes>\n-01\n</parameter>",
        "\n<function=get_weather>\n<parameter=nope>",
    ] {
        assert!(walk(&grammar, &tokenizer, &tools, text).is_err(), "the grammar allowed {text:?}");
    }
}
