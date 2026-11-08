//! Token sampling over one row of logits: greedy, or temperature + top-k + top-p, with an
//! optional presence penalty against repetition.

use serde::Deserialize;
use toktrie::SimpleVob;

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct SamplingOptions {
    /// 0 means greedy.
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
    /// Subtracted from the logit of every token already generated (0 turns it off). Qwen
    /// suggests 1.5-2 against endless repetition. Not applied under a grammar mask, where
    /// repeating a token (a digit in an argument) can be exactly right.
    pub presence_penalty: f32,
    /// Added to the `<tool_call>` logit before sampling, while decoding is still free text.
    /// Positive makes a call more likely; negative steers the model to answer instead.
    #[serde(default)]
    pub tool_call_bias: f32,
}

impl Default for SamplingOptions {
    /// Temperature, top-k and top-p follow the model's `generation_config.json`.
    fn default() -> Self {
        Self { temperature: 0.6, top_k: 20, top_p: 0.95, seed: 42, presence_penalty: 0.0, tool_call_bias: 0.0 }
    }
}

pub struct Sampler {
    opts: SamplingOptions,
    state: u64,
    scratch: Vec<(f32, u32)>,
    /// Tokens generated so far, for the presence penalty.
    seen: Vec<bool>,
}

impl Sampler {
    pub fn new(opts: SamplingOptions) -> Self {
        // xorshift has a fixed point at zero.
        let state = opts.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        Self { opts, state, scratch: Vec::new(), seen: Vec::new() }
    }

    fn next_f32(&mut self) -> f32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        ((x >> 40) as f32) / ((1u64 << 24) as f32)
    }

    /// `mask` comes from the grammar; tokens outside it are never chosen.
    /// Fails closed instead of producing token zero for an impossible distribution.
    pub fn try_sample(&mut self, logits: &[f32], mask: Option<&SimpleVob>) -> anyhow::Result<u32> {
        let mut finite = false;
        for (i, &value) in logits.iter().enumerate() {
            if mask.is_some_and(|m| i >= m.len() || !m.is_allowed(i as u32)) { continue; }
            if value.is_nan() || value == f32::INFINITY { anyhow::bail!("non-finite model logits"); }
            finite |= value.is_finite();
        }
        if !finite { anyhow::bail!("no finite token is allowed by the sampling mask"); }
        Ok(self.sample(logits, mask))
    }

    pub fn sample(&mut self, logits: &[f32], mask: Option<&SimpleVob>) -> u32 {
        let token = self.choose(logits, mask);
        if self.opts.presence_penalty != 0.0 {
            if self.seen.len() <= token as usize {
                self.seen.resize(logits.len().max(token as usize + 1), false);
            }
            self.seen[token as usize] = true;
        }
        token
    }

    fn choose(&mut self, logits: &[f32], mask: Option<&SimpleVob>) -> u32 {
        let allowed = |i: usize| mask.is_none_or(|m| i < m.len() && m.is_allowed(i as u32));
        let penalty = if mask.is_none() { self.opts.presence_penalty } else { 0.0 };
        let seen = &self.seen;
        let logit = |i: usize, l: f32| if penalty != 0.0 && seen.get(i).copied().unwrap_or(false) { l - penalty } else { l };

        if self.opts.temperature <= 0.0 {
            let mut best = (f32::NEG_INFINITY, 0u32);
            for (i, &l) in logits.iter().enumerate() {
                let l = logit(i, l);
                if l > best.0 && allowed(i) {
                    best = (l, i as u32);
                }
            }
            return best.1;
        }

        self.scratch.clear();
        self.scratch.extend(logits.iter().enumerate().filter(|(i, _)| allowed(*i)).map(|(i, &l)| (logit(i, l), i as u32)));
        if self.scratch.is_empty() {
            return 0;
        }

        let k = self.opts.top_k.max(1).min(self.scratch.len());
        if k < self.scratch.len() {
            self.scratch.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
            self.scratch.truncate(k);
        }
        self.scratch.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

        let t = self.opts.temperature;
        let top = self.scratch[0].0;
        let weights: Vec<f32> = self.scratch.iter().map(|(l, _)| ((l - top) / t).exp()).collect();
        let total: f32 = weights.iter().sum();

        // Keep the smallest prefix reaching top_p of the probability mass.
        let mut kept = 0.0;
        let mut n = 0;
        for w in &weights {
            kept += w;
            n += 1;
            if kept >= self.opts.top_p * total {
                break;
            }
        }

        let mut r = self.next_f32() * kept;
        for (i, w) in weights[..n].iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return self.scratch[i].1;
            }
        }
        self.scratch[n - 1].1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn impossible_and_corrupt_distributions_are_errors() {
        let mut sampler = Sampler::new(SamplingOptions::default());
        let empty = SimpleVob::alloc(2);
        assert!(sampler.try_sample(&[1.0,2.0],Some(&empty)).is_err());
        assert!(sampler.try_sample(&[f32::NAN],None).is_err());
        assert!(sampler.try_sample(&[f32::NEG_INFINITY],None).is_err());
        assert!(sampler.try_sample(&[],None).is_err());
    }

    #[test]
    fn greedy_picks_max_and_respects_mask() {
        let logits = [0.0, 3.0, 1.0, 2.0];
        let mut s = Sampler::new(SamplingOptions { temperature: 0.0, ..Default::default() });
        assert_eq!(s.sample(&logits, None), 1);
        let mut mask = SimpleVob::alloc(4);
        mask.allow_token(2);
        mask.allow_token(3);
        assert_eq!(s.sample(&logits, Some(&mask)), 3);
    }

    #[test]
    fn stochastic_respects_mask_and_top_k() {
        let logits = [5.0, 4.9, 4.8, -9.0];
        let mut mask = SimpleVob::alloc(4);
        mask.allow_token(1);
        mask.allow_token(3);
        let mut s = Sampler::new(SamplingOptions { temperature: 1.0, top_k: 1, top_p: 1.0, seed: 7, ..Default::default() });
        for _ in 0..50 {
            assert_eq!(s.sample(&logits, Some(&mask)), 1);
        }
    }

    #[test]
    fn same_seed_is_deterministic() {
        let logits: Vec<f32> = (0..100).map(|i| (i as f32 * 0.7).sin()).collect();
        let draw = |seed| {
            let mut s = Sampler::new(SamplingOptions { temperature: 1.0, top_k: 50, top_p: 0.95, seed, ..Default::default() });
            (0..30).map(|_| s.sample(&logits, None)).collect::<Vec<_>>()
        };
        assert_eq!(draw(1), draw(1));
        assert_ne!(draw(1), draw(2));
    }

    #[test]
    fn tool_call_bias_flips_greedy_when_added_to_logits() {
        let mut logits = [1.0, 3.0, 0.0];
        let mut s = Sampler::new(SamplingOptions { temperature: 0.0, ..Default::default() });
        assert_eq!(s.sample(&logits, None), 1);
        logits[0] += 4.0;
        let mut s = Sampler::new(SamplingOptions { temperature: 0.0, ..Default::default() });
        assert_eq!(s.sample(&logits, None), 0);
        logits[0] -= 12.0;
        let mut s = Sampler::new(SamplingOptions { temperature: 0.0, ..Default::default() });
        assert_eq!(s.sample(&logits, None), 1);
    }

    #[test]
    fn presence_penalty_avoids_repeats_but_not_under_a_mask() {
        let logits = [0.0, 3.0, 2.5, 0.0];
        let mut s = Sampler::new(SamplingOptions { temperature: 0.0, presence_penalty: 1.0, ..Default::default() });
        assert_eq!(s.sample(&logits, None), 1);
        assert_eq!(s.sample(&logits, None), 2, "token 1 is penalised once generated");
        let mut mask = SimpleVob::alloc(4);
        mask.allow_token(1);
        mask.allow_token(2);
        assert_eq!(s.sample(&logits, Some(&mask)), 1, "no penalty inside the grammar");
    }
}
