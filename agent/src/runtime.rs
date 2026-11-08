//! Resource bounds and cooperative cancellation shared by the transport and inference loop.
use std::{io::BufRead, sync::{Arc, atomic::{AtomicBool, Ordering}}, time::{Duration, Instant}};
use anyhow::{Result, bail};
use serde_json::Value;

pub const MAX_FRAME: usize = 1024 * 1024;

#[derive(Clone)]
pub struct TurnControl {
    cancelled: Arc<AtomicBool>,
    deadline: Instant,
}

#[derive(Debug)]
pub struct Interrupted(pub &'static str);
impl std::fmt::Display for Interrupted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(self.0) }
}
impl std::error::Error for Interrupted {}

#[derive(Debug)]
pub struct UnknownToolOutcome(pub String);
impl std::fmt::Display for UnknownToolOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "tool outcome unknown: {} (not retried)", self.0) }
}
impl std::error::Error for UnknownToolOutcome {}

impl TurnControl {
    pub fn new(timeout: Duration) -> Self { Self { cancelled: Arc::new(AtomicBool::new(false)), deadline: Instant::now() + timeout } }
    pub fn cancel(&self) { self.cancelled.store(true, Ordering::Release); }
    pub fn check(&self) -> Result<()> {
        if self.cancelled.load(Ordering::Acquire) { return Err(Interrupted("cancelled").into()); }
        if Instant::now() >= self.deadline { return Err(Interrupted("deadline exceeded").into()); }
        Ok(())
    }
}

/// Reads one complete frame without ever allocating more than `limit + 1` bytes.
/// A framing error is terminal: callers must close the stream, not retry a partial frame.
pub fn read_json(reader: &mut impl BufRead, limit: usize) -> Result<Option<Value>> {
    let mut data = Vec::new();
    loop {
        let buf = reader.fill_buf()?;
        if buf.is_empty() {
            if data.is_empty() { return Ok(None); }
            bail!("truncated JSON frame");
        }
        let n = buf.iter().position(|b| *b == b'\n').map_or(buf.len(), |n| n + 1);
        if data.len() + n > limit { bail!("JSON frame exceeds {limit} bytes"); }
        let complete = buf[n - 1] == b'\n';
        data.extend_from_slice(&buf[..n]);
        reader.consume(n);
        if complete { return Ok(Some(serde_json::from_slice(&data)?)); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn frames_are_bounded_and_partial_eof_is_an_error() {
        assert!(read_json(&mut &b"{\"x\":123456}\n"[..], 5).is_err());
        assert!(read_json(&mut &b"{}"[..], 10).is_err());
        assert!(read_json(&mut &b""[..], 10).unwrap().is_none());
        assert_eq!(read_json(&mut &b"{}\n{}\n"[..], 10).unwrap(), Some(serde_json::json!({})));
    }
    #[test] fn cancellation_and_deadline_are_distinct() {
        let c = TurnControl::new(Duration::from_secs(10));
        c.clone().cancel();
        assert_eq!(c.check().unwrap_err().to_string(), "cancelled");
        assert_eq!(TurnControl::new(Duration::ZERO).check().unwrap_err().to_string(), "deadline exceeded");
    }
}
