//! Durable replacement: a failed write never truncates the previous snapshot.
use std::{fs::{File, OpenOptions}, io::Write, path::Path, sync::atomic::{AtomicU64, Ordering}};
use anyhow::{Context, Result, bail};
static SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub fn unique_id() -> String {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos();
    format!("{nanos}-{}-{}", std::process::id(), SEQUENCE.fetch_add(1, Ordering::Relaxed))
}

pub fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 100 && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    let temp = parent.join(format!(".{}.tmp", unique_id()));
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new().write(true).create_new(true).open(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&temp, path).with_context(|| format!("replacing {}", path.display()))?;
        #[cfg(unix)]
        File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() { let _ = std::fs::remove_file(&temp); }
    result
}

/// OS-owned lock; a crashed process releases it without a stale-lock timeout.
pub fn model_lease(path: &Path) -> Result<File> {
    if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
    let file = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(path)?;
    if file.try_lock().is_err() { bail!("another agent already owns the model; connect to it or stop it before loading another instance"); }
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn replacement_and_model_lease() {
        let dir = std::env::temp_dir().join(unique_id());
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("saved.json");
        atomic_write(&path, b"old").unwrap();
        atomic_write(&path, b"new").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
        let lock = model_lease(&dir.join("lock")).unwrap();
        assert!(model_lease(&dir.join("lock")).is_err());
        drop(lock);
        drop(model_lease(&dir.join("lock")).unwrap());
        std::fs::remove_file(path).unwrap();
        std::fs::remove_file(dir.join("lock")).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }
    #[test] fn identifiers_cannot_escape_storage() {
        for id in ["../x", "..", "C:\\x", "x/y", "", "a.b"] { assert!(!valid_id(id)); }
        assert!(valid_id(&unique_id()));
    }
}
