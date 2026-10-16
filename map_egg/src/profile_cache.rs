//! Content-addressed candidate pools. Current leaf timing is deliberately not a key.
use crate::{
    emap_cover::{self, MappedCoverChoice},
    techlib::TechLib,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};
static MEMORY: OnceLock<Mutex<BTreeMap<String, Vec<MappedCoverChoice>>>> = OnceLock::new();
static MEMORY_HITS: AtomicU64 = AtomicU64::new(0);
static DISK_HITS: AtomicU64 = AtomicU64::new(0);
static BUILDS: AtomicU64 = AtomicU64::new(0);
static WRITE_FAILURES: AtomicU64 = AtomicU64::new(0);
static CORRUPT: AtomicU64 = AtomicU64::new(0);
static SERIAL: AtomicU64 = AtomicU64::new(0);
static FALLBACKS: AtomicU64 = AtomicU64::new(0);
#[derive(Serialize, Deserialize)]
struct Record {
    schema: u32,
    key: String,
    payload_sha256: String,
    covers: Vec<MappedCoverChoice>,
}
fn digest(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn env_limit(name: &str, default: usize, max: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
        .clamp(1, max)
}
fn limits() -> emap_cover::profile::Limits {
    emap_cover::profile::Limits {
        frontier: env_limit("MAP_EGG_PROFILE_FRONTIER", 8, 64),
        combinations: env_limit("MAP_EGG_PROFILE_COMBINATIONS", 20000, 1000000),
        visits: env_limit("MAP_EGG_PROFILE_VISITS", 4000, 100000),
    }
}
fn namespace(tech: &TechLib) -> String {
    static SOURCE: OnceLock<String> = OnceLock::new();
    let source = SOURCE.get_or_init(|| {
        let mut hash = Sha256::new();
        hash.update(b"mapped-profile-v1-load-independent-physical-inversions");
        for src in [
            include_str!("lib.rs"),
            include_str!("emap_cover.rs"),
            include_str!("emap_cover/profile.rs"),
            include_str!("techlib.rs"),
            include_str!("emap_egg.rs"),
            include_str!("profile_cache.rs"),
            include_str!("tt9.rs"),
            include_str!("params.rs"),
        ] {
            hash.update(src.as_bytes());
        }
        format!("{:x}", hash.finalize())
    });
    digest(
        &serde_json::to_vec(&(
            source,
            &tech.cells,
            tech.inv_area,
            tech.inv_delay,
            &tech.inv_variant,
            &tech.and2_variant,
        ))
        .unwrap(),
    )
}
fn key(expr: &str, tech: &TechLib, n: usize) -> String {
    let search = emap_cover::cut_mapping_params();
    key_with_search(
        expr,
        tech,
        n,
        limits(),
        (
            search.egg_iter_limit,
            search.egg_node_limit,
            search.max_cover_nodes,
        ),
    )
}
fn key_with_search(
    expr: &str,
    tech: &TechLib,
    n: usize,
    bounds: emap_cover::profile::Limits,
    search: (usize, usize, usize),
) -> String {
    let full = std::env::var("MAP_EGG_FULL_BUNDLE").is_ok_and(|v| v == "1");
    let narrow = emap_cover::profile::narrow_enabled();
    let phase_keep = emap_cover::profile::phase_keep_enabled();
    let data = serde_json::to_vec(&(namespace(tech), expr, n, bounds, search, full, narrow, phase_keep)).unwrap();
    digest(&data)
}
fn valid(c: &MappedCoverChoice, n: usize) -> bool {
    if c.graph.root >= c.graph.nodes.len()
        || c.pin_to_root_delay.len() != n
        || !c.area.is_finite()
        || !c.delay.is_finite()
        || c.area < 0.
        || c.delay < 0.
    {
        return false;
    }
    if c.used_leaf_indices.iter().any(|&i| i >= n)
        || c.pin_to_root_delay
            .iter()
            .any(|d| !d.is_finite() || *d < 0.)
    {
        return false;
    }
    c.graph
        .nodes
        .iter()
        .enumerate()
        .all(|(i, node)| match node {
            emap_cover::MappedCoverNode::Input(k) => *k < n,
            emap_cover::MappedCoverNode::Const(_) => true,
            emap_cover::MappedCoverNode::Cell { cell, fanins } => {
                fanins.len() == cell.n_inputs as usize
                    && cell.pin_delay.len() == fanins.len()
                    && cell.pin_names.len() == fanins.len()
                    && fanins.iter().all(|&j| j < i)
            }
        })
}
fn read(path: &std::path::Path, key: &str, n: usize) -> Option<Vec<MappedCoverChoice>> {
    if std::fs::metadata(path).ok()?.len() > 4 * 1024 * 1024 {
        return None;
    }
    let bytes = std::fs::read(path).ok()?;
    let record: Record = serde_json::from_slice(&bytes).ok()?;
    if bytes.len() > 4 * 1024 * 1024
        || record.schema != 1
        || record.key != key
        || record.covers.iter().any(|c| !valid(c, n))
        || digest(&serde_json::to_vec(&record.covers).ok()?) != record.payload_sha256
    {
        return None;
    }
    Some(record.covers)
}
fn write(path: &std::path::Path, key: &str, covers: &[MappedCoverChoice]) -> std::io::Result<()> {
    let record = Record {
        schema: 1,
        key: key.into(),
        payload_sha256: digest(&serde_json::to_vec(covers)?),
        covers: covers.to_vec(),
    };
    let bytes = serde_json::to_vec(&record)?;
    if bytes.len() > 4 * 1024 * 1024 {
        return Ok(());
    }
    let parent = path.parent().unwrap();
    std::fs::create_dir_all(parent)?;
    let mut count = 0;
    let mut total = 0u64;
    for entry in std::fs::read_dir(parent)? {
        let entry = entry?;
        if entry.path().extension().is_some_and(|e| e == "json") {
            count += 1;
            total += entry.metadata()?.len();
        }
    }
    if !path.exists() && (count >= 2048 || total + bytes.len() as u64 > 64 * 1024 * 1024) {
        return Ok(());
    }
    let tmp = path.with_extension(format!(
        "{}.{}.tmp",
        std::process::id(),
        SERIAL.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&tmp, bytes)?;
    let result = std::fs::rename(&tmp, path);
    if result.is_err() {
        let _ = std::fs::remove_file(tmp);
    }
    result
}
pub(crate) fn pool(expr: &str, tech: &TechLib, n: usize) -> Vec<MappedCoverChoice> {
    let key = key(expr, tech, n);
    let memory = MEMORY.get_or_init(|| Mutex::new(BTreeMap::new()));
    if let Some(c) = memory.lock().unwrap().get(&key).cloned() {
        MEMORY_HITS.fetch_add(1, Ordering::Relaxed);
        return c;
    }
    let disk = std::env::var_os("MAP_EGG_PROFILE_CACHE_DIR")
        .map(PathBuf::from)
        .map(|p| p.join(format!("{key}.json")));
    let covers = if let Some(c) = disk.as_ref().and_then(|p| read(p, &key, n)) {
        DISK_HITS.fetch_add(1, Ordering::Relaxed);
        c
    } else {
        if disk.as_ref().is_some_and(|p| p.exists()) {
            CORRUPT.fetch_add(1, Ordering::Relaxed);
        }
        BUILDS.fetch_add(1, Ordering::Relaxed);
        let covers = emap_cover::profile::extract(expr, tech, n, limits()).unwrap_or_default();
        if let Some(p) = &disk {
            if write(p, &key, &covers).is_err() {
                WRITE_FAILURES.fetch_add(1, Ordering::Relaxed);
            }
        }
        covers
    };
    let mut store = memory.lock().unwrap();
    if store.len() >= 1024 {
        if let Some(first) = store.keys().next().cloned() {
            store.remove(&first);
        }
    }
    store.insert(key, covers.clone());
    covers
}
pub(crate) fn fallback() {
    FALLBACKS.fetch_add(1, Ordering::Relaxed);
}
pub(crate) fn stats() -> String {
    format!(
        "memory_hits={} disk_hits={} builds={} corrupt={} write_failures={} legacy_fallbacks={}",
        MEMORY_HITS.load(Ordering::Relaxed),
        DISK_HITS.load(Ordering::Relaxed),
        BUILDS.load(Ordering::Relaxed),
        CORRUPT.load(Ordering::Relaxed),
        WRITE_FAILURES.load(Ordering::Relaxed),
        FALLBACKS.load(Ordering::Relaxed)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn keys_track_library_search_and_variable_order() {
        assert_eq!(
            digest(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let mut tech = TechLib::default();
        let bounds = emap_cover::profile::Limits::default();
        let k = key_with_search("(& a b)", &tech, 2, bounds, (2, 500, 100));
        assert_ne!(
            k,
            key_with_search("(& b a)", &tech, 2, bounds, (2, 500, 100))
        );
        assert_ne!(
            k,
            key_with_search("(& a b)", &tech, 3, bounds, (2, 500, 100))
        );
        assert_ne!(
            k,
            key_with_search("(& a b)", &tech, 2, bounds, (3, 500, 100))
        );
        assert_ne!(
            k,
            key_with_search(
                "(& a b)",
                &tech,
                2,
                emap_cover::profile::Limits {
                    frontier: 16,
                    ..bounds
                },
                (2, 500, 100)
            )
        );
        tech.inv_delay = 1.;
        assert_ne!(
            k,
            key_with_search("(& a b)", &tech, 2, bounds, (2, 500, 100))
        );
    }
    #[test]
    fn corrupt_payload_is_rejected() {
        let path =
            std::env::temp_dir().join(format!("profile-cache-test-{}.json", std::process::id()));
        write(&path, "key", &[]).unwrap();
        assert!(read(&path, "key", 2).is_some());
        assert!(read(&path, "wrong", 2).is_none());
        std::fs::write(&path, b"{broken}").unwrap();
        assert!(read(&path, "key", 2).is_none());
        std::fs::remove_file(path).unwrap();
    }
}

#[cfg(test)]
pub(crate) fn test_serial() -> u64 {
    SERIAL.fetch_add(1, Ordering::Relaxed)
}
