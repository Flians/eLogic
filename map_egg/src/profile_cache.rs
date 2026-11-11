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
#[derive(Default)]
struct MemoryPool {
    entries: BTreeMap<String, MemoryEntry>,
    clock: u64,
    payload_bytes: usize,
    peak_payload_bytes: usize,
    evictions: u64,
}
struct MemoryEntry { covers: Vec<MappedCoverChoice>, touched: u64, payload_bytes: usize }
impl MemoryPool {
    fn get(&mut self, key: &str) -> Option<Vec<MappedCoverChoice>> {
        self.clock += 1;
        let tick = self.clock;
        self.entries.get_mut(key).map(|entry| {entry.touched=tick;entry.covers.clone()})
    }
    // Bound serialized payloads and keys, rather than claiming an exact RSS cap.
    fn insert(&mut self, key: String, covers: Vec<MappedCoverChoice>, capacity: usize, budget: usize) {
        let bytes=key.len()+serde_json::to_vec(&covers).map_or(usize::MAX-key.len(), |v|v.len());
        if bytes>budget || capacity==0 {return;}
        if let Some(old)=self.entries.remove(&key) {self.payload_bytes-=old.payload_bytes;}
        while self.entries.len()>=capacity || self.payload_bytes+bytes>budget {
            let Some(victim)=self.entries.iter().min_by_key(|(_,e)|e.touched).map(|(k,_)|k.clone()) else {break;};
            let old=self.entries.remove(&victim).unwrap();self.payload_bytes-=old.payload_bytes;self.evictions+=1;
        }
        self.clock+=1;self.payload_bytes+=bytes;self.peak_payload_bytes=self.peak_payload_bytes.max(self.payload_bytes);
        self.entries.insert(key,MemoryEntry{covers,touched:self.clock,payload_bytes:bytes});
    }
}
static MEMORY: OnceLock<Mutex<MemoryPool>> = OnceLock::new();
pub(crate) static MEMORY_HITS: AtomicU64 = AtomicU64::new(0);
pub(crate) static DISK_HITS: AtomicU64 = AtomicU64::new(0);
pub(crate) static BUILDS: AtomicU64 = AtomicU64::new(0);
pub(crate) static WRITE_FAILURES: AtomicU64 = AtomicU64::new(0);
pub(crate) static CORRUPT: AtomicU64 = AtomicU64::new(0);
pub(crate) static SERIAL: AtomicU64 = AtomicU64::new(0);
static FALLBACKS: AtomicU64 = AtomicU64::new(0);
static READONLY_LOOKUPS: AtomicU64 = AtomicU64::new(0);
static READONLY_HITS: AtomicU64 = AtomicU64::new(0);
static PRIVATE_DISK_HITS: AtomicU64 = AtomicU64::new(0);
static DISK_WRITES: AtomicU64 = AtomicU64::new(0);
static BUILD_NS: AtomicU64=AtomicU64::new(0);
static READ_NS: AtomicU64=AtomicU64::new(0);
static WRITE_NS: AtomicU64=AtomicU64::new(0);
static FALLBACK_NS: AtomicU64=AtomicU64::new(0);
pub(crate) struct PhaseClock { start: Option<std::time::Instant>, counter: &'static AtomicU64 }
impl Drop for PhaseClock {fn drop(&mut self){if let Some(start)=self.start{self.counter.fetch_add(start.elapsed().as_nanos().min(u64::MAX as u128)as u64,Ordering::Relaxed);}}}
fn phase_clock(counter: &'static AtomicU64)->PhaseClock {
    static ENABLED:OnceLock<bool>=OnceLock::new();
    let enabled=*ENABLED.get_or_init(||std::env::var("MAP_EGG_PROFILE_TIMING").is_ok_and(|v|v=="1"));
    PhaseClock{start:enabled.then(std::time::Instant::now),counter}
}
pub(crate) fn fallback_clock()->PhaseClock {phase_clock(&FALLBACK_NS)}

#[derive(Serialize, Deserialize)]
struct Record {
    schema: u32,
    key: String,
    payload_sha256: String,
    covers: Vec<MappedCoverChoice>,
}
pub(crate) fn digest(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn env_limit(name: &str, default: usize, max: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
        .clamp(1, max)
}
pub(crate) fn limits() -> emap_cover::profile::Limits {
    emap_cover::profile::Limits {
        frontier: env_limit("MAP_EGG_PROFILE_FRONTIER", 8, 64),
        combinations: env_limit("MAP_EGG_PROFILE_COMBINATIONS", 20000, 1000000),
        visits: env_limit("MAP_EGG_PROFILE_VISITS", 4000, 100000),
    }
}
pub(crate) fn namespace(tech: &TechLib) -> String {
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
            include_str!("tt_profile_cache.rs"),
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
fn key(expr: &str, n: usize) -> String {
    // The production profile library is immutable and process-owned. Do not
    // serialize all cells on every query; generic key tests still hash their
    // supplied library afresh, including timing changes.
    static NAMESPACE: OnceLock<String> = OnceLock::new();
    let namespace = NAMESPACE.get_or_init(|| namespace(crate::profile_techlib_store()));
    let search = emap_cover::cut_mapping_params();
    key_with_namespace(
        expr,
        namespace,
        n,
        limits(),
        (
            search.egg_iter_limit,
            search.egg_node_limit,
            search.max_cover_nodes,
        ),
    )
}
#[cfg(test)]
fn key_with_search(
    expr: &str,
    tech: &TechLib,
    n: usize,
    bounds: emap_cover::profile::Limits,
    search: (usize, usize, usize),
) -> String {
    key_with_namespace(expr, &namespace(tech), n, bounds, search)
}
pub(crate) fn key_with_namespace(
    expr: &str,
    namespace: &str,
    n: usize,
    bounds: emap_cover::profile::Limits,
    search: (usize, usize, usize),
) -> String {
    let full = std::env::var("MAP_EGG_FULL_BUNDLE").is_ok_and(|v| v == "1");
    let narrow = emap_cover::profile::narrow_enabled();
    let phase_keep = emap_cover::profile::phase_keep_enabled();
    let data = serde_json::to_vec(&(namespace, expr, n, bounds, search, full, narrow, phase_keep)).unwrap();
    digest(&data)
}
pub(crate) fn valid(c: &MappedCoverChoice, n: usize) -> bool {
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
// Read-only packages contain the same immutable, content-addressed records as
// private pools. No directory creation, repair, eviction or write targets this path.
fn read_pools(shared: Option<&std::path::Path>, private: Option<&std::path::Path>,
              key: &str, n: usize) -> Option<Vec<MappedCoverChoice>> {
    for (path, readonly) in [(shared, true), (private, false)] {
        let Some(path) = path else {continue};
        if readonly {READONLY_LOOKUPS.fetch_add(1, Ordering::Relaxed);}
        if let Some(covers) = read(path, key, n) {
            DISK_HITS.fetch_add(1, Ordering::Relaxed);
            if readonly {READONLY_HITS.fetch_add(1, Ordering::Relaxed);}
            else {PRIVATE_DISK_HITS.fetch_add(1, Ordering::Relaxed);}
            return Some(covers);
        }
        if path.exists() {CORRUPT.fetch_add(1, Ordering::Relaxed);}
    }
    None
}
pub(crate) fn pool(expr: &str, n: usize, expression_arity: u8, tt: &crate::tt9::Tt9) -> Vec<MappedCoverChoice> {
    if !std::env::var("MAP_EGG_PROFILE_TT_CACHE").is_ok_and(|v| v == "0") {
        return crate::tt_profile_cache::pool(expr,n,expression_arity,tt);
    }
    let tech = crate::profile_techlib_store();
    let key = key(expr, n);
    let memory = MEMORY.get_or_init(|| Mutex::new(MemoryPool::default()));
    if let Some(c) = memory.lock().unwrap().get(&key) {
        MEMORY_HITS.fetch_add(1, Ordering::Relaxed);
        return c;
    }
    let disk = std::env::var_os("MAP_EGG_PROFILE_CACHE_DIR")
        .map(PathBuf::from)
        .map(|p| p.join(format!("{key}.json")));
    let shared = std::env::var_os("MAP_EGG_PROFILE_READONLY_DIR")
        .filter(|p| !p.is_empty()).map(PathBuf::from)
        .map(|p| p.join(format!("{key}.json")));
    let covers = if let Some(c) = {
        let _timer=phase_clock(&READ_NS);
        read_pools(shared.as_deref(), disk.as_deref(), &key, n)
    } {
        c
    } else {
        BUILDS.fetch_add(1, Ordering::Relaxed);
        let covers = {let _timer=phase_clock(&BUILD_NS);emap_cover::profile::extract(expr, tech, n, limits()).unwrap_or_default()};
        if let Some(p) = &disk {
            let _timer=phase_clock(&WRITE_NS);
            match write(p, &key, &covers) {
                Err(_) => {WRITE_FAILURES.fetch_add(1, Ordering::Relaxed);},
                Ok(()) => {if p.is_file() {DISK_WRITES.fetch_add(1, Ordering::Relaxed);}},
            }
        }
        covers
    };
    let mut store = memory.lock().unwrap();
    store.insert(key, covers.clone(),
        env_limit("MAP_EGG_PROFILE_MEMORY_ENTRIES",2048,8192),
        env_limit("MAP_EGG_PROFILE_MEMORY_BYTES",64*1024*1024,512*1024*1024));
    covers
}
pub(crate) fn fallback() {
    FALLBACKS.fetch_add(1, Ordering::Relaxed);
}
pub(crate) fn stats() -> String {
    let timing=format!("profile_build_seconds={:.6} profile_read_seconds={:.6} profile_write_seconds={:.6} profile_fallback_seconds={:.6}",BUILD_NS.load(Ordering::Relaxed)as f64/1e9,READ_NS.load(Ordering::Relaxed)as f64/1e9,WRITE_NS.load(Ordering::Relaxed)as f64/1e9,FALLBACK_NS.load(Ordering::Relaxed)as f64/1e9);
    let retention=MEMORY.get().map(|m|{let m=m.lock().unwrap();format!("profile_entries={} profile_evictions={} profile_payload_bytes={} profile_peak_payload_bytes={}",m.entries.len(),m.evictions,m.payload_bytes,m.peak_payload_bytes)}).unwrap_or_default();
    format!(
        "memory_hits={} disk_hits={} builds={} corrupt={} write_failures={} legacy_fallbacks={} {}",
        MEMORY_HITS.load(Ordering::Relaxed),
        DISK_HITS.load(Ordering::Relaxed),
        BUILDS.load(Ordering::Relaxed),
        CORRUPT.load(Ordering::Relaxed),
        WRITE_FAILURES.load(Ordering::Relaxed),
        FALLBACKS.load(Ordering::Relaxed),
        format!("{} {} {} readonly_lookups={} readonly_hits={} private_disk_hits={} profile_writes={}",crate::tt_profile_cache::stats(),retention,timing,
            READONLY_LOOKUPS.load(Ordering::Relaxed),READONLY_HITS.load(Ordering::Relaxed),
            PRIVATE_DISK_HITS.load(Ordering::Relaxed),DISK_WRITES.load(Ordering::Relaxed))
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retention_is_recent_and_payload_bounded() {
        let mut pool=MemoryPool::default();
        pool.insert("a".into(),vec![],2,100);
        pool.insert("b".into(),vec![],2,100);
        assert!(pool.get("a").is_some());
        pool.insert("c".into(),vec![],2,100);
        assert!(pool.get("b").is_none());
        assert!(pool.get("a").is_some());
        pool.insert("d".into(),vec![],10,3);
        assert!(pool.get("a").is_none());
        assert!(pool.get("d").is_some());
        assert!(pool.payload_bytes<=3);
        pool.insert("oversize".into(),vec![],10,3);
        assert!(pool.get("oversize").is_none());
    }
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
    fn readonly_pool_precedes_private_and_never_repairs_shared_files() {
        let dir=std::env::temp_dir().join(format!("profile-readonly-{}-{}",std::process::id(),test_serial()));
        std::fs::create_dir_all(&dir).unwrap();
        let shared=dir.join("shared.json");let private=dir.join("private.json");
        write(&shared,"key",&[]).unwrap();
        std::fs::write(&private,b"bad private record").unwrap();
        let original=std::fs::read(&shared).unwrap();
        assert!(read_pools(Some(&shared),Some(&private),"key",2).is_some());
        assert_eq!(std::fs::read(&shared).unwrap(),original);
        std::fs::write(&shared,b"bad shared record").unwrap();
        write(&private,"key",&[]).unwrap();
        assert!(read_pools(Some(&shared),Some(&private),"key",2).is_some());
        assert_eq!(std::fs::read(&shared).unwrap(),b"bad shared record");
        assert!(read_pools(Some(&shared),Some(&private),"wrong namespace",2).is_none());
        std::fs::remove_dir_all(dir).unwrap();
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
