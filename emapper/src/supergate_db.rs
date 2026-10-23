//! Line-delimited JSON cache for multi-gate covers (truth-table keyed).
//!
//! Path: `EMAPPER_SUPERGATE_DB`; if unset, `supergates.jsonl` in the process working directory,
//! falling back to the repository copy next to the `emapper` crate (created on first append).
//! The database is keyed by `(n_inputs, truth table, objective, required-time bucket)` and stores
//! a full [`MappedCoverChoice`], i.e. a precomputed mapped sub-network.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::emap_cover::MappedCoverChoice;
use crate::techlib::MappingObjective;
use crate::tt9::{tt9_from_hex, tt9_to_hex, Tt9};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
enum ObjectiveTag {
    Area,
    Delay,
}

impl From<MappingObjective> for ObjectiveTag {
    fn from(value: MappingObjective) -> Self {
        match value {
            MappingObjective::AreaFirst => Self::Area,
            MappingObjective::DelayFirst => Self::Delay,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct SupergateKey {
    n_inputs: u8,
    tt: Tt9,
    objective: ObjectiveTag,
    req_bits: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SupergateRecord {
    n_inputs: u8,
    tt_hex: String,
    objective: ObjectiveTag,
    req_bits: u32,
    choice: MappedCoverChoice,
}

#[derive(Default)]
struct SupergateDb {
    path: PathBuf,
    map: HashMap<SupergateKey, MappedCoverChoice>,
    /// Best `req_bits` bucket per `(n_inputs, tt, objective)`, used by [`SupergateDb::get_relaxed`]
    /// when the exact required-time bucket has no entry (epmapping runs unconstrained and would
    /// otherwise never see the area-oriented, timing-bucketed records).
    best_req: HashMap<(u8, Tt9, ObjectiveTag), u32>,
    best_score: HashMap<(u8, Tt9, ObjectiveTag), f32>,
}

fn score(objective: ObjectiveTag, choice: &MappedCoverChoice) -> f32 {
    match objective {
        ObjectiveTag::Area => choice.area,
        ObjectiveTag::Delay => choice.delay,
    }
}

impl SupergateDb {
    fn load(path: &Path) -> Self {
        let mut db = Self {
            path: path.to_path_buf(),
            ..Self::default()
        };
        let Ok(file) = File::open(path) else {
            return db;
        };
        let reader = BufReader::new(file);
        for line in reader.lines().map_while(Result::ok) {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let Ok(rec) = serde_json::from_str::<SupergateRecord>(trimmed) else {
                continue;
            };
            let Some(tt) = tt9_from_hex(&rec.tt_hex) else {
                continue;
            };
            let key = SupergateKey {
                n_inputs: rec.n_inputs,
                tt,
                objective: rec.objective,
                req_bits: rec.req_bits,
            };
            db.update_best(&key, &rec.choice);
            db.map.insert(key, rec.choice);
        }
        db
    }

    /// Prefer the lowest area (resp. delay) cover for the class; break ties towards the
    /// unconstrained `u32::MAX` bucket.
    fn update_best(&mut self, key: &SupergateKey, choice: &MappedCoverChoice) {
        let class = (key.n_inputs, key.tt, key.objective);
        let s = score(key.objective, choice);
        let better = match self.best_score.get(&class) {
            None => true,
            Some(&cur) => {
                s < cur - 1e-9
                    || (s <= cur + 1e-9
                        && key.req_bits == u32::MAX
                        && self.best_req.get(&class) != Some(&u32::MAX))
            }
        };
        if better {
            self.best_score.insert(class, s);
            self.best_req.insert(class, key.req_bits);
        }
    }

    fn get(
        &self,
        n_inputs: u8,
        tt: Tt9,
        objective: MappingObjective,
        req_bits: u32,
    ) -> Option<MappedCoverChoice> {
        self.map
            .get(&SupergateKey {
                n_inputs,
                tt,
                objective: objective.into(),
                req_bits,
            })
            .cloned()
    }

    /// Look up the best stored cover for the class, ignoring the exact required-time bucket.
    fn get_relaxed(
        &self,
        n_inputs: u8,
        tt: Tt9,
        objective: MappingObjective,
    ) -> Option<MappedCoverChoice> {
        let tag: ObjectiveTag = objective.into();
        let req_bits = *self.best_req.get(&(n_inputs, tt, tag))?;
        self.map
            .get(&SupergateKey {
                n_inputs,
                tt,
                objective: tag,
                req_bits,
            })
            .cloned()
    }

    fn insert(
        &mut self,
        n_inputs: u8,
        tt: Tt9,
        objective: MappingObjective,
        req_bits: u32,
        choice: &MappedCoverChoice,
    ) {
        let key = SupergateKey {
            n_inputs,
            tt,
            objective: objective.into(),
            req_bits,
        };
        if self.map.contains_key(&key) {
            return;
        }
        self.update_best(&key, choice);
        self.map.insert(key, choice.clone());
        let rec = SupergateRecord {
            n_inputs,
            tt_hex: tt9_to_hex(&tt),
            objective: objective.into(),
            req_bits,
            choice: choice.clone(),
        };
        let Ok(line) = serde_json::to_string(&rec) else {
            return;
        };
        let mut fh = match OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
        {
            Ok(f) => f,
            Err(_) => return,
        };
        let _ = writeln!(fh, "{line}");
    }
}

#[derive(Default)]
struct DbState {
    initialized: bool,
    db: Option<SupergateDb>,
}

thread_local! {
    static DB_STATE: RefCell<DbState> = RefCell::new(DbState::default());
}

fn supergate_db_path() -> PathBuf {
    if let Some(p) = std::env::var("EMAPPER_SUPERGATE_DB")
        .ok()
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
    {
        return p;
    }
    /* Prefer the CWD copy (scripts run from the repo root), then the repo copy next to the crate so
    that binaries started elsewhere still find the 4000+ precomputed covers. */
    let local = PathBuf::from("supergates.jsonl");
    if local.exists() {
        return local;
    }
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../supergates.jsonl");
    if repo.exists() {
        return repo;
    }
    local
}

fn init_if_needed(state: &mut DbState) {
    if state.initialized {
        return;
    }
    state.initialized = true;
    let path = supergate_db_path();
    state.db = Some(SupergateDb::load(&path));
}

pub fn lookup(
    n_inputs: u8,
    tt: Tt9,
    objective: MappingObjective,
    req_bits: u32,
) -> Option<MappedCoverChoice> {
    DB_STATE.with(|slot| {
        let mut state = slot.borrow_mut();
        init_if_needed(&mut state);
        state
            .db
            .as_ref()
            .and_then(|db| db.get(n_inputs, tt, objective, req_bits))
    })
}

/// Every cover stored for `(n_inputs, tt)`, up to `cap` of them.
///
/// The database keeps one cover per `(objective, required-time)` bucket, so a function can have
/// several recorded implementations. Callers that want to choose among them — for instance a
/// unit-depth/area Pareto selection — use this instead of the single best [`lookup`].
#[must_use]
pub fn lookup_all(n_inputs: u8, tt: Tt9, cap: usize) -> Vec<MappedCoverChoice> {
    DB_STATE.with(|slot| {
        let mut state = slot.borrow_mut();
        init_if_needed(&mut state);
        let Some(db) = state.db.as_ref() else {
            return Vec::new();
        };
        db.map
            .iter()
            .filter(|(k, _)| k.n_inputs == n_inputs && k.tt == tt)
            .map(|(_, v)| v.clone())
            .take(cap)
            .collect()
    })
}

/// Like [`lookup`] but ignores the required-time bucket, returning the lowest-area (resp.
/// lowest-delay) cover stored for the class.
///
/// This is what unconstrained post-mapping optimization needs: the database also contains records
/// produced under timing constraints, and their bucket never equals `u32::MAX`.
pub fn lookup_relaxed(
    n_inputs: u8,
    tt: Tt9,
    objective: MappingObjective,
) -> Option<MappedCoverChoice> {
    DB_STATE.with(|slot| {
        let mut state = slot.borrow_mut();
        init_if_needed(&mut state);
        state
            .db
            .as_ref()
            .and_then(|db| db.get_relaxed(n_inputs, tt, objective))
    })
}

pub fn store(
    n_inputs: u8,
    tt: Tt9,
    objective: MappingObjective,
    req_bits: u32,
    choice: &MappedCoverChoice,
) {
    DB_STATE.with(|slot| {
        let mut state = slot.borrow_mut();
        init_if_needed(&mut state);
        if let Some(db) = state.db.as_mut() {
            db.insert(n_inputs, tt, objective, req_bits, choice);
        }
    });
}

/// Number of entries loaded for this thread (`0` when the database file is missing).
///
/// Loading happens lazily on the first [`lookup`]/[`store`] call.
#[must_use]
pub fn loaded_entries() -> usize {
    DB_STATE.with(|slot| {
        let mut state = slot.borrow_mut();
        init_if_needed(&mut state);
        state.db.as_ref().map_or(0, |db| db.map.len())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// These tests exercise the **repository** database, which is the only one guaranteed to hold
    /// both objective classes. `supergate_db_path` prefers `EMAPPER_SUPERGATE_DB`, so a developer
    /// running `EMAPPER_SUPERGATE_DB=target/scratch.jsonl cargo test` would otherwise test against a
    /// partial run-local cache and fail with "database should contain both objectives". Force the
    /// repo copy for the duration of the test process (the path is resolved once, on first use).
    fn use_repo_database() {
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../supergates.jsonl");
        if repo.exists() {
            std::env::set_var("EMAPPER_SUPERGATE_DB", &repo);
        }
    }

    #[test]
    fn repo_database_loads_entries() {
        use_repo_database();
        let n = loaded_entries();
        eprintln!("supergate_db loaded_entries = {n}");
        assert!(
            n > 0,
            "expected the repository supergates.jsonl to be found"
        );
    }

    #[test]
    fn relaxed_lookup_resolves_every_objective_class() {
        use_repo_database();
        let _ = loaded_entries();
        DB_STATE.with(|slot| {
            let state = slot.borrow();
            let Some(db) = state.db.as_ref() else { return };
            for tag in [ObjectiveTag::Area, ObjectiveTag::Delay] {
                let objective = match tag {
                    ObjectiveTag::Area => MappingObjective::AreaFirst,
                    ObjectiveTag::Delay => MappingObjective::DelayFirst,
                };
                let key = db
                    .map
                    .keys()
                    .find(|k| k.objective == tag)
                    .cloned()
                    .expect("database should contain both objectives");
                let hit = db.get_relaxed(key.n_inputs, key.tt, objective);
                assert!(
                    hit.is_some(),
                    "relaxed lookup missed a stored {tag:?} class"
                );
            }
        });
    }
}
