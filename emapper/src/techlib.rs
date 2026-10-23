use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs,
    path::Path,
    sync::{Arc, OnceLock},
    time::UNIX_EPOCH,
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::emap_egg::EggRewriteBundle;
use crate::tt9::{
    permute_negate_tt9, tt9_bit_get, tt9_bit_set, tt9_from_hex, tt9_to_hex, tt9_zero, Tt9,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum MappingObjective {
    AreaFirst,
    #[default]
    DelayFirst,
}

/// How candidate variants are ranked by **area** during mapping.
///
/// - [`MappingAreaCost::Physical`]: [`TechLib::variant_area_physical_invs`] (netlist-accurate invs).
/// - [`MappingAreaCost::Flow`]: mockturtle-style `gate.area + Σ leaf_flow[pin]` (area-flow proxy).
///
/// Final `NodeBinding.area` in emap is always **physical** for reporting; flow is for comparison only.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum MappingAreaCost {
    /// Physical inverter + cell area (matches emitted Verilog).
    Physical,
    /// Gate area plus per-pin accumulated flow (matches mockturtle/ABC-style mapper).
    #[default]
    Flow,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PhaseArrival {
    pub pos: f32,
    pub neg: f32,
}

#[derive(Debug, Error)]
pub enum GenlibError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse error on line {line}: {msg}")]
    Parse { line: usize, msg: String },
    #[error("expression uses too many inputs ({inputs} > {max}) for cell {cell}")]
    TooManyInputs {
        cell: String,
        inputs: usize,
        max: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    Const(bool),
    Var(String),
    Not(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
}

impl Expr {
    fn vars(&self, out: &mut BTreeSet<String>) {
        match self {
            Self::Const(_) => {}
            Self::Var(v) => {
                out.insert(v.clone());
            }
            Self::Not(a) => a.vars(out),
            Self::And(a, b) | Self::Or(a, b) => {
                a.vars(out);
                b.vars(out);
            }
        }
    }

    fn eval(&self, env: &HashMap<&str, bool>) -> bool {
        match self {
            Self::Const(c) => *c,
            Self::Var(v) => *env.get(v.as_str()).unwrap_or(&false),
            Self::Not(a) => !a.eval(env),
            Self::And(a, b) => a.eval(env) & b.eval(env),
            Self::Or(a, b) => a.eval(env) | b.eval(env),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cell {
    pub name: String,
    pub area: f32,
    /// Output pin name (e.g. "Y" or "z").
    pub out_pin: String,
    pub expr: Expr,
    /// Pin block delay (max of rise/fall block).
    pub pin_delay: BTreeMap<String, f32>,
    /// Ordered variable names (sorted) for the truth table / pin order.
    pub vars: Vec<String>,
    /// Truth table for `expr` with `vars` order (LSB is assignment 0); up to 9 inputs (512 bits).
    pub tt: Tt9,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CellVariant {
    pub cell_name: String,
    /// Output pin name.
    pub out_pin: String,
    pub n_inputs: u8,
    /// For each input i in the *variant*, which original variable index it connects to.
    pub perm: Vec<u8>,
    /// Input inversion mask on variant inputs.
    pub neg_mask: u16,
    /// Whether output is inverted.
    pub out_neg: bool,
    pub area: f32,
    /// Pin delays corresponding to variant input order (before optional input inverter).
    pub pin_delay: Vec<f32>,
    /// Pin names corresponding to variant input order (before optional input inverter).
    pub pin_names: Vec<String>,
}

#[derive(Debug, Default)]
pub struct TechLib {
    pub cells: Vec<Cell>,
    /// Smallest inverter (if present).
    pub inv_area: f32,
    pub inv_delay: f32,
    pub inv_cell: Option<String>,
    pub inv_in_pin: Option<String>,
    pub inv_out_pin: Option<String>,
    pub inv_variant: Option<CellVariant>,
    /// Smallest exact 2-input AND gate (if present).
    pub and2_area: f32,
    pub and2_delay: [f32; 2],
    pub and2_variant: Option<CellVariant>,
    /// Smallest buffer (if present).
    pub buf_area: f32,
    pub buf_delay: f32,
    pub buf_cell: Option<String>,
    pub buf_in_pin: Option<String>,
    pub buf_out_pin: Option<String>,
    /// Constant drivers (if present).
    pub const0_cell: Option<String>,
    pub const0_out_pin: Option<String>,
    pub const1_cell: Option<String>,
    pub const1_out_pin: Option<String>,
    /// Index from (n_inputs, truth_table) to possible implementations.
    pub index: HashMap<(u8, Tt9), Vec<CellVariant>>,
    /// Lazy egg rewrite rules + gate symbol table; see [`TechLib::egg_rewrite_bundle`] in `emap_egg`.
    pub(crate) egg_bundle_cache: OnceLock<Arc<EggRewriteBundle>>,
    /// Lazy cover-only (`AIG -> gate`) egg rules; see [`TechLib::egg_cover_bundle`].
    pub(crate) egg_cover_bundle_cache: OnceLock<Arc<EggRewriteBundle>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TechLibCacheIndexEntry {
    n_inputs: u8,
    tt_hex: String,
    variants: Vec<CellVariant>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TechLibCacheSnapshot {
    version: u32,
    max_inputs: usize,
    genlib_size: u64,
    genlib_mtime_secs: u64,
    cells: Vec<Cell>,
    inv_area: f32,
    inv_delay: f32,
    inv_cell: Option<String>,
    inv_in_pin: Option<String>,
    inv_out_pin: Option<String>,
    inv_variant: Option<CellVariant>,
    and2_area: f32,
    and2_delay: [f32; 2],
    and2_variant: Option<CellVariant>,
    buf_area: f32,
    buf_delay: f32,
    buf_cell: Option<String>,
    buf_in_pin: Option<String>,
    buf_out_pin: Option<String>,
    const0_cell: Option<String>,
    const0_out_pin: Option<String>,
    const1_cell: Option<String>,
    const1_out_pin: Option<String>,
    index: Vec<TechLibCacheIndexEntry>,
}

const TECHLIB_CACHE_VERSION: u32 = 1;

/// Cell-name substrings excluded from matching via `EMAPPER_BAN_CELL` (diagnostic only).
fn banned_cells() -> &'static [String] {
    static V: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    V.get_or_init(|| {
        std::env::var("EMAPPER_BAN_CELL")
            .ok()
            .map(|s| {
                s.split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    })
}

impl TechLib {
    /// Parse a genlib file and build a TT index for cells with up to `max_inputs` variables.
    ///
    /// Truth tables use **512 bits** (`Tt9`); cells with more than **9** inputs are skipped.
    /// For each cell, the index stores NP (+ output inversion) variants, similar to mockturtle's
    /// `tech_library`.
    /// A copy of this library in which every timing number is 1.0 (see [`Self::unit_delays`]).

    /// Built by cloning the already-parsed cells and truth-table index — **no genlib re-parse and no
    /// index rebuild** — with fresh (empty) egg rewrite caches that fill lazily.
    #[must_use]
    pub fn unit_delay_variant(&self) -> Self {
        let mut lib = Self {
            cells: self.cells.clone(),
            inv_area: self.inv_area,
            inv_delay: self.inv_delay,
            inv_cell: self.inv_cell.clone(),
            inv_in_pin: self.inv_in_pin.clone(),
            inv_out_pin: self.inv_out_pin.clone(),
            inv_variant: self.inv_variant.clone(),
            and2_area: self.and2_area,
            and2_delay: self.and2_delay,
            and2_variant: self.and2_variant.clone(),
            buf_area: self.buf_area,
            buf_delay: self.buf_delay,
            buf_cell: self.buf_cell.clone(),
            buf_in_pin: self.buf_in_pin.clone(),
            buf_out_pin: self.buf_out_pin.clone(),
            const0_cell: self.const0_cell.clone(),
            const0_out_pin: self.const0_out_pin.clone(),
            const1_cell: self.const1_cell.clone(),
            const1_out_pin: self.const1_out_pin.clone(),
            index: self.index.clone(),
            egg_bundle_cache: OnceLock::new(),
            egg_cover_bundle_cache: OnceLock::new(),
        };
        lib.unit_delays();
        lib
    }

    /// Rewrite every timing number to **1.0** so that the mapper's "delay" becomes a *unit node
    /// depth* (one level per cell, one more for a materialized inverter), independent of arrival
    /// times. Used by the unit-depth + area cost mode.
    pub fn unit_delays(&mut self) {
        let unit = |v: &mut Vec<f32>| {
            for d in v.iter_mut() {
                *d = 1.0;
            }
        };
        for cell in self.cells.iter_mut() {
            for d in cell.pin_delay.values_mut() {
                *d = 1.0;
            }
        }
        for (_, variants) in self.index.iter_mut() {
            for v in variants.iter_mut() {
                unit(&mut v.pin_delay);
            }
        }
        self.inv_delay = 1.0;
        self.buf_delay = 1.0;
        self.and2_delay = [1.0, 1.0];
        if let Some(v) = self.inv_variant.as_mut() {
            unit(&mut v.pin_delay);
        }
        if let Some(v) = self.and2_variant.as_mut() {
            unit(&mut v.pin_delay);
        }
    }

    pub fn from_genlib<P: AsRef<Path>>(path: P, max_inputs: usize) -> Result<Self, GenlibError> {
        let contents = fs::read_to_string(path)?;
        let mut lib = Self {
            inv_area: f32::INFINITY,
            inv_delay: f32::INFINITY,
            and2_area: f32::INFINITY,
            and2_delay: [f32::INFINITY; 2],
            ..Self::default()
        };
        lib.cells = parse_genlib(&contents, max_inputs)?;
        lib.detect_inverter();
        lib.detect_and2();
        lib.detect_buffer();
        lib.detect_constants();
        lib.build_index();
        Ok(lib)
    }

    /// Load techlib from cache when possible; otherwise parse/build and refresh cache.
    ///
    /// Cache is automatically invalidated when the source genlib file size or mtime changes,
    /// or when `max_inputs` differs.
    pub fn from_genlib_cached<P: AsRef<Path>, Q: AsRef<Path>>(
        path: P,
        max_inputs: usize,
        cache_path: Q,
    ) -> Result<Self, GenlibError> {
        let genlib_path = path.as_ref();
        let cache_path = cache_path.as_ref();
        if let Some(lib) = Self::try_load_cache(genlib_path, max_inputs, cache_path) {
            return Ok(lib);
        }
        let lib = Self::from_genlib(genlib_path, max_inputs)?;
        lib.try_store_cache(genlib_path, max_inputs, cache_path);
        Ok(lib)
    }

    fn genlib_fingerprint(path: &Path) -> Option<(u64, u64)> {
        let md = fs::metadata(path).ok()?;
        let mtime = md
            .modified()
            .ok()?
            .duration_since(UNIX_EPOCH)
            .ok()?
            .as_secs();
        Some((md.len(), mtime))
    }

    fn try_load_cache(path: &Path, max_inputs: usize, cache_path: &Path) -> Option<Self> {
        let (genlib_size, genlib_mtime_secs) = Self::genlib_fingerprint(path)?;
        let text = fs::read_to_string(cache_path).ok()?;
        let snap: TechLibCacheSnapshot = serde_json::from_str(&text).ok()?;
        if snap.version != TECHLIB_CACHE_VERSION
            || snap.max_inputs != max_inputs
            || snap.genlib_size != genlib_size
            || snap.genlib_mtime_secs != genlib_mtime_secs
        {
            return None;
        }
        Self::from_snapshot(snap)
    }

    fn try_store_cache(&self, path: &Path, max_inputs: usize, cache_path: &Path) {
        let Some((genlib_size, genlib_mtime_secs)) = Self::genlib_fingerprint(path) else {
            return;
        };
        let snap = self.to_snapshot(max_inputs, genlib_size, genlib_mtime_secs);
        let Ok(text) = serde_json::to_string(&snap) else {
            return;
        };
        if let Some(parent) = cache_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(cache_path, text);
    }

    fn to_snapshot(
        &self,
        max_inputs: usize,
        genlib_size: u64,
        genlib_mtime_secs: u64,
    ) -> TechLibCacheSnapshot {
        let mut index = Vec::with_capacity(self.index.len());
        for ((n_inputs, tt), variants) in &self.index {
            index.push(TechLibCacheIndexEntry {
                n_inputs: *n_inputs,
                tt_hex: tt9_to_hex(tt),
                variants: variants.clone(),
            });
        }
        TechLibCacheSnapshot {
            version: TECHLIB_CACHE_VERSION,
            max_inputs,
            genlib_size,
            genlib_mtime_secs,
            cells: self.cells.clone(),
            inv_area: self.inv_area,
            inv_delay: self.inv_delay,
            inv_cell: self.inv_cell.clone(),
            inv_in_pin: self.inv_in_pin.clone(),
            inv_out_pin: self.inv_out_pin.clone(),
            inv_variant: self.inv_variant.clone(),
            and2_area: self.and2_area,
            and2_delay: self.and2_delay,
            and2_variant: self.and2_variant.clone(),
            buf_area: self.buf_area,
            buf_delay: self.buf_delay,
            buf_cell: self.buf_cell.clone(),
            buf_in_pin: self.buf_in_pin.clone(),
            buf_out_pin: self.buf_out_pin.clone(),
            const0_cell: self.const0_cell.clone(),
            const0_out_pin: self.const0_out_pin.clone(),
            const1_cell: self.const1_cell.clone(),
            const1_out_pin: self.const1_out_pin.clone(),
            index,
        }
    }

    fn from_snapshot(snap: TechLibCacheSnapshot) -> Option<Self> {
        let mut index = HashMap::with_capacity(snap.index.len());
        for e in snap.index {
            let tt = tt9_from_hex(&e.tt_hex)?;
            index.insert((e.n_inputs, tt), e.variants);
        }
        Some(Self {
            cells: snap.cells,
            inv_area: snap.inv_area,
            inv_delay: snap.inv_delay,
            inv_cell: snap.inv_cell,
            inv_in_pin: snap.inv_in_pin,
            inv_out_pin: snap.inv_out_pin,
            inv_variant: snap.inv_variant,
            and2_area: snap.and2_area,
            and2_delay: snap.and2_delay,
            and2_variant: snap.and2_variant,
            buf_area: snap.buf_area,
            buf_delay: snap.buf_delay,
            buf_cell: snap.buf_cell,
            buf_in_pin: snap.buf_in_pin,
            buf_out_pin: snap.buf_out_pin,
            const0_cell: snap.const0_cell,
            const0_out_pin: snap.const0_out_pin,
            const1_cell: snap.const1_cell,
            const1_out_pin: snap.const1_out_pin,
            index,
            egg_bundle_cache: OnceLock::new(),
            egg_cover_bundle_cache: OnceLock::new(),
        })
    }

    fn detect_inverter(&mut self) {
        // Pick smallest-area 1-input cell whose function is NOT (tt = 10b).
        // vars[0] assignment order: 0->0,1->1 so NOT is 0b01? Actually output bits [a=0, a=1]
        // NOT: [1,0] => 0b01? No: bit0 for a=0 => 1, bit1 for a=1 => 0 => 0b01.
        let mut best: Option<(&Cell, f32)> = None;
        for c in &self.cells {
            if c.vars.len() != 1 {
                continue;
            }
            if (c.tt[0] & 0x3) != 0b01 {
                continue;
            }
            let pin = c.vars[0].clone();
            let delay = c.pin_delay.get(&pin).copied().unwrap_or(0.0);
            match best {
                None => best = Some((c, delay)),
                Some((bc, _)) if c.area < bc.area => best = Some((c, delay)),
                _ => {}
            }
        }
        if let Some((c, d)) = best {
            self.inv_area = c.area;
            self.inv_delay = d;
            self.inv_cell = Some(c.name.clone());
            self.inv_in_pin = c.vars.first().cloned();
            self.inv_out_pin = Some(c.out_pin.clone());
            self.inv_variant = Some(CellVariant {
                cell_name: c.name.clone(),
                out_pin: c.out_pin.clone(),
                n_inputs: 1,
                perm: vec![0],
                neg_mask: 0,
                out_neg: false,
                area: c.area,
                pin_delay: vec![d],
                pin_names: c.vars.clone(),
            });
        }
    }

    fn detect_and2(&mut self) {
        let mut best: Option<(&Cell, [f32; 2])> = None;
        for c in &self.cells {
            if c.vars.len() != 2 || (c.tt[0] & 0xF) != 0b1000 {
                continue;
            }
            let d0 = c.pin_delay.get(&c.vars[0]).copied().unwrap_or(0.0);
            let d1 = c.pin_delay.get(&c.vars[1]).copied().unwrap_or(0.0);
            match best {
                None => best = Some((c, [d0, d1])),
                Some((bc, _)) if c.area < bc.area => best = Some((c, [d0, d1])),
                _ => {}
            }
        }
        if let Some((c, delays)) = best {
            self.and2_area = c.area;
            self.and2_delay = delays;
            self.and2_variant = Some(CellVariant {
                cell_name: c.name.clone(),
                out_pin: c.out_pin.clone(),
                n_inputs: 2,
                perm: vec![0, 1],
                neg_mask: 0,
                out_neg: false,
                area: c.area,
                pin_delay: delays.to_vec(),
                pin_names: c.vars.clone(),
            });
        }
    }

    fn detect_buffer(&mut self) {
        // Buffer: tt bits [a=0]=0, [a=1]=1 => 0b10.
        let mut best: Option<(&Cell, f32)> = None;
        for c in &self.cells {
            if c.vars.len() != 1 {
                continue;
            }
            if (c.tt[0] & 0x3) != 0b10 {
                continue;
            }
            let pin = c.vars[0].clone();
            let delay = c.pin_delay.get(&pin).copied().unwrap_or(0.0);
            match best {
                None => best = Some((c, delay)),
                Some((bc, _)) if c.area < bc.area => best = Some((c, delay)),
                _ => {}
            }
        }
        if let Some((c, d)) = best {
            self.buf_area = c.area;
            self.buf_delay = d;
            self.buf_cell = Some(c.name.clone());
            self.buf_in_pin = c.vars.first().cloned();
            self.buf_out_pin = Some(c.out_pin.clone());
        }
    }

    fn detect_constants(&mut self) {
        // CONST0: 0-input, bit0==0; CONST1: bit0==1
        for c in &self.cells {
            if !c.vars.is_empty() {
                continue;
            }
            if (c.tt[0] & 1) == 0 {
                self.const0_cell = Some(c.name.clone());
                self.const0_out_pin = Some(c.out_pin.clone());
            } else if (c.tt[0] & 1) == 1 {
                self.const1_cell = Some(c.name.clone());
                self.const1_out_pin = Some(c.out_pin.clone());
            }
        }
    }

    fn build_index(&mut self) {
        self.index.clear();
        for cell in &self.cells {
            let n = cell.vars.len();
            if n == 0 || n > 9 {
                continue;
            }

            let pin_delays = cell
                .vars
                .iter()
                .map(|v| cell.pin_delay.get(v).copied().unwrap_or(0.0))
                .collect::<Vec<_>>();
            let pin_names = cell.vars.clone();

            let perms = all_permutations(n as u8);
            for perm in perms {
                let mut permuted_pin_delays = vec![0.0; n];
                let mut permuted_pin_names = vec![String::new(); n];
                for (i, &src) in perm.iter().enumerate() {
                    permuted_pin_delays[i] = pin_delays[src as usize];
                    permuted_pin_names[i] = pin_names[src as usize].clone();
                }

                // For wide-input cells, skip input-phase enumeration to control index explosion.
                // Keep only the non-inverted input phase (neg_mask = 0) when n > 6.
                let neg_limit = if n <= 6 { 1u16 << n } else { 1 };
                for neg_mask in 0..neg_limit {
                    for out_neg in [false, true] {
                        let tt = permute_negate_tt9(&cell.tt, n as u8, &perm, neg_mask, out_neg);
                        let v = CellVariant {
                            cell_name: cell.name.clone(),
                            out_pin: cell.out_pin.clone(),
                            n_inputs: n as u8,
                            perm: perm.clone(),
                            neg_mask,
                            out_neg,
                            area: cell.area,
                            pin_delay: permuted_pin_delays.clone(),
                            pin_names: permuted_pin_names.clone(),
                        };
                        self.index.entry((n as u8, tt)).or_default().push(v);
                    }
                }
            }
        }
    }

    #[must_use]
    pub fn candidates(&self, n_inputs: u8, tt: Tt9) -> Option<&[CellVariant]> {
        self.index.get(&(n_inputs, tt)).map(Vec::as_slice)
    }

    #[must_use]
    pub fn exact_inv_variant(&self) -> Option<&CellVariant> {
        self.inv_variant.as_ref()
    }

    #[must_use]
    pub fn exact_and2_variant(&self) -> Option<&CellVariant> {
        self.and2_variant.as_ref()
    }

    /// `cell.area` plus physical input inverters and optional output inversion (`out_neg`).
    /// `pin_count` is usually `v.n_inputs` / `arrivals.len()`; when `native_phases` lengths differ,
    /// falls back to counting `neg_mask` bits (legacy proxy).
    pub(crate) fn variant_area_physical_invs(
        &self,
        v: &CellVariant,
        pin_count: usize,
        native_phases: Option<(&[bool], &[bool])>,
    ) -> f32 {
        let inv_area = self.inv_area;
        let input_phys_inv = if let Some((np, nn)) = native_phases {
            if np.len() != pin_count || nn.len() != pin_count {
                (0..pin_count)
                    .filter(|&i| ((v.neg_mask >> i) & 1) == 1)
                    .count() as f32
            } else {
                let mut c = 0.0f32;
                for i in 0..pin_count {
                    let need_neg = ((v.neg_mask >> i) & 1) == 1;
                    let needs_inv = if need_neg { !nn[i] } else { !np[i] };
                    if needs_inv {
                        c += 1.0;
                    }
                }
                c
            }
        } else {
            (0..pin_count)
                .filter(|&i| ((v.neg_mask >> i) & 1) == 1)
                .count() as f32
        };
        v.area + input_phys_inv * inv_area + if v.out_neg { inv_area } else { 0.0 }
    }

    /// Returns `(mapping_area, physical_area)` for variant selection vs reporting.
    pub(crate) fn mapping_and_physical_area(
        &self,
        v: &CellVariant,
        pin_count: usize,
        native_phases: Option<(&[bool], &[bool])>,
        leaf_flow: Option<(&[f32], &[f32])>,
        area_cost: MappingAreaCost,
    ) -> (f32, f32) {
        let physical = self.variant_area_physical_invs(v, pin_count, native_phases);
        let mapping = match area_cost {
            MappingAreaCost::Physical => physical,
            MappingAreaCost::Flow => {
                if let Some((fp, fn_)) = leaf_flow {
                    if fp.len() == pin_count && fn_.len() == pin_count {
                        let mut s = v.area;
                        for i in 0..pin_count {
                            s += if ((v.neg_mask >> i) & 1) == 1 {
                                fn_[i]
                            } else {
                                fp[i]
                            };
                        }
                        s
                    } else {
                        physical
                    }
                } else {
                    physical
                }
            }
        };
        (mapping, physical)
    }

    fn best_variant_from_candidates<'a>(
        &self,
        cands: impl IntoIterator<Item = &'a CellVariant>,
        target_tt: Tt9,
        arrivals: &[PhaseArrival],
        objective: MappingObjective,
        // When `Some`, area counts only physical input inverters: each pin needs an extra
        // inverter iff the logical phase required by `neg_mask` is not already available on the
        // net (see `emap` native pos/neg flags). When `None`, use the legacy proxy
        // `inv_area * (#neg_mask bits)` for inputs.
        native_phases: Option<(&[bool], &[bool])>,
        leaf_flow: Option<(&[f32], &[f32])>,
        area_cost: MappingAreaCost,
    ) -> Option<(f32, f32, f32, &'a CellVariant)> {
        /* `EMAPPER_BAN_CELL` (comma-separated substrings): diagnostic knob that removes cells from
        matching. Used to measure how much of the delay gap is a *cell choice* rather than a
        structural one -- e.g. banning `XOR`/`XNOR` shows whether the mapper has a cheaper
        multi-level alternative for those cones or is forced onto the slowest cells in the
        library (XOR2x2/XNOR2x2 carry the highest pin delays in asap7: 36.3/36.9). */
        let banned = banned_cells();
        let mut best: Option<(f32, f32, f32, &'a CellVariant)> = None;
        for v in cands {
            if banned.iter().any(|b| v.cell_name.contains(b.as_str())) {
                continue;
            }
            // Semantic guard: ensure this variant really realizes `target_tt` under
            // the same pin/negation wiring semantics used by netlist emission.
            if self.variant_tt_wiring(v).map_or(true, |t| t != target_tt) {
                continue;
            }
            let mut delay = 0.0f32;
            for (i, arrival) in arrivals.iter().enumerate() {
                let phase_arrival = if ((v.neg_mask >> i) & 1) == 1 {
                    arrival.neg
                } else {
                    arrival.pos
                };
                let d = phase_arrival + v.pin_delay[i];
                delay = delay.max(d);
            }
            if v.out_neg {
                delay += self.inv_delay;
            }

            let (mapping_area, physical_area) = self.mapping_and_physical_area(
                v,
                arrivals.len(),
                native_phases,
                leaf_flow,
                area_cost,
            );

            /* `physical_area` (cell + charged input/output inverters) is the final tie-break for
            **both** objectives. `mapping_area` is the `Flow` proxy by default, which adds the
            leaves' flow and never an `inv_area` -- so with only `(delay, mapping_area)` as the
            key, two variants of the same truth table that differ solely in how many inverters
            they need compare equal and the first one wins. Measured on c432/c5315 that left
            ~half of the total inversion overhead on `AND2x2` instances that each carry exactly
            one input inverter, while `emap` on the same AIG instantiates no `AND2` at all. */
            let cand_key = match objective {
                MappingObjective::DelayFirst => (delay, mapping_area, physical_area),
                MappingObjective::AreaFirst => (mapping_area, delay, physical_area),
            };
            match best {
                None => best = Some((delay, mapping_area, physical_area, v)),
                Some((bd, ba_map, bp, _)) => {
                    let best_key = match objective {
                        MappingObjective::DelayFirst => (bd, ba_map, bp),
                        MappingObjective::AreaFirst => (ba_map, bd, bp),
                    };
                    if cand_key < best_key {
                        best = Some((delay, mapping_area, physical_area, v));
                    }
                }
            }
        }
        best
    }

    /// Pick the best matching cell variant for a function, using pin delays and area.
    ///
    /// - **Delay model**: \( \max_i(arrival[i] + pinDelay[i] + invDelay_if_input_negated) \) plus
    ///   `inv_delay` if output is negated.
    /// - **Area model**: `cell.area + inv_area * (physical_input_inverters + out_neg)`.
    ///   With `native_phases` in [`Self::best_for_tt_phase_arrivals`], physical input inverters
    ///   are counted only when the required phase is not already available on the net.
    ///
    /// Returns `(best_delay, best_area, best_variant)`.
    pub fn best_for_tt(
        &self,
        n_inputs: u8,
        tt: Tt9,
        arrivals: &[f32],
        objective: MappingObjective,
    ) -> Option<(f32, f32, &CellVariant)> {
        let phase_arrivals = arrivals
            .iter()
            .map(|&a| PhaseArrival {
                pos: a,
                neg: a + self.inv_delay,
            })
            .collect::<Vec<_>>();
        self.best_for_tt_phase_arrivals(
            n_inputs,
            tt,
            &phase_arrivals,
            objective,
            None,
            None,
            MappingAreaCost::Physical,
        )
        .map(|(d, _, p, v)| (d, p, v))
    }

    pub fn best_for_tt_phase_arrivals(
        &self,
        n_inputs: u8,
        tt: Tt9,
        arrivals: &[PhaseArrival],
        objective: MappingObjective,
        native_phases: Option<(&[bool], &[bool])>,
        leaf_flow: Option<(&[f32], &[f32])>,
        area_cost: MappingAreaCost,
    ) -> Option<(f32, f32, f32, &CellVariant)> {
        let cands = self.candidates(n_inputs, tt)?;
        if arrivals.len() != n_inputs as usize {
            return None;
        }
        self.best_variant_from_candidates(
            cands.iter(),
            tt,
            arrivals,
            objective,
            native_phases,
            leaf_flow,
            area_cost,
        )
    }

    /// Pick the best variant for a specific base standard cell.
    #[must_use]
    pub fn best_for_cell(
        &self,
        cell: &Cell,
        arrivals: &[f32],
        objective: MappingObjective,
    ) -> Option<(f32, f32, &CellVariant)> {
        let phase_arrivals = arrivals
            .iter()
            .map(|&a| PhaseArrival {
                pos: a,
                neg: a + self.inv_delay,
            })
            .collect::<Vec<_>>();
        self.best_for_cell_phase_arrivals(
            cell,
            &phase_arrivals,
            objective,
            None,
            None,
            MappingAreaCost::Physical,
        )
        .map(|(d, _, p, v)| (d, p, v))
    }

    #[must_use]
    pub fn best_for_cell_phase_arrivals(
        &self,
        cell: &Cell,
        arrivals: &[PhaseArrival],
        objective: MappingObjective,
        native_phases: Option<(&[bool], &[bool])>,
        leaf_flow: Option<(&[f32], &[f32])>,
        area_cost: MappingAreaCost,
    ) -> Option<(f32, f32, f32, &CellVariant)> {
        let n_inputs = cell.vars.len();
        if arrivals.len() != n_inputs {
            return None;
        }
        let cands = self
            .candidates(n_inputs as u8, cell.tt)?
            .iter()
            .filter(|v| v.cell_name == cell.name);
        self.best_variant_from_candidates(
            cands,
            cell.tt,
            arrivals,
            objective,
            native_phases,
            leaf_flow,
            area_cost,
        )
    }

    pub(crate) fn variant_tt_wiring(&self, v: &CellVariant) -> Option<Tt9> {
        let n = v.n_inputs as usize;
        if n == 0 || n > 9 {
            return None;
        }
        let cell = self.cells.iter().find(|c| c.name == v.cell_name)?;
        let mut var_pos = BTreeMap::<String, usize>::new();
        for (i, name) in cell.vars.iter().enumerate() {
            var_pos.insert(name.clone(), i);
        }
        let mut out = tt9_zero();
        for a in 0..(1usize << n) {
            // Build assignment in original cell var space.
            let mut old_bits = 0usize;
            for i in 0..n {
                let mut bit = ((a >> i) & 1) == 1;
                if ((v.neg_mask >> i) & 1) == 1 {
                    bit = !bit;
                }
                let pin = v.pin_names.get(i)?;
                let &old_idx = var_pos.get(pin)?;
                if bit {
                    old_bits |= 1usize << old_idx;
                }
            }
            let mut y = tt9_bit_get(&cell.tt, old_bits);
            if v.out_neg {
                y = !y;
            }
            if y {
                tt9_bit_set(&mut out, a);
            }
        }
        Some(out)
    }
}

fn parse_genlib(contents: &str, max_inputs: usize) -> Result<Vec<Cell>, GenlibError> {
    let mut cells = Vec::new();
    let mut cur: Option<(String, f32, Expr, BTreeMap<String, f32>)> = None;

    for (line_idx, raw) in contents.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if let Some(rest) = line.strip_prefix("GATE") {
            if let Some((name, area, out_pin, expr)) =
                parse_gate_line(rest).map_err(|msg| GenlibError::Parse { line: line_no, msg })?
            {
                if let Some((n, a, e, p)) = cur.take() {
                    if let Some(cell) = finalize_cell(n, a, e, p, max_inputs)? {
                        cells.push(cell);
                    }
                }
                // Store out_pin inside expr by wrapping it later in finalize_cell; keep it in name temporarily.
                // We encode `out_pin` into the cell name tuple by prefixing with "name\0outpin".
                cur = Some((format!("{name}\0{out_pin}"), area, expr, BTreeMap::new()));
            }
            continue;
        }

        if let Some(rest) = line.strip_prefix("PIN") {
            let (_, _, _, ref mut pin_delays) = cur.as_mut().ok_or_else(|| GenlibError::Parse {
                line: line_no,
                msg: "PIN line before any GATE".to_string(),
            })?;
            let (pin, delay) =
                parse_pin_line(rest).map_err(|msg| GenlibError::Parse { line: line_no, msg })?;
            pin_delays.insert(pin, delay);
            continue;
        }
    }

    if let Some((n, a, e, p)) = cur.take() {
        if let Some(cell) = finalize_cell(n, a, e, p, max_inputs)? {
            cells.push(cell);
        }
    }

    Ok(cells)
}

fn finalize_cell(
    name: String,
    area: f32,
    expr: Expr,
    pin_delay: BTreeMap<String, f32>,
    max_inputs: usize,
) -> Result<Option<Cell>, GenlibError> {
    let (name, out_pin) = if let Some((n, p)) = name.split_once('\0') {
        (n.to_string(), p.to_string())
    } else {
        (name, "Y".to_string())
    };
    let mut vars = BTreeSet::new();
    expr.vars(&mut vars);

    // Remove known constant identifiers (genlib uses CONST0/CONST1).
    vars.remove("CONST0");
    vars.remove("CONST1");

    let vars = vars.into_iter().collect::<Vec<_>>();
    if vars.len() > max_inputs {
        // For MVP we silently skip cells wider than `max_inputs`.
        return Ok(None);
    }
    if vars.len() > 9 {
        return Ok(None);
    }
    let tt = expr_to_tt9(&expr, &vars);
    Ok(Some(Cell {
        name,
        area,
        out_pin,
        expr,
        pin_delay,
        vars,
        tt,
    }))
}

fn parse_gate_line(rest: &str) -> Result<Option<(String, f32, String, Expr)>, String> {
    // Example:
    // " INVx1_ASAP7_75t_R        0.04  Y=!A;"
    let mut it = rest.split_whitespace();
    let name = it.next().ok_or("missing gate name")?.to_string();
    let area_str = it.next().ok_or("missing gate area")?;
    let area: f32 = area_str.parse().map_err(|_| "invalid gate area")?;

    let expr_start = rest.find('=').ok_or("missing '=' in gate function")?;
    let lhs = rest[..expr_start].trim();
    let out_pin = lhs
        .split_whitespace()
        .last()
        .ok_or("missing output pin")?
        .to_string();
    let mut expr = rest[expr_start + 1..].trim().to_string();
    if let Some(semi) = expr.rfind(';') {
        expr.truncate(semi);
    }
    expr = expr.trim().to_string();
    if expr.is_empty() {
        return Err("empty gate function expression".to_string());
    }

    // Support CONST0/CONST1 shortcuts (e.g., z=CONST0)
    let parsed = parse_bool_expr(&expr)?;
    Ok(Some((name, area, out_pin, parsed)))
}

fn parse_pin_line(rest: &str) -> Result<(String, f32), String> {
    // Example:
    // "  A  UNKNOWN   1 999    14.77     0.00    14.77     0.00"
    let parts = rest.split_whitespace().collect::<Vec<_>>();
    // Note: `rest` does not include the leading "PIN" token.
    // Expected tokens: <pin> <phase> <load> <max_load> <rise_block> <rise_fanout> <fall_block> <fall_fanout>
    if parts.len() < 8 {
        return Err("PIN line too short".to_string());
    }
    let pin = parts[0].to_string();
    // Take rise_block (parts[4]) and fall_block (parts[6])
    let rise_block: f32 = parts[4]
        .parse()
        .map_err(|_| "invalid PIN rise block delay")?;
    let fall_block: f32 = parts[6]
        .parse()
        .map_err(|_| "invalid PIN fall block delay")?;
    Ok((pin, rise_block.max(fall_block)))
}

fn expr_to_tt9(expr: &Expr, vars: &[String]) -> Tt9 {
    let n = vars.len();
    debug_assert!(n <= 9);
    let mut env: HashMap<&str, bool> = HashMap::new();
    let mut tt = tt9_zero();
    for a in 0..(1usize << n) {
        env.clear();
        for (i, v) in vars.iter().enumerate() {
            env.insert(v.as_str(), ((a >> i) & 1) == 1);
        }
        env.insert("CONST0", false);
        env.insert("CONST1", true);
        if expr.eval(&env) {
            tt9_bit_set(&mut tt, a);
        }
    }
    tt
}

/// Permute and negate a truth table with `n` variables (for tests; `n <= 6` fits in `u64`).
#[cfg(test)]
fn permute_negate_tt64(tt: u64, n: u8, perm: &[u8], neg_mask: u16, out_neg: bool) -> u64 {
    let t = permute_negate_tt9(&crate::tt9::tt9_from_u64(tt), n, perm, neg_mask, out_neg);
    let n = n as usize;
    if n >= 6 {
        t[0]
    } else {
        let bits = 1u32 << n;
        if bits >= 64 {
            t[0]
        } else {
            t[0] & ((1u64 << bits) - 1)
        }
    }
}

fn all_permutations(n: u8) -> Vec<Vec<u8>> {
    fn rec(cur: &mut Vec<u8>, used: &mut Vec<bool>, n: u8, out: &mut Vec<Vec<u8>>) {
        if cur.len() == n as usize {
            out.push(cur.clone());
            return;
        }
        for i in 0..n {
            if used[i as usize] {
                continue;
            }
            used[i as usize] = true;
            cur.push(i);
            rec(cur, used, n, out);
            cur.pop();
            used[i as usize] = false;
        }
    }
    let mut out = Vec::new();
    let mut cur = Vec::new();
    let mut used = vec![false; n as usize];
    rec(&mut cur, &mut used, n, &mut out);
    out
}

// ---------------- Bool expr parser ----------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tok<'a> {
    Not,
    And,
    Or,
    LParen,
    RParen,
    Ident(&'a str),
}

fn tokenize(s: &str) -> Result<Vec<Tok<'_>>, String> {
    let mut toks = Vec::new();
    let mut i = 0;
    let b = s.as_bytes();
    while i < b.len() {
        let c = b[i] as char;
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        match c {
            '!' => {
                toks.push(Tok::Not);
                i += 1;
            }
            '*' => {
                toks.push(Tok::And);
                i += 1;
            }
            '+' => {
                toks.push(Tok::Or);
                i += 1;
            }
            '(' => {
                toks.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                toks.push(Tok::RParen);
                i += 1;
            }
            _ => {
                if c.is_ascii_alphanumeric() || c == '_' {
                    let start = i;
                    i += 1;
                    while i < b.len() {
                        let c2 = b[i] as char;
                        if c2.is_ascii_alphanumeric() || c2 == '_' {
                            i += 1;
                        } else {
                            break;
                        }
                    }
                    toks.push(Tok::Ident(&s[start..i]));
                } else {
                    return Err(format!("unexpected char '{c}'"));
                }
            }
        }
    }
    Ok(toks)
}

fn parse_bool_expr(s: &str) -> Result<Expr, String> {
    let toks = tokenize(s)?;
    let mut p = Parser { toks: &toks, i: 0 };
    let e = p.parse_or()?;
    if p.i != toks.len() {
        return Err("trailing tokens".to_string());
    }
    Ok(e)
}

struct Parser<'a> {
    toks: &'a [Tok<'a>],
    i: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<Tok<'a>> {
        self.toks.get(self.i).copied()
    }

    fn bump(&mut self) -> Option<Tok<'a>> {
        let t = self.peek()?;
        self.i += 1;
        Some(t)
    }

    fn parse_or(&mut self) -> Result<Expr, String> {
        let mut e = self.parse_and()?;
        while self.peek() == Some(Tok::Or) {
            self.bump();
            let r = self.parse_and()?;
            e = Expr::Or(Box::new(e), Box::new(r));
        }
        Ok(e)
    }

    fn parse_and(&mut self) -> Result<Expr, String> {
        let mut e = self.parse_unary()?;
        while self.peek() == Some(Tok::And) {
            self.bump();
            let r = self.parse_unary()?;
            e = Expr::And(Box::new(e), Box::new(r));
        }
        Ok(e)
    }

    fn parse_unary(&mut self) -> Result<Expr, String> {
        if self.peek() == Some(Tok::Not) {
            self.bump();
            let a = self.parse_unary()?;
            return Ok(Expr::Not(Box::new(a)));
        }
        self.parse_atom()
    }

    fn parse_atom(&mut self) -> Result<Expr, String> {
        match self.bump().ok_or("unexpected end of input")? {
            Tok::LParen => {
                let e = self.parse_or()?;
                match self.bump() {
                    Some(Tok::RParen) => Ok(e),
                    _ => Err("missing ')'".to_string()),
                }
            }
            Tok::Ident(id) => match id {
                "0" | "CONST0" => Ok(Expr::Const(false)),
                "1" | "CONST1" => Ok(Expr::Const(true)),
                _ => Ok(Expr::Var(id.to_string())),
            },
            t => Err(format!("unexpected token {t:?}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_eval_tt() {
        let e = parse_bool_expr("(A * B) + (!A * !B)").unwrap();
        let vars = vec!["A".to_string(), "B".to_string()];
        let tt = expr_to_tt9(&e, &vars);
        // XNOR: 00->1, 01->0, 10->0, 11->1 => bits 0..3 => 1001b
        assert_eq!(tt[0] & 0xF, 0b1001);
    }

    #[test]
    fn perm_neg_outneg() {
        // AND(a,b): 00 01 10 11 => 0001
        let tt_and = 0b0001u64;
        // perm swap, negate input0 => (~b) & a
        let tt2 = permute_negate_tt64(tt_and, 2, &[1, 0], 0b01, false) & 0xF;
        // With our bit/perm convention, this corresponds to (x0 & ~x1):
        // 00->0, 01->1, 10->0, 11->0 => 0010
        assert_eq!(tt2, 0b0010);
        // output neg => NAND
        let tt3 = permute_negate_tt64(tt_and, 2, &[0, 1], 0, true) & 0xF;
        assert_eq!(tt3, 0b1110);
    }
}
