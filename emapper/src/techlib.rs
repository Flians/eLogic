use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs,
    path::Path,
    sync::{Arc, OnceLock},
};

use thiserror::Error;

use crate::emap_egg::EggRewriteBundle;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum MappingObjective {
    AreaFirst,
    #[default]
    DelayFirst,
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
    TooManyInputs { cell: String, inputs: usize, max: usize },
}

#[derive(Clone, Debug, PartialEq)]
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

#[derive(Clone, Debug)]
pub struct Cell {
    pub name: String,
    pub area: f32,
    /// Output pin name (e.g. "Y" or "z").
    pub out_pin: String,
    pub expr: Expr,
    /// Pin block delay (max of rise/fall block).
    pub pin_delay: BTreeMap<String, f32>,
    /// Ordered variable names used by `tt64` (sorted).
    pub vars: Vec<String>,
    /// Truth table for `expr` with `vars` order (LSB is assignment 0).
    pub tt64: u64,
}

#[derive(Clone, Debug)]
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
    pub index: HashMap<(u8, u64), Vec<CellVariant>>,
    /// Lazy egg rewrite rules + gate symbol table; see [`TechLib::egg_rewrite_bundle`] in `emap_egg`.
    pub(crate) egg_bundle_cache: OnceLock<Arc<EggRewriteBundle>>,
}

impl TechLib {
    /// Parse a genlib file and build a TT index for cells with up to `max_inputs`.
    ///
    /// For each cell, the index stores NP (+ output inversion) variants, similar in spirit to
    /// mockturtle's `tech_library`.
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

    fn detect_inverter(&mut self) {
        // Pick smallest-area 1-input cell whose function is NOT (tt = 10b).
        // vars[0] assignment order: 0->0,1->1 so NOT is 0b01? Actually output bits [a=0, a=1]
        // NOT: [1,0] => 0b01? No: bit0 for a=0 => 1, bit1 for a=1 => 0 => 0b01.
        let mut best: Option<(&Cell, f32)> = None;
        for c in &self.cells {
            if c.vars.len() != 1 {
                continue;
            }
            if c.tt64 != 0b01 {
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
            if c.vars.len() != 2 || c.tt64 != 0b1000 {
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
            if c.tt64 != 0b10 {
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
        // CONST0: 0-input, tt64==0
        // CONST1: 0-input, tt64==1 (single bit)
        for c in &self.cells {
            if !c.vars.is_empty() {
                continue;
            }
            if c.tt64 == 0 {
                self.const0_cell = Some(c.name.clone());
                self.const0_out_pin = Some(c.out_pin.clone());
            } else if c.tt64 == 1 {
                self.const1_cell = Some(c.name.clone());
                self.const1_out_pin = Some(c.out_pin.clone());
            }
        }
    }

    fn build_index(&mut self) {
        self.index.clear();
        for cell in &self.cells {
            let n = cell.vars.len();
            if n == 0 || n > 15 {
                continue;
            }
            if n > 6 {
                // Egg-side TT analysis will be limited to 6 inputs for now.
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

                let neg_limit = 1u16 << n;
                for neg_mask in 0..neg_limit {
                    for out_neg in [false, true] {
                        let tt = permute_negate_tt64(cell.tt64, n as u8, &perm, neg_mask, out_neg);
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
    pub fn candidates(&self, n_inputs: u8, tt64: u64) -> Option<&[CellVariant]> {
        self.index.get(&(n_inputs, tt64)).map(Vec::as_slice)
    }

    #[must_use]
    pub fn exact_inv_variant(&self) -> Option<&CellVariant> {
        self.inv_variant.as_ref()
    }

    #[must_use]
    pub fn exact_and2_variant(&self) -> Option<&CellVariant> {
        self.and2_variant.as_ref()
    }

    fn best_variant_from_candidates<'a>(
        &self,
        cands: impl IntoIterator<Item = &'a CellVariant>,
        tt64: u64,
        arrivals: &[PhaseArrival],
        objective: MappingObjective,
    ) -> Option<(f32, f32, &'a CellVariant)> {
        let inv_area = self.inv_area;

        let mut best: Option<(f32, f32, &'a CellVariant)> = None;
        for v in cands {
            // Semantic guard: ensure this variant really realizes `tt64` under
            // the same pin/negation wiring semantics used by netlist emission.
            if self.variant_tt_wiring(v).map_or(true, |t| t != tt64) {
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

            let invs = (0..arrivals.len())
                .filter(|&i| ((v.neg_mask >> i) & 1) == 1)
                .count() as f32
                + if v.out_neg { 1.0 } else { 0.0 };
            let area = v.area + invs * inv_area;

            match best {
                None => best = Some((delay, area, v)),
                Some((bd, ba, _)) => {
                    let better = match objective {
                        MappingObjective::DelayFirst => (delay, area) < (bd, ba),
                        MappingObjective::AreaFirst => (area, delay) < (ba, bd),
                    };
                    if better {
                        best = Some((delay, area, v));
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
    /// - **Area model**: `cell.area + inv_area * (#neg_inputs + out_neg)`.
    ///
    /// Returns `(best_delay, best_area, best_variant)`.
    pub fn best_for_tt(
        &self,
        n_inputs: u8,
        tt64: u64,
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
        self.best_for_tt_phase_arrivals(n_inputs, tt64, &phase_arrivals, objective)
    }

    pub fn best_for_tt_phase_arrivals(
        &self,
        n_inputs: u8,
        tt64: u64,
        arrivals: &[PhaseArrival],
        objective: MappingObjective,
    ) -> Option<(f32, f32, &CellVariant)> {
        let cands = self.candidates(n_inputs, tt64)?;
        if arrivals.len() != n_inputs as usize {
            return None;
        }
        self.best_variant_from_candidates(cands.iter(), tt64, arrivals, objective)
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
        self.best_for_cell_phase_arrivals(cell, &phase_arrivals, objective)
    }

    #[must_use]
    pub fn best_for_cell_phase_arrivals(
        &self,
        cell: &Cell,
        arrivals: &[PhaseArrival],
        objective: MappingObjective,
    ) -> Option<(f32, f32, &CellVariant)> {
        let n_inputs = cell.vars.len();
        if arrivals.len() != n_inputs {
            return None;
        }
        let cands = self
            .candidates(n_inputs as u8, cell.tt64)?
            .iter()
            .filter(|v| v.cell_name == cell.name);
        self.best_variant_from_candidates(cands, cell.tt64, arrivals, objective)
    }

    fn variant_tt_wiring(&self, v: &CellVariant) -> Option<u64> {
        let n = v.n_inputs as usize;
        if n == 0 || n > 6 {
            return None;
        }
        let cell = self.cells.iter().find(|c| c.name == v.cell_name)?;
        let mut var_pos = BTreeMap::<String, usize>::new();
        for (i, name) in cell.vars.iter().enumerate() {
            var_pos.insert(name.clone(), i);
        }
        let mut out = 0u64;
        for a in 0..(1u64 << n) {
            // Build assignment in original cell var space.
            let mut old_bits = 0u64;
            for i in 0..n {
                let mut bit = ((a >> i) & 1) == 1;
                if ((v.neg_mask >> i) & 1) == 1 {
                    bit = !bit;
                }
                let pin = v.pin_names.get(i)?;
                let &old_idx = var_pos.get(pin)?;
                if bit {
                    old_bits |= 1u64 << (old_idx as u64);
                }
            }
            let mut y = ((cell.tt64 >> old_bits) & 1) == 1;
            if v.out_neg {
                y = !y;
            }
            if y {
                out |= 1u64 << a;
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
            if let Some((name, area, out_pin, expr)) = parse_gate_line(rest).map_err(|msg| GenlibError::Parse {
                line: line_no,
                msg,
            })? {
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
            let (pin, delay) = parse_pin_line(rest).map_err(|msg| GenlibError::Parse {
                line: line_no,
                msg,
            })?;
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
    let tt64 = expr_to_tt64(&expr, &vars);
    Ok(Some(Cell {
        name,
        area,
        out_pin,
        expr,
        pin_delay,
        vars,
        tt64,
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
    let rise_block: f32 = parts[4].parse().map_err(|_| "invalid PIN rise block delay")?;
    let fall_block: f32 = parts[6].parse().map_err(|_| "invalid PIN fall block delay")?;
    Ok((pin, rise_block.max(fall_block)))
}

fn expr_to_tt64(expr: &Expr, vars: &[String]) -> u64 {
    let n = vars.len();
    let mut env: HashMap<&str, bool> = HashMap::new();
    let mut tt = 0u64;
    for a in 0..(1u64 << n) {
        env.clear();
        for (i, v) in vars.iter().enumerate() {
            env.insert(v.as_str(), ((a >> i) & 1) == 1);
        }
        // Constants are looked up by name; set them explicitly.
        env.insert("CONST0", false);
        env.insert("CONST1", true);
        if expr.eval(&env) {
            tt |= 1u64 << a;
        }
    }
    tt
}

/// Permute and negate a truth table with `n` variables.
///
/// `perm[i] = src` means new variable i maps to old variable `src`.
/// `neg_mask` inverts new variable i when bit i is 1.
/// `out_neg` inverts the output.
fn permute_negate_tt64(tt: u64, n: u8, perm: &[u8], neg_mask: u16, out_neg: bool) -> u64 {
    let n = n as usize;
    let mut out = 0u64;
    for a in 0..(1u64 << n) {
        let mut old = 0u64;
        for i in 0..n {
            let mut bit = ((a >> i) & 1) == 1;
            if ((neg_mask >> i) & 1) == 1 {
                bit = !bit;
            }
            if bit {
                old |= 1u64 << (perm[i] as u64);
            }
        }
        let mut v = ((tt >> old) & 1) == 1;
        if out_neg {
            v = !v;
        }
        if v {
            out |= 1u64 << a;
        }
    }
    out
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
        let tt = expr_to_tt64(&e, &vars);
        // XNOR: 00->1, 01->0, 10->0, 11->1 => bits 0..3 => 1001b
        assert_eq!(tt & 0xF, 0b1001);
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

