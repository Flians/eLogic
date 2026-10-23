//! Technology mapping directly on a saturated e-graph.
//!
//! Precise equalities stay in egg. SDC is a reachable-image check on a fixed boundary and only
//! admits external physical matches; it never unions conditionally equal expressions. CBC selects
//! instances and output bindings separately, paying once for a cell with any number of live pins.
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::Write;
use std::time::Instant;

use coin_cbc::{Col, Model, Sense};
use egg::Language;
use egg::{Id, Runner};

use crate::aig_graph::Aig;
use crate::egraph_sat::{
    cheapest_nodes, cheapest_nodes_queued, import_aig, literal_inputs, xor_inputs, tt3_inputs, rules, AigLang, SatParams,
};
use crate::techlib::{Cell, Expr, TechLib};
use crate::tt9::tt9_from_u64;
#[path = "egraph_regions.rs"]
mod regions;
#[path = "egraph_source_import.rs"]
mod source_import;
#[path="egraph_cover_exchange.rs"]
mod exchange;
pub use exchange::BindingSource;

/// Whole-cover objective. Lexicographic modes never trade the primary metric
/// for the secondary metric; Weighted retains the legacy scalar objective.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MapObjective {
    #[default]
    Weighted,
    AreaDelay,
    DelayArea,
}

impl MapObjective {
    fn key(self, value: (f64, f64), weight: f64) -> (f64, f64) {
        match self {
            Self::AreaDelay => value,
            Self::DelayArea => (value.1, value.0),
            Self::Weighted => (
                if weight == 0.0 {
                    value.0
                } else {
                    value.0 + weight * value.1
                },
                value.1,
            ),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MapParams {
    pub saturation: SatParams,
    pub max_inputs: usize,
    pub cuts_per_class: usize,
    pub cut_rounds: usize,
    /// Bound on optional physical candidates. Exact fallback candidates may exceed this budget.
    pub max_instances: usize,
    pub multioutput: bool,
    /// Also accumulate bounded boundary partners for three or more independent outputs.
    pub multioutput_cumulative: bool,
    /// Zero disables SDC. Exact PI-support enumeration is bounded by this many variables.
    pub dc_support: usize,
    pub dc_cone: usize,
    /// Ablation only: optional cuts over one finite witness, with identical source fallbacks.
    pub single_structure_cuts: bool,
    /// Ablation only: disable complemented physical input matching; INV remains.
    pub no_input_phase_matches: bool,
    /// Ablation only: use sorted physical pin order, retaining real pin delays.
    pub canonical_pin_matching: bool,
    /// Ablation only: omit the final actual-area substitution pass.
    pub no_actual_area_tail: bool,
    /// Same-binary ablation: omit functional-quotient construction, not structural cuts.
    pub ablate_functional_cuts: bool,
    /// Same-binary ablation: omit post-phase required-time area flow and actual-area recovery.
    pub ablate_shared_area: bool,
    /// Same-binary ablation: remove the dual-metric seed guard, retaining legality proofs.
    pub ablate_incumbent_guard: bool,
    pub time_limit_secs: f64,
    pub delay_weight: f64,
    pub objective: MapObjective,
    pub max_delay: Option<f64>,
    pub refine_rounds: usize,
    pub refine_trials: usize,
    /// 0: original equalities, 1: OR factoring, 2: also associative alternatives (larger graphs).
    pub boolean_rules: u8,
    /// Explore faster equal-area flow choices; can change DAG sharing, so opt-in.
    pub flow_delay_ties: bool,
    /// Additional area-flow seed using the finite witness's physical fanouts.
    pub structural_refs: bool,
    /// Event-driven flow relaxation under the same recovery deadline; opt-in.
    pub flow_worklist: bool,
    /// Ready physical-frontier delay witness; opt-in.
    pub fast_delay_seed: bool,
    /// Fixed-required-time DAG area-flow alternative; opt-in.
    pub required_area_dag: bool,
    /// Keep the validated area-flow alternative separate until final selection.
    pub defer_area_dag: bool,
    /// Preserve original cold seeds/ties and a separate legacy working incumbent.
    pub preserve_legacy_seeds: bool,
    /// Reuse exact arrival labels with per-cover epochs; no arithmetic/model change.
    pub metric_epochs: bool,
    /// Retain only the existing trial-budget prefix of ranked local proposals.
    pub bounded_proposals: bool,
    /// Fraction of the remaining global recovery window available to seed work.
    pub seed_window_fraction: f64,
    /// Keep the normal incumbent, then try worklist recovery before the same deadline.
    pub flow_compare: bool,
    /// Rank local recovery by critical-path improvement before area; opt-in.
    pub critical_recovery: bool,
    /// Rank bounded optional cuts by earliest structural boundary depth; opt-in.
    pub cut_depth_priority: bool,
    /// Reserve this many optional slots for depth ranking, keeping the total cap.
    pub cut_depth_quota: usize,
    /// Exact-library timing cut ranking and bounded original-class reassociation; opt-in.
    pub library_timing: bool,
    /// Spend only remaining recovery time on required-time-aware area recovery.
    pub slack_flow: bool,
    /// Rank capped producers by witness input arrival and dependent pin delay.
    pub timing_cap: bool,
    /// Reject actual pool, node and recovery overruns; diagnostic opt-in.
    pub strict_budgets: bool,
    /// Canonical physical phases across exact logical Not edges; opt-in.
    pub canonical_phases: bool,
    /// Preserve original logical cut banks/interfaces; canonicalize physical wires only.
    pub canonical_wire_only: bool,
    /// Global eclass regions for physical matching; zero preserves legacy ordering.
    pub region_classes: usize,
    /// Joint opposite-phase recovery under the existing shared recovery deadline.
    pub phase_recovery: bool,
    pub literal_import: bool,
    pub literal_auto: bool,
    /// Exact bounded-cone import above the fixed90% capacity threshold; opt-in.
    pub cone_import: bool,
    /// Optional exact source preprocessing; all admitted enodes remain globally charged.
    pub shared_majority: bool,
    pub proof_source: bool,
    pub source_propagation: bool,
    /// Retain an already validated scored cover when application time runs out.
    pub scored_incumbent: bool,
    /// Repair otherwise cyclic substitutions with the already validated fast all-signal witness.
    pub joint_cones: bool,
    /// Atomically retain a complete fast physical input/phase witness inside the same candidate cap.
    pub timing_witness_pool: bool,
    pub online_area: bool,
    /// Keep legacy recovery/pool trajectory unless a validated root delay improves.
    pub preserve_shared_trajectory: bool,
    /// Fuse witness fill, full physical validation and pin metrics; same cover semantics.
    pub fused_cover_scoring: bool,
    /// Conservative live-sharing lower bound filters only non-improving area trials.
    pub shared_area_bound: bool,
    pub online_shared_bound: bool,
    /// Refresh actual arrival/required labels after committed online substitutions.
    pub refresh_area_timing: bool,
    /// Recover area from the best validated tail reference while retaining its predecessor.
    pub tail_shared_area: bool,
    /// Continue after complete legacy recovery, using unused rounds/window only.
    pub terminal_area: bool,
    pub protected_exchange: bool,
    pub exchange_functional_seed: bool,
    pub retain_generation_frontier: bool,
    pub fixed_area_bound: bool,
    pub protected_area_limit: Option<f64>,
    /// Total full-care quotient cuts using existing counted classes; default off.
    pub functional_cuts: bool,
    pub functional_phase_screen: bool,
    pub delta_area_filter: bool,
    pub cut_pin_flow: bool,
    /// Exact pre-admission source operand permutation/TT normalization.
    pub canonical_source_inputs: bool,
    /// Reallocate unused fixed initial node capacity to structural helper functions.
    pub import_helper_fill: bool,
    /// Introduce already-admitted inverse bindings only after faster-reference activation.
    pub introduce_phase_inverters: bool,
    pub activate_deferred: bool,
    /// Fraction of the remaining post-seed window reserved for late timing-constrained area recovery.
    pub area_time_fraction: f64,
    pub return_reserve: f64,
    pub fast_area_scan: bool,
    /// Maximum fraction of the remaining global recovery window for phase trials.
    pub phase_time_fraction: f64,
    pub recover_live_refs: bool,
    pub recover_cones: bool,
    /// Independent wall-clock budget for heuristic cover recovery, including warm starts.
    pub refine_time_secs: f64,
}

impl Default for MapParams {
    fn default() -> Self {
        Self {
            saturation: SatParams::default(),
            max_inputs: 6,
            cuts_per_class: 32,
            cut_rounds: 32,
            max_instances: 100_000,
            multioutput: true,
            multioutput_cumulative: true,
            dc_support: 0,
            dc_cone: 1024,
            single_structure_cuts: false,
            no_input_phase_matches: false,
            canonical_pin_matching: false,
            no_actual_area_tail: false,
            ablate_functional_cuts: false,
            ablate_shared_area: false,
            ablate_incumbent_guard: false,
            time_limit_secs: 30.0,
            delay_weight: 0.0,
            objective: MapObjective::Weighted,
            max_delay: None,
            refine_rounds: 4,
            refine_trials: 2000,
            boolean_rules: 0,
            flow_delay_ties: false,
            structural_refs: false,
            flow_worklist: false,
            fast_delay_seed: false,
            required_area_dag: false,
            defer_area_dag: false,
            preserve_legacy_seeds: false,
            metric_epochs: false,
            bounded_proposals: false,
            seed_window_fraction: 1.0,
            flow_compare: false,
            critical_recovery: false,
            cut_depth_priority: false,
            cut_depth_quota: 0,
            library_timing: false,
            slack_flow: false,
            timing_cap: false,
            strict_budgets: false,
            canonical_phases: false,
            canonical_wire_only: false,
            region_classes: 0,
            phase_recovery: false,
            literal_import: false,
            literal_auto: false,
            cone_import: false,
            shared_majority: false,
            proof_source: false,
            source_propagation: false,
            scored_incumbent: false,
            joint_cones: false,
            timing_witness_pool: false,
            online_area: false,
            preserve_shared_trajectory: false,
            fused_cover_scoring: false,
            shared_area_bound: false,
            online_shared_bound: false,
            refresh_area_timing: false,
            tail_shared_area: false,
            terminal_area: false,
            protected_exchange: false,
            exchange_functional_seed: false,
            retain_generation_frontier: false,
            fixed_area_bound: false,
            protected_area_limit: None,
            functional_cuts: false,
            functional_phase_screen: false,
            delta_area_filter: false,
            cut_pin_flow: false,
            canonical_source_inputs: false,
            import_helper_fill: false,
            introduce_phase_inverters: false,
            activate_deferred: false,
            area_time_fraction: 0.0,
            return_reserve: 0.05,
            fast_area_scan: false,
            phase_time_fraction: 1.0,
            recover_live_refs: true,
            recover_cones: true,
            refine_time_secs: 2.0,
        }
    }
}

/// A signal is `2 * dense_eclass + complemented_phase`.
#[derive(Clone, Debug)]
pub struct OutputPin {
    pub name: String,
    pub delays: Vec<f32>,
    /// Function of the *physical*, already phased inputs in Instance::inputs order.
    pub function: u64,
    pub targets: Vec<usize>,
    pub sources: Vec<BindingSource>,
}

impl OutputPin {
    fn has_arc(&self, input: usize) -> bool {
        const LOW: [u64;6]=[0x5555555555555555,0x3333333333333333,0x0f0f0f0f0f0f0f0f,0x00ff00ff00ff00ff,0x0000ffff0000ffff,0x00000000ffffffff];
        // Check one member of each input-toggle pair in the bounded truth table.
        // Exactly the same dependence predicate as assignment enumeration.
        ((self.function ^ (self.function >> (1usize << input))) & LOW[input] & mask(self.delays.len())) != 0
    }
}

#[derive(Clone, Debug)]
pub struct Instance {
    pub cell: String,
    pub area: f32,
    pub inputs: Vec<usize>,
    pub input_pins: Vec<String>,
    pub outputs: Vec<OutputPin>,
}

#[derive(Clone, Copy, Debug)]
pub struct Binding {
    pub instance: usize,
    pub pin: usize,
}

#[derive(Clone, Debug, Default)]
pub struct MapStats {
    pub cut_rounds_started: usize,
    pub exchange_trials: usize,
    pub exchange_cuts: usize,
    pub protected_bindings: usize,
    pub exchange_source_bindings: usize,
    pub exchange_fallback: bool,
    pub functional_cut_trials: usize,
    pub functional_cut_proofs: usize,
    pub functional_cut_admitted: usize,
    pub regions: usize,
    pub cross_region_boundaries: usize,
    pub phase_trials: usize,
    pub joint_attempts: usize,
    pub joint_valid: usize,
    pub online_area_updates: usize,
    pub phase_accepted: usize,
    pub phase_secs: f64,
    pub saturation_budget_skips: usize,
    /// Resident nodes after import rebuild (legacy field, not lifetime budget).
    pub imported_enodes: usize,
    pub cumulative_imported_enodes: usize,
    pub cumulative_admitted_enodes: usize,
    pub phase_initial_delay: f64,
    pub phase_final_delay: f64,
    pub eclasses: usize,
    pub enodes: usize,
    pub cuts: usize,
    pub max_cuts_per_class: usize,
    pub instances: usize,
    pub generated_instances: usize,
    pub instance_budget_pruned: usize,
    pub bindings: usize,
    pub dc_boundaries: usize,
    pub dc_with_unreachable: usize,
    pub dc_skipped: usize,
    pub dc_bindings: usize,
    pub cut_limit_hit: bool,
    /// Final-round optional boundaries replacing the legacy ranking on the same pool.
    pub cut_depth_swaps: usize,
    pub instance_limit_hit: bool,
    pub selected_multioutput: usize,
    pub selected_joint_multioutput: usize,
    pub used_fallback: bool,
    /// Optimal only within the bounded generated candidate set.
    pub proven_optimal: bool,
    pub solver_status: String,
    pub saturation_stop: String,
    pub build_secs: f64,
    pub solve_secs: f64,
    pub initial_area: f64,
    pub initial_delay: f64,
    /// Fastest complete seed seen, not a lower bound or optimality certificate.
    pub seed_delay: f64,
    pub heuristic_area: f64,
    pub heuristic_secs: f64,
    pub refinement_trials: usize,
    pub flow_compare_attempted: bool,
    pub flow_compare_accepted: bool,
    pub refinement_accepted: usize,
    pub slack_attempts: usize,
    pub slack_valid: usize,
    pub slack_accepted: usize,
    pub slack_rounds_started: usize,
    pub terminal_area_attempted: bool,
    pub terminal_area_accepted: bool,
    pub slack_local_scored: usize,
    pub slack_local_trials: usize,
    pub slack_local_accepted: usize,
}

pub struct MappedGraph {
    pub instances: Vec<Instance>,
    pub chosen: Vec<Option<Binding>>,
    pub inputs: Vec<(usize, u32, String)>,
    pub outputs: Vec<(usize, String)>,
    pub constants: Vec<Option<bool>>,
    pub area: f64,
    pub delay: f64,
    pub stats: MapStats,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct Cut {
    leaves: Vec<usize>,
    function: u64,
}

fn mask(k: usize) -> u64 {
    if k == 6 {
        u64::MAX
    } else {
        (1u64 << (1usize << k)) - 1
    }
}

fn expand(function: u64, from: &[usize], to: &[usize]) -> u64 {
    let positions: Vec<_> = from.iter().map(|x| to.binary_search(x).unwrap()).collect();
    let mut result = 0;
    for a in 0..1usize << to.len() {
        let old = positions
            .iter()
            .enumerate()
            .fold(0, |v, (i, p)| v | (((a >> p) & 1) << i));
        result |= ((function >> old) & 1) << a;
    }
    result
}

fn reduce(mut cut: Cut) -> Cut {
    let k = cut.leaves.len();
    let essential: Vec<_> = (0..k)
        .filter(|&i| {
            (0..1usize << k)
                .any(|a| ((cut.function >> a) ^ (cut.function >> (a ^ (1 << i)))) & 1 != 0)
        })
        .collect();
    if essential.len() < k {
        let mut tt = 0;
        for a in 0..1usize << essential.len() {
            let old = essential
                .iter()
                .enumerate()
                .fold(0, |v, (i, p)| v | (((a >> i) & 1) << p));
            tt |= ((cut.function >> old) & 1) << a;
        }
        cut.leaves = essential.iter().map(|&i| cut.leaves[i]).collect();
        cut.function = tt;
    }
    cut
}

// Optional is ordered by the legacy key. Keep budget-quota of those cuts,
// then fill only the remaining slots in depth order. Overlap never wastes slots.
fn mixed_optional_cuts(
    optional: &[Cut],
    depth: &[usize],
    budget: usize,
    quota: usize,
) -> BTreeSet<Cut> {
    let mut selected: BTreeSet<_> = optional
        .iter()
        .take(budget.saturating_sub(quota.min(budget)))
        .cloned()
        .collect();
    if quota == 0 {
        return selected;
    }
    let mut ranked: Vec<_> = optional.iter().enumerate().collect();
    ranked.sort_by_key(|(rank, cut)| {
        (
            cut.leaves
                .iter()
                .map(|&leaf| depth[leaf])
                .max()
                .unwrap_or(0),
            *rank,
        )
    });
    for (_, cut) in ranked {
        if selected.len() >= budget {
            break;
        }
        selected.insert(cut.clone());
    }
    selected
}

struct BoundedAssociativity {
    rhs: egg::PatternAst<AigLang>,
    original_classes: usize,
    nodes: usize,
    per_class: usize,
}
impl egg::Applier<AigLang, ()> for BoundedAssociativity {
    fn get_pattern_ast(&self) -> Option<&egg::PatternAst<AigLang>> { Some(&self.rhs) }
    fn vars(&self) -> Vec<egg::Var> { vec!["?a".parse().unwrap(), "?b".parse().unwrap(), "?c".parse().unwrap()] }

    fn apply_one(
        &self,
        graph: &mut egg::EGraph<AigLang, ()>,
        eclass: Id,
        subst: &egg::Subst,
        _ast: Option<&egg::PatternAst<AigLang>>,
        _rule: egg::Symbol,
    ) -> Vec<Id> {
        if usize::from(eclass) >= self.original_classes
            || graph[eclass].nodes.len() >= self.per_class
            || graph.total_size().saturating_add(2) > self.nodes
        {
            return Vec::new();
        }
        let a = subst["?a".parse().unwrap()];
        let b = subst["?b".parse().unwrap()];
        let c = subst["?c".parse().unwrap()];
        let bc = graph.add(AigLang::And([b, c]));
        let root = graph.add(AigLang::And([a, bc]));
        if graph.union(eclass, root) {
            vec![eclass]
        } else {
            Vec::new()
        }
    }
}

// Exact logical Not edges only. No conditional/DC relation is admitted.
fn phase_aliases(nodes: &[Vec<AigLang>], index: impl Fn(Id)->usize) -> Result<Vec<usize>,String> {
    fn find(p:&mut [usize],x:usize)->usize {
        if p[x]!=x {p[x]=find(p,p[x]);} p[x]
    }
    fn join(p:&mut [usize],a:usize,b:usize) {
        let x=find(p,a);let y=find(p,b);if x!=y {p[x.max(y)]=x.min(y);}
    }
    let mut parent: Vec<_>=(0..nodes.len()*2).collect();
    for (c,ns) in nodes.iter().enumerate() {for n in ns {
        if let AigLang::Not(a)=n {let a=index(*a);join(&mut parent,c*2,a*2+1);join(&mut parent,c*2+1,a*2);}
    }}
    let result: Vec<_>=(0..parent.len()).map(|s|find(&mut parent,s)).collect();
    for s in 0..result.len() {
        if result[s^1]!=(result[s]^1) {return Err("contradictory exact phase alias".into());}
    }
    Ok(result)
}

fn normalize_phase_cut(cut:&Cut, aliases:&[usize])->Cut {
    let leaves: Vec<_>=cut.leaves.iter().map(|&c|aliases[c*2]/2).collect::<BTreeSet<_>>().into_iter().collect();
    let mut function=0;
    for a in 0..1usize<<leaves.len() {
        let old=cut.leaves.iter().enumerate().fold(0,|v,(i,&c)| {
            let s=aliases[c*2];let bit=((a>>leaves.iter().position(|&x|x==s/2).unwrap())&1)^(s%2);
            v|(bit<<i)
        });function|=((cut.function>>old)&1)<<a;
    }
    reduce(Cut{leaves,function})
}

fn canonical_cut_banks(
    cuts: &[BTreeSet<Cut>], nodes:&[Vec<AigLang>], index:impl Fn(Id)->usize,
    aliases:&[usize], ps:&MapParams, stats:&mut MapStats,
) -> Result<Vec<BTreeSet<Cut>>,String> {
    let mut all=vec![BTreeSet::new();nodes.len()];let mut mandatory=all.clone();
    for c in 0..nodes.len() {
        let s=aliases[c*2];let rep=s/2;
        mandatory[rep].insert(Cut{leaves:vec![rep],function:2});
        let convert=|cut:&Cut| {
            let mut x=normalize_phase_cut(cut,aliases);
            if s%2!=0 {x.function^=mask(x.leaves.len());} x
        };
        for cut in &cuts[c] {let x=convert(cut);if !x.leaves.contains(&rep){all[rep].insert(x);}}
        for n in &nodes[c] {
            let x=match n {
                AigLang::Zero=>Cut{leaves:vec![],function:0},
                AigLang::One=>Cut{leaves:vec![],function:1},
                AigLang::Not(a)=>Cut{leaves:vec![index(*a)],function:1},
                AigLang::And([a,b])=> {let leaves:Vec<_>=[index(*a),index(*b)].into_iter().collect::<BTreeSet<_>>().into_iter().collect();let function=if leaves.len()==1 {2}else{8};Cut{leaves,function}},
                _=>continue,
            };
            let x=convert(&x);if !x.leaves.contains(&rep){mandatory[rep].insert(x);}
        }
    }
    let small_quota=std::env::var("EGM_SMALL_CUT_QUOTA").ok().and_then(|s|s.parse::<usize>().ok()).unwrap_or(0);
    let primary:Vec<_>=nodes.iter().map(|ns|ns.iter().any(|n|matches!(n,AigLang::Pi(_)|AigLang::Zero|AigLang::One))).collect();
    let mut result=vec![BTreeSet::new();nodes.len()];
    for c in 0..nodes.len() {
        if mandatory[c].is_empty(){continue;}
        if mandatory[c].len()>ps.cuts_per_class {return Err(format!("budget: canonical mandatory cuts {} exceed {}",mandatory[c].len(),ps.cuts_per_class));}
        let budget=ps.cuts_per_class-mandatory[c].len();
        let mut optional:Vec<_>=all[c].difference(&mandatory[c]).cloned().collect();
        optional.sort_by_key(|x| {let internal=x.leaves.iter().filter(|&&i|!primary[i]).count();(x.leaves.len()+internal,internal,x.leaves.len(),x.clone())});
        stats.cut_limit_hit|=optional.len()>budget;
        let mut selected:BTreeSet<_>=optional.iter().filter(|x|x.leaves.len()<=3).take(small_quota.min(budget)).cloned().collect();
        for x in &optional {if selected.len()>=budget{break;}selected.insert(x.clone());}
        result[c]=mandatory[c].clone();result[c].extend(selected);
    }
    stats.cuts=result.iter().map(|x|x.len()).sum();Ok(result)
}

/// Ranking labels only: exact local-cell implementations on mandatory cuts.
/// No label is used as a timing bound or a substitute for full-cover validation.
fn mandatory_pin_arrivals(
    mandatory:&[BTreeSet<Cut>], nodes:&[Vec<AigLang>], aliases:&[usize], tech:&TechLib,
)->Vec<f64> {
    use std::collections::VecDeque;
    let n=nodes.len()*2;
    let mut label=vec![f64::INFINITY;n];
    for (c,ns) in nodes.iter().enumerate(){
        if ns.iter().any(|x|matches!(x,AigLang::Pi(_))){label[aliases[c*2]]=0.;}
        if ns.iter().any(|x|matches!(x,AigLang::Zero|AigLang::One)){label[aliases[c*2]]=0.;label[aliases[c*2+1]]=0.;}
    }
    let mut edges:Vec<(usize,Vec<usize>,Vec<f32>)>=Vec::new();
    for c in 0..nodes.len(){
        for phase in 0..2 {edges.push((aliases[c*2+phase],vec![aliases[c*2+(phase^1)]],vec![tech.inv_delay]));}
        for cut in &mandatory[c]{
            let k=cut.leaves.len();if k==0 || cut.leaves==[c]{continue;}
            for phase in 0..2 {
                let truth=cut.function^if phase==1{mask(k)}else{0};
                if let Some(variants)=tech.candidates(k as u8,tt9_from_u64(truth)){
                    for v in variants.iter().filter(|v|!v.out_neg){
                        edges.push((aliases[c*2+phase],cut.leaves.iter().enumerate().map(|(p,&l)|aliases[l*2+usize::from(v.neg_mask&(1<<p)!=0)]).collect(),v.pin_delay.clone()));
                    }
                }
            }
        }
    }
    let mut users=vec![Vec::new();n];
    for (i,(_,inputs,_)) in edges.iter().enumerate(){for &s in inputs{users[s].push(i);}}
    let mut queue:VecDeque<_>=(0..edges.len()).collect();let mut queued=vec![true;edges.len()];
    while let Some(i)=queue.pop_front(){
        queued[i]=false;let (target,inputs,delays)=&edges[i];
        let value=inputs.iter().zip(delays).map(|(&s,&d)|label[s]+f64::from(d)).fold(0.,f64::max);
        if value<label[*target]{label[*target]=value;for &j in &users[*target]{if !queued[j]{queued[j]=true;queue.push_back(j);}}}
    }
    label
}
fn pin_arrival_cut_phase_scores(cut:&Cut,labels:&[f64],aliases:&[usize],tech:&TechLib)->[f64;2] {
    let k=cut.leaves.len();if k==0{return [0.;2];}
    let mut best=[f64::INFINITY;2];
    for phase in 0..2 {
        let truth=cut.function^if phase==1{mask(k)}else{0};
        if let Some(variants)=tech.candidates(k as u8,tt9_from_u64(truth)){
            for v in variants.iter().filter(|v|!v.out_neg){
                let value=cut.leaves.iter().enumerate().map(|(p,&l)|labels[aliases[l*2+usize::from(v.neg_mask&(1<<p)!=0)]]+f64::from(v.pin_delay[p])).fold(0.,f64::max);
                best[phase]=best[phase].min(value);
            }
        }
    }
    best
}

fn pin_arrival_cut_score(cut:&Cut,labels:&[f64],aliases:&[usize],tech:&TechLib)->f64 {
 pin_arrival_cut_phase_scores(cut,labels,aliases,tech).into_iter().fold(f64::INFINITY,f64::min)
}
fn phase_screen_accepts(proposal:[f64;2],reference:[f64;2])->bool {
 proposal.iter().all(|v|v.is_finite()) && (0..2).all(|p|proposal[p]<=reference[p]) && (0..2).any(|p|proposal[p]<reference[p])
}
// A quotient is admitted only if every new boundary assignment has a unique
// full-care value. Neither SDC completion nor conditional eclass unions are used.
fn exact_functional_quotient(root:&Cut, helper:usize, part:&Cut)->Option<Cut> {
    if !(3..=4).contains(&root.leaves.len()) || part.leaves.len()<2
        || part.leaves.len()>=root.leaves.len() || part.function==0
        || part.function==mask(part.leaves.len())
        || !part.leaves.iter().all(|v|root.leaves.contains(v)) {return None;}
    let rest:Vec<_>=root.leaves.iter().filter(|v|!part.leaves.contains(v)).copied().collect();
    if rest.contains(&helper){return None;}
    let leaves:Vec<_>=rest.iter().copied().chain([helper]).collect::<BTreeSet<_>>().into_iter().collect();
    let mut seen=0u64;let mut truth=0u64;
    for x in 0..1usize<<root.leaves.len() {
        let sub=part.leaves.iter().enumerate().fold(0,|v,(i,p)|v|(((x>>root.leaves.binary_search(p).unwrap())&1)<<i));
        let bit=((part.function>>sub)&1)as usize;
        let a=leaves.iter().enumerate().fold(0,|v,(i,p)|v|((if *p==helper{bit}else{(x>>root.leaves.binary_search(p).unwrap())&1})<<i));
        let value=(root.function>>x)&1;
        if seen&(1<<a)!=0 && ((truth>>a)&1)!=value{return None;}
        seen|=1<<a;truth|=value<<a;
    }
    if seen!=mask(leaves.len()){return None;}
    Some(reduce(Cut{leaves,function:truth}))
}

// At most four existing partners per support, thirty-two proof attempts per
// root per ORIGINAL cut round; temporary index only, not another retained bank.
fn functional_support_index(old:&[BTreeSet<Cut>])->BTreeMap<Vec<usize>,Vec<(usize,Cut)>> {
    functional_support_index_until(old,None).expect("unbounded original index")
}
fn functional_support_index_until(old:&[BTreeSet<Cut>],deadline:Option<Instant>)->Option<BTreeMap<Vec<usize>,Vec<(usize,Cut)>>> {
    if deadline.is_some_and(|d|Instant::now()>=d){return None;}
    let mut index:BTreeMap<Vec<usize>,Vec<(usize,Cut)>>=BTreeMap::new();let mut visited=0usize;
    for (g,bank) in old.iter().enumerate(){for cut in bank {
        if visited%128==0 && deadline.is_some_and(|d|Instant::now()>=d){return None;}visited+=1;
        if !(2..=3).contains(&cut.leaves.len()) || cut.leaves.contains(&g)
            || cut.function==0 || cut.function==mask(cut.leaves.len()){continue;}
        let bucket=index.entry(cut.leaves.clone()).or_default();
        if bucket.len()<4 {bucket.push((g,cut.clone()));}
    }}
    if deadline.is_some_and(|d|Instant::now()>=d){None}else{Some(index)}
}
fn functional_proposals(root:usize,bases:&BTreeSet<Cut>,index:&BTreeMap<Vec<usize>,Vec<(usize,Cut)>>,tech:&TechLib,stats:&mut MapStats)->BTreeSet<Cut> {
    let mut out=BTreeSet::new();let mut trials=0;
    'bases:for base in bases.iter().filter(|b|(3..=4).contains(&b.leaves.len())) {
        for subset in 1..(1usize<<base.leaves.len())-1 {
            if subset.count_ones()<2{continue;}
            let support:Vec<_>=base.leaves.iter().enumerate().filter_map(|(i,&v)|(subset&(1<<i)!=0).then_some(v)).collect();
            if let Some(partners)=index.get(&support){for (g,part) in partners {
                if *g==root || base.leaves.contains(g){continue;}
                if trials>=32{break 'bases;}trials+=1;stats.functional_cut_trials+=1;
                if let Some(cut)=exact_functional_quotient(base,*g,part){
                    stats.functional_cut_proofs+=1;let k=cut.leaves.len();
                    if k>0 && (0..2).any(|p|tech.candidates(k as u8,tt9_from_u64(cut.function^if p==1{mask(k)}else{0})).is_some()) {out.insert(cut);}
                }
            }}
        }
    }out
}

fn enumerate(
    nodes: &[Vec<AigLang>],
    ids: &HashMap<Id, usize>,
    graph: &egg::EGraph<AigLang, ()>,
    tech: &TechLib,
    ps: &MapParams,
    stats: &mut MapStats,
) -> Vec<BTreeSet<Cut>> {
    enumerate_with_source(nodes,ids,graph,tech,ps,stats,&[])
}

fn enumerate_mapping_cuts(nodes:&[Vec<AigLang>],witness_nodes:&[Vec<AigLang>],ids:&HashMap<Id,usize>,graph:&egg::EGraph<AigLang,()>,tech:&TechLib,ps:&MapParams,stats:&mut MapStats,source_pairs:&[regions::SourceAnd])->Vec<BTreeSet<Cut>> {
 let cut_nodes=if ps.single_structure_cuts {witness_nodes} else {nodes};
 let cut_source=source_pairs; // Identical mandatory physical fallback in both settings.
 enumerate_with_source(cut_nodes,ids,graph,tech,ps,stats,cut_source)
}

fn enumerate_with_source(
    nodes: &[Vec<AigLang>],
    ids: &HashMap<Id, usize>,
    graph: &egg::EGraph<AigLang, ()>,
    tech: &TechLib,
    ps: &MapParams,
    stats: &mut MapStats,
    source_pairs:&[regions::SourceAnd],
) -> Vec<BTreeSet<Cut>> {
    let index = |id: Id| ids[&graph.find(id)];
    let phase_map=if ps.canonical_phases {phase_aliases(nodes,index).expect("checked phase aliases")} else {(0..nodes.len()*2).collect()};
    let mut source=vec![Vec::new();nodes.len()];
    for pair in source_pairs{source[index(pair.root)].push((index(pair.left),index(pair.right),pair.left_neg,pair.right_neg));}
    let area_quota = std::env::var("EGM_AREA_CUT_QUOTA").ok().and_then(|s|s.parse::<usize>().ok()).unwrap_or(0);
    let mut area_scores: HashMap<(usize,u64),f64> = HashMap::new();
    let mut cuts: Vec<BTreeSet<Cut>> = (0..nodes.len())
        .map(|c| {
            BTreeSet::from([Cut {
                leaves: vec![c],
                function: 2,
            }])
        })
        .collect();
    // Always retain local enode realizations, even when the optional cut/round budget is tiny.
    for (c, enodes) in nodes.iter().enumerate() {
        for node in enodes {
            let cut = match node {
                AigLang::Zero => Cut {
                    leaves: vec![],
                    function: 0,
                },
                AigLang::One => Cut {
                    leaves: vec![],
                    function: 1,
                },
                AigLang::Not(a) => Cut {
                    leaves: vec![index(*a)],
                    function: 1,
                },
                node if tt3_inputs(node).is_some()=>{
                    let (args,tt)=tt3_inputs(node).unwrap();let inputs:Vec<_>=args.iter().map(|&id|index(id)).collect();let leaves:Vec<_>=inputs.iter().copied().collect::<BTreeSet<_>>().into_iter().collect();let mut function=0;
                    for assignment in 0..1usize<<leaves.len(){let tuple=inputs.iter().enumerate().map(|(p,id)|((assignment>>leaves.binary_search(id).unwrap())&1)<<p).sum::<usize>();function|=((tt>>tuple)&1)<<assignment;}reduce(Cut{leaves,function})
                }
                node if literal_inputs(node).is_some() || xor_inputs(node).is_some() => {
                    let is_xor=xor_inputs(node).is_some();let (a,b,phases)=literal_inputs(node).or_else(||xor_inputs(node)).unwrap();
                    let ai=index(a);let bi=index(b);
                    let leaves:Vec<_>=[ai,bi].into_iter().collect::<BTreeSet<_>>().into_iter().collect();
                    let ap=leaves.binary_search(&ai).unwrap();let bp=leaves.binary_search(&bi).unwrap();
                    let mut function=0;
                    for assignment in 0..1usize<<leaves.len() {
                        let av=((assignment>>ap)&1!=0)^(phases&1!=0);
                        let bv=((assignment>>bp)&1!=0)^(phases&2!=0);
                        function|=u64::from(if is_xor{av^bv}else{av&bv})<<assignment;
                    }
                    reduce(Cut{function,leaves})
                }
                _ => continue,
            };
            let cut=if ps.canonical_phases {normalize_phase_cut(&cut,&phase_map)} else {cut};
            if !cut.leaves.contains(&c) {
                cuts[c].insert(cut);
            }
        }
    }
    for (root,bank) in source.iter().enumerate(){for &(a,b,an,bn) in bank{
        let leaves:Vec<_>=[a,b].into_iter().collect::<BTreeSet<_>>().into_iter().collect();let mut function=0;for x in 0..1usize<<leaves.len(){let av=((x>>leaves.binary_search(&a).unwrap())&1!=0)^an;let bv=((x>>leaves.binary_search(&b).unwrap())&1!=0)^bn;function|=u64::from(av&bv)<<x;}let cut=reduce(Cut{leaves,function});let cut=if ps.canonical_phases{normalize_phase_cut(&cut,&phase_map)}else{cut};if !cut.leaves.contains(&root){cuts[root].insert(cut);}
    }}
    let mandatory = cuts.clone();
    let pin_labels=if ps.cut_pin_flow || ps.functional_phase_screen {mandatory_pin_arrivals(&mandatory,nodes,&phase_map,tech)}else{Vec::new()};
    let mut pin_scores:HashMap<Cut,f64>=HashMap::new();
    let is_primary: Vec<_> = nodes
        .iter()
        .map(|ns| {
            ns.iter()
                .any(|n| matches!(n, AigLang::Pi(_) | AigLang::Zero | AigLang::One))
        })
        .collect();
    // Structural depth is a ranking heuristic, not a technology-delay bound.
    // Only finite enode witnesses participate; exact structural cuts stay mandatory.
    let mut boundary_depth = vec![usize::MAX; nodes.len()];
    if ps.cut_depth_priority
        || ps.cut_depth_quota > 0
        || (ps.library_timing && ps.objective == MapObjective::DelayArea)
    {
        for _ in 0..nodes.len() {
            let mut changed = false;
            for (c, enodes) in nodes.iter().enumerate() {
                for node in enodes {
                    let d = match node {
                        AigLang::Pi(_) | AigLang::Zero | AigLang::One => 0,
                        AigLang::Not(a) => boundary_depth[index(*a)].saturating_add(1),
                        node if literal_inputs(node).is_some() || xor_inputs(node).is_some() => {
                            let (a,b,_)=literal_inputs(node).or_else(||xor_inputs(node)).unwrap();
                            boundary_depth[index(a)].max(boundary_depth[index(b)]).saturating_add(1)
                        },
                        node if tt3_inputs(node).is_some()=>tt3_inputs(node).unwrap().0.iter().map(|&id|boundary_depth[index(id)]).max().unwrap_or(0).saturating_add(1),
                        _ => usize::MAX,
                    };
                    if d < boundary_depth[c] {
                        boundary_depth[c] = d;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }
    let mut stable = false;
    for _ in 0..ps.cut_rounds {
        stats.cut_rounds_started += 1;
        let old = cuts.clone();
        let functional_index=if ps.functional_cuts && !ps.ablate_functional_cuts {functional_support_index(&old)}else{BTreeMap::new()};
        let mut changed = false;
        stats.cut_depth_swaps = 0;
        for (c, enodes) in nodes.iter().enumerate() {
            let mut pending = BTreeSet::new();
            for node in enodes {
                match node {
                    AigLang::Zero => {
                        pending.insert(Cut {
                            leaves: vec![],
                            function: 0,
                        });
                    }
                    AigLang::One => {
                        pending.insert(Cut {
                            leaves: vec![],
                            function: 1,
                        });
                    }
                    AigLang::Not(a) => {
                        for cut in &old[index(*a)] {
                            pending.insert(Cut {
                                leaves: cut.leaves.clone(),
                                function: cut.function ^ mask(cut.leaves.len()),
                            });
                        }
                    }
                    node if tt3_inputs(node).is_some() && source[c].is_empty()=>{
                        for base in &old[c]{if base.leaves.contains(&c){continue}for (position,&leaf) in base.leaves.iter().enumerate(){for child in &old[leaf]{if child.leaves==vec![leaf]&&child.function==2{continue}let leaves:Vec<_>=base.leaves.iter().enumerate().filter_map(|(p,&s)|(p!=position).then_some(s)).chain(child.leaves.iter().copied()).collect::<BTreeSet<_>>().into_iter().collect();if leaves.len()>ps.max_inputs{continue}let mut function=0;for a in 0..1usize<<leaves.len(){let sub=child.leaves.iter().enumerate().map(|(p,s)|((a>>leaves.binary_search(s).unwrap())&1)<<p).sum::<usize>();let bit=((child.function>>sub)&1)as usize;let tuple=base.leaves.iter().enumerate().map(|(p,s)|if p==position{bit<<p}else{((a>>leaves.binary_search(s).unwrap())&1)<<p}).sum::<usize>();function|=((base.function>>tuple)&1)<<a;}let cut=reduce(Cut{leaves,function});pending.insert(if ps.canonical_phases{normalize_phase_cut(&cut,&phase_map)}else{cut});}}}
                    }
                    node if literal_inputs(node).is_some() || xor_inputs(node).is_some() => {
                        let is_xor=xor_inputs(node).is_some();let (a,b,phases)=literal_inputs(node).or_else(||xor_inputs(node)).unwrap();
                        for x in &old[index(a)] { for y in &old[index(b)] {
                            let leaves:Vec<_>=x.leaves.iter().chain(&y.leaves).copied().collect::<BTreeSet<_>>().into_iter().collect();
                            if leaves.len()<=ps.max_inputs {
                                let mut left=expand(x.function,&x.leaves,&leaves);
                                let mut right=expand(y.function,&y.leaves,&leaves);
                                if phases&1!=0 {left^=mask(leaves.len());}
                                if phases&2!=0 {right^=mask(leaves.len());}
                                let cut=reduce(Cut{leaves,function:if is_xor{left^right}else{left&right}});
                                pending.insert(if ps.canonical_phases {normalize_phase_cut(&cut,&phase_map)} else {cut});
                            }
                        }}
                    }
                    _ => {}
                }
            }
            for &(a,b,an,bn) in &source[c]{for x in &old[a]{for y in &old[b]{let leaves:Vec<_>=x.leaves.iter().chain(&y.leaves).copied().collect::<BTreeSet<_>>().into_iter().collect();if leaves.len()<=ps.max_inputs{let mut left=expand(x.function,&x.leaves,&leaves);let mut right=expand(y.function,&y.leaves,&leaves);if an{left^=mask(leaves.len());}if bn{right^=mask(leaves.len());}let cut=reduce(Cut{leaves,function:left&right});pending.insert(if ps.canonical_phases{normalize_phase_cut(&cut,&phase_map)}else{cut});}}}}
            // Reserving structural cuts guarantees a finite fallback. Rank optional cuts by a
            // fixed key so newly discovered PI boundaries can displace early shallow internal cuts.
            pending.extend(cuts[c].iter().cloned());
            let mut functional=if ps.functional_cuts && !ps.ablate_functional_cuts {functional_proposals(c,&pending,&functional_index,tech,stats)}else{BTreeSet::new()};
            if ps.functional_phase_screen {
                let mut reference=[f64::INFINITY;2];
                for cut in pending.iter().filter(|x|!x.leaves.contains(&c)) {
                    let score=pin_arrival_cut_phase_scores(cut,&pin_labels,&phase_map,tech);
                    for p in 0..2 {reference[p]=reference[p].min(score[p]);}
                }
                functional.retain(|cut|phase_screen_accepts(pin_arrival_cut_phase_scores(cut,&pin_labels,&phase_map,tech),reference));
            }
            pending.extend(functional.iter().cloned());
            let mut optional: Vec<_> = pending
                .into_iter()
                .map(|cut|if ps.canonical_phases {normalize_phase_cut(&cut,&phase_map)} else {cut})
                .collect::<BTreeSet<_>>().into_iter()
                .filter(|cut| !cut.leaves.contains(&c) && !mandatory[c].contains(cut))
                .collect();
            let budget = ps.cuts_per_class.saturating_sub(mandatory[c].len());
            stats.cut_limit_hit |= optional.len() > budget;
            let legacy_key = |cut: &Cut| {
                let internal = cut.leaves.iter().filter(|&&leaf| !is_primary[leaf]).count();
                (
                    cut.leaves.len() + internal,
                    internal,
                    cut.leaves.len(),
                    cut.clone(),
                )
            };
            optional.sort_by_key(&legacy_key);
            let legacy: BTreeSet<_> = optional.iter().take(budget).cloned().collect();
            let mut next = mandatory[c].clone();
            let quota = if ps.cut_depth_priority {
                budget
            } else {
                ps.cut_depth_quota.min(budget)
            };
            let mut selected = if ps.cut_pin_flow && ps.objective==MapObjective::DelayArea {
                for cut in &optional {pin_scores.entry(cut.clone()).or_insert_with(||pin_arrival_cut_score(cut,&pin_labels,&phase_map,tech));}
                optional.sort_by(|a,b|pin_scores[a].total_cmp(&pin_scores[b]).then(legacy_key(a).cmp(&legacy_key(b))));
                optional.iter().take(budget).cloned().collect()
            } else if ps.library_timing && ps.objective == MapObjective::DelayArea {
                let unit = tech.and2_delay.iter().copied().fold(0.0, f32::max) as f64;
                let score = |cut: &Cut| {
                    let k = cut.leaves.len();
                    if k == 0 {
                        return 0.0;
                    }
                    (0..2)
                        .flat_map(|phase| {
                            tech.candidates(
                                k as u8,
                                tt9_from_u64(cut.function ^ if phase == 1 { mask(k) } else { 0 }),
                            )
                            .into_iter()
                            .flatten()
                            .filter(|v| !v.out_neg)
                        })
                        .map(|v| {
                            cut.leaves
                                .iter()
                                .enumerate()
                                .map(|(p, &leaf)| {
                                    boundary_depth[leaf] as f64 * unit
                                        + f64::from(v.pin_delay[p])
                                        + if v.neg_mask & (1 << p) != 0 {
                                            f64::from(tech.inv_delay)
                                        } else {
                                            0.0
                                        }
                                })
                                .fold(0.0, f64::max)
                        })
                        .fold(f64::INFINITY, f64::min)
                };
                optional.sort_by(|a, b| {
                    score(a)
                        .total_cmp(&score(b))
                        .then(legacy_key(a).cmp(&legacy_key(b)))
                });
                optional.iter().take(budget).cloned().collect()
            } else {
                mixed_optional_cuts(&optional, &boundary_depth, budget, quota)
            };
            // Diagnostic prototype: preserve small-support diversity inside the
            // existing optional budget. No new cut slots or library changes.
            let small_quota = std::env::var("EGM_SMALL_CUT_QUOTA").ok()
                .and_then(|v| v.parse::<usize>().ok()).unwrap_or(0).min(budget);
            if small_quota > 0 || area_quota > 0 {
                let mut wide: Vec<_> = optional.iter().filter(|cut|cut.leaves.len()>3).collect();
                let mut score = |cut: &Cut| {
                    let k=cut.leaves.len();
                    *area_scores.entry((k,cut.function)).or_insert_with(|| {
                        (0..2).flat_map(|phase|tech.candidates(k as u8,
                            tt9_from_u64(cut.function ^ if phase==1 {mask(k)} else {0}))
                            .into_iter().flatten().filter(|v|!v.out_neg))
                            .map(|v| (f64::from(v.area)+f64::from(tech.inv_area)*v.neg_mask.count_ones() as f64)
                                / k.saturating_sub(1).max(1) as f64)
                            .fold(f64::INFINITY,f64::min)
                    })
                };
                if area_quota > 0 {
                    wide.sort_by(|a,b|score(a).total_cmp(&score(b)).then(legacy_key(a).cmp(&legacy_key(b))));
                }
                let mut mixed: BTreeSet<Cut> = wide.into_iter().filter(|cut|score(cut).is_finite())
                    .take(area_quota.min(budget)).cloned().collect();
                for cut in optional.iter().filter(|cut|cut.leaves.len()<=3).take(small_quota) {
                    if mixed.len()>=budget {break;}
                    mixed.insert(cut.clone());
                }
                for cut in optional.iter().filter(|cut| selected.contains(*cut)) {
                    if mixed.len() >= budget { break; }
                    mixed.insert(cut.clone());
                }
                for cut in &optional {
                    if mixed.len() >= budget { break; }
                    mixed.insert(cut.clone());
                }
                selected = mixed;
            }
            stats.cut_depth_swaps += selected.difference(&legacy).count();
            stats.functional_cut_admitted += selected.intersection(&functional).count();
            next.extend(selected);
            changed |= next != cuts[c];
            cuts[c] = next;
        }
        if !changed {
            stable = true;
            break;
        }
    }
    stats.cut_limit_hit |= !stable;
    stats.cuts = cuts.iter().map(BTreeSet::len).sum();
    cuts
}

/// Finite representatives are used only to prove boundary reachability, never to choose the cover.
struct Reachability {
    nodes: Vec<AigLang>,
    ids: HashMap<Id, usize>,
    cache: HashMap<Vec<usize>, Option<u64>>,
}

impl Reachability {
    fn care(&mut self, leaves: &[usize], ps: &MapParams, stats: &mut MapStats) -> u64 {
        let full = mask(leaves.len());
        if ps.dc_support == 0 {
            return full;
        }
        if let Some(care) = self.cache.get(leaves) {
            return care.unwrap_or(full);
        }
        let mut cone = BTreeSet::new();
        let mut vars = BTreeSet::new();
        let mut stack = leaves.to_vec();
        while let Some(c) = stack.pop() {
            if !cone.insert(c) {
                continue;
            }
            match self.nodes[c] {
                AigLang::Pi(v) => {
                    vars.insert(v);
                }
                AigLang::Not(a) => stack.push(self.ids[&a]),
                AigLang::And([a, b]) => {
                    stack.push(self.ids[&a]);
                    stack.push(self.ids[&b]);
                }
                _ => {
                    if let Some((args,_))=tt3_inputs(&self.nodes[c]){stack.extend(args.iter().map(|id|self.ids[id]));}
                    if let Some((a,b,_))=literal_inputs(&self.nodes[c]).or_else(||xor_inputs(&self.nodes[c])) {
                        stack.push(self.ids[&a]);stack.push(self.ids[&b]);
                    }
                }
            }
            if cone.len() > ps.dc_cone || vars.len() > ps.dc_support {
                self.cache.insert(leaves.to_vec(), None);
                stats.dc_skipped += 1;
                return full;
            }
        }
        let vars: Vec<_> = vars.into_iter().collect();
        let mut care = 0;
        for assignment in 0..1usize << vars.len() {
            let mut memo = vec![None; self.nodes.len()];
            let mut tuple = 0;
            for (i, &c) in leaves.iter().enumerate() {
                tuple |= usize::from(self.eval(c, &vars, assignment, &mut memo)) << i;
            }
            care |= 1u64 << tuple;
            if care == full {
                break;
            }
        }
        stats.dc_boundaries += 1;
        stats.dc_with_unreachable += usize::from(care != full);
        self.cache.insert(leaves.to_vec(), Some(care));
        care
    }

    fn eval(&self, c: usize, vars: &[u32], a: usize, memo: &mut [Option<bool>]) -> bool {
        if let Some(v) = memo[c] {
            return v;
        }
        let v = match self.nodes[c] {
            AigLang::Zero => false,
            AigLang::One => true,
            AigLang::Pi(v) => (a >> vars.binary_search(&v).unwrap()) & 1 != 0,
            AigLang::Not(x) => !self.eval(self.ids[&x], vars, a, memo),
            AigLang::And([x, y]) => {
                self.eval(self.ids[&x], vars, a, memo) & self.eval(self.ids[&y], vars, a, memo)
            }
            ref node if tt3_inputs(node).is_some()=>{let (args,tt)=tt3_inputs(node).unwrap();let tuple=args.iter().enumerate().map(|(p,id)|usize::from(self.eval(self.ids[id],vars,a,memo))<<p).sum::<usize>();(tt>>tuple)&1!=0},
            AigLang::Cell(_, _) => {
                let node=&self.nodes[c];let (x,y,phases)=literal_inputs(node).or_else(||xor_inputs(node)).expect("unknown logical macro");
                let left=self.eval(self.ids[&x],vars,a,memo)^(phases&1!=0);let right=self.eval(self.ids[&y],vars,a,memo)^(phases&2!=0);if xor_inputs(node).is_some(){left^right}else{left&right}
            },
        };
        memo[c] = Some(v);
        v
    }
}

fn admit_input_assignment(phase:usize,pins:&[String],canonical:&[String],ps:&MapParams)->bool {
    (!ps.no_input_phase_matches || phase==0) && (!ps.canonical_pin_matching || pins==canonical)
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    fn visit(n: usize, p: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if p.len() == n {
            out.push(p.clone());
            return;
        }
        for i in 0..n {
            if !p.contains(&i) {
                p.push(i);
                visit(n, p, out);
                p.pop();
            }
        }
    }
    let mut out = Vec::new();
    visit(n, &mut Vec::new(), &mut out);
    out
}

fn physical_function(cell: &Cell, pins: &[String]) -> u64 {
    let positions: Vec<_> = cell
        .vars
        .iter()
        .map(|p| pins.iter().position(|x| x == p).unwrap())
        .collect();
    let mut tt = 0;
    for a in 0..1usize << pins.len() {
        let original = positions
            .iter()
            .enumerate()
            .fold(0, |v, (i, p)| v | (((a >> p) & 1) << i));
        tt |= ((cell.tt[0] >> original) & 1) << a;
    }
    tt
}

fn input_phase(tt: u64, n: usize, phase: usize) -> u64 {
    let mut out = 0;
    for a in 0..1usize << n {
        out |= ((tt >> (a ^ phase)) & 1) << a;
    }
    out
}

fn subsets(n: usize, k: usize) -> Vec<Vec<usize>> {
    (0..1usize << n)
        .filter(|bits| bits.count_ones() as usize == k)
        .map(|bits| (0..n).filter(|i| bits & (1 << i) != 0).collect())
        .collect()
}

type InstanceKey = (String, Vec<usize>, Vec<(String, u64, Vec<u32>)>);

fn instance_key(instance: &Instance) -> InstanceKey {
    let mut order: Vec<_> = (0..instance.inputs.len()).collect();
    order.sort_by_key(|&i| instance.inputs[i]);
    let inputs = order.iter().map(|&i| instance.inputs[i]).collect();
    let outputs = instance
        .outputs
        .iter()
        .map(|o| {
            let mut function = 0;
            for a in 0..1usize << order.len() {
                let old = order
                    .iter()
                    .enumerate()
                    .fold(0, |v, (i, p)| v | (((a >> i) & 1) << p));
                function |= ((o.function >> old) & 1) << a;
            }
            let delays = order.iter().map(|&i| o.delays[i].to_bits()).collect();
            (o.name.clone(), function, delays)
        })
        .collect();
    (instance.cell.clone(), inputs, outputs)
}

fn normalize_physical_instance(m:&mut Instance,aliases:&[usize],free:&[bool])->bool {
    for s in &mut m.inputs {*s=aliases[*s];}
    for o in &mut m.outputs {
        for s in &mut o.targets {*s=aliases[*s];}
        o.targets.retain(|s|!free[*s] && !m.inputs.contains(s));
        o.targets.sort_unstable();o.targets.dedup();
        exchange::normalize_sources(o,aliases);
    }
    m.outputs.iter().any(|o|!o.targets.is_empty())
}

fn insert_instance(
    instances: &mut Vec<Instance>,
    index: &mut HashMap<InstanceKey, usize>,
    mut instance: Instance,
) {
    // Collapse pin symmetries only when every output function AND timing arc is preserved.
    let key = instance_key(&instance);
    insert_instance_key(instances,index,instance,key);
}
fn insert_instance_key(instances:&mut Vec<Instance>,index:&mut HashMap<InstanceKey,usize>,mut instance:Instance,key:InstanceKey)->usize {
    if let Some(&old) = index.get(&key) {
        for (output, new) in instances[old].outputs.iter_mut().zip(&mut instance.outputs) {
            output.targets.extend(new.targets.iter().copied());
            output.targets.sort_unstable();
            output.targets.dedup();
            for source in &new.sources {if !output.sources.contains(source){output.sources.push(source.clone());}}
        }
        old
    } else {
        let id=instances.len();index.insert(key,id);
        instances.push(instance);id
    }
}

#[derive(Clone, Copy)]
struct Target {
    signal: usize,
    function: u64,
}

fn targets_for(tt: u64, targets: &[Target], care: u64, stats: &mut MapStats) -> Vec<usize> {
    targets
        .iter()
        .filter_map(|target| {
            if (tt ^ target.function) & care == 0 {
                stats.dc_bindings += usize::from(tt != target.function);
                Some(target.signal)
            } else {
                None
            }
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn generate(
    cuts: &[BTreeSet<Cut>],
    free: &[bool],
    constants: &[Option<bool>],
    tech: &TechLib,
    ps: &MapParams,
    reach: &mut Reachability,
    stats: &mut MapStats,
    physical_aliases: Option<(&[usize], &[bool])>,
    deadline: Option<Instant>,
) -> Vec<Instance> {
    let mut boundaries: BTreeMap<Vec<usize>, Vec<Target>> = BTreeMap::new();
    for (c, alternatives) in cuts.iter().enumerate() {
        for cut in alternatives {
            if cut.leaves == [c] {
                continue;
            }
            for phase in 0..2 {
                let s = c * 2 + phase;
                if free[s] {
                    continue;
                }
                boundaries
                    .entry(cut.leaves.clone())
                    .or_default()
                    .push(Target {
                        signal: s,
                        function: cut.function
                            ^ if phase == 1 {
                                mask(cut.leaves.len())
                            } else {
                                0
                            },
                    });
            }
        }
    }
    let mut groups: BTreeMap<&str, Vec<&Cell>> = BTreeMap::new();
    for cell in &tech.cells {
        groups.entry(&cell.name).or_default().push(cell);
    }
    let multi_names: HashSet<_> = groups
        .iter()
        .filter(|(_, cells)| cells.len() > 1)
        .map(|(&name, _)| name)
        .collect();
    // Boundary unions let outputs with different supports share a complete physical pin set.
    if ps.multioutput && !multi_names.is_empty() {
        let union_limit = groups
            .values()
            .filter(|cells| cells.len() > 1)
            .map(|cells| {
                cells
                    .iter()
                    .flat_map(|c| &c.vars)
                    .collect::<HashSet<_>>()
                    .len()
            })
            .max()
            .unwrap_or(0)
            .min(ps.max_inputs);
        let originals: Vec<_> = boundaries
            .keys()
            .filter(|b| !b.is_empty())
            .cloned()
            .collect();
        let mut by_leaf: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, b) in originals.iter().enumerate() {
            for &leaf in b {
                let bucket = by_leaf.entry(leaf).or_default();
                if bucket.len() < 64 {
                    bucket.push(i);
                }
            }
        }
        let mut unions = BTreeSet::new();
        for (i, b) in originals.iter().enumerate() {
            // Shared-leaf pairing alone misses cells whose output supports are disjoint.
            // A bounded neighboring-boundary window adds such candidates without quadratic
            // enumeration. This is deliberately incomplete under a candidate budget.
            let shared = b
                .iter()
                .flat_map(|l| by_leaf[l].iter().copied())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .take(16);
            let neighbors = (i + 1..originals.len().min(i + 17)).into_iter();
            let partners: BTreeSet<_> = shared.chain(neighbors).collect();
            let mut accumulated = b.clone();
            for partner in partners {
                let union: Vec<_> = b
                    .iter()
                    .chain(&originals[partner])
                    .copied()
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect();
                if union.len() <= union_limit && !boundaries.contains_key(&union) {
                    unions.insert(union);
                }
                // Pairwise unions cannot cover three or more independent output supports.
                // Also accumulate the same bounded partner window, retaining intermediate
                // boundaries. Oversized partners are skipped, not allowed to poison the prefix.
                let combined: Vec<_> = accumulated
                    .iter()
                    .chain(&originals[partner])
                    .copied()
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect();
                if ps.multioutput_cumulative
                    && combined.len() <= union_limit
                    && combined != accumulated
                {
                    accumulated = combined;
                    if !boundaries.contains_key(&accumulated) {
                        unions.insert(accumulated.clone());
                    }
                }
            }
            if unions.len() >= ps.max_instances {
                break;
            }
        }
        for b in unions {
            boundaries.entry(b).or_default();
        }
        let source = boundaries.clone();
        for (boundary, targets) in &mut boundaries {
            for subset in 0..1usize << boundary.len() {
                let small: Vec<_> = boundary
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| subset & (1 << i) != 0)
                    .map(|(_, &c)| c)
                    .collect();
                if small == *boundary {
                    continue;
                }
                if let Some(existing) = source.get(&small) {
                    targets.extend(
                        existing
                            .iter()
                            .filter(|t| !boundary.contains(&(t.signal / 2)))
                            .map(|t| Target {
                                signal: t.signal,
                                function: expand(t.function, &small, boundary),
                            }),
                    );
                }
            }
        }
    }
    let ordered_boundaries =
        regions::ordered_boundaries(&boundaries, cuts.len(), ps.region_classes, stats);
    let mut instances = Vec::new();
    // Baseline cells first. Optional DC/multi-output variants cannot exhaust the fallback budget.
    for &(boundary, targets) in &ordered_boundaries {
        if deadline.is_some_and(|d|Instant::now()>=d){return instances;}
        let k = boundary.len();
        if k == 0 {
            continue;
        }
        let mut seen = HashSet::new();
        for target in targets {
            if deadline.is_some_and(|d|Instant::now()>=d){return instances;}
            if let Some(variants) = tech.candidates(k as u8, tt9_from_u64(target.function)) {
                for v in variants {
                    if v.out_neg || multi_names.contains(v.cell_name.as_str()) {
                        continue;
                    }
                    let key = (v.cell_name.clone(), v.pin_names.clone(), v.neg_mask);
                    if !seen.insert(key) {
                        continue;
                    }
                    let cell = tech
                        .cells
                        .iter()
                        .find(|c| c.name == v.cell_name && c.out_pin == v.out_pin)
                        .unwrap();
                    if !admit_input_assignment(v.neg_mask as usize,&v.pin_names,&cell.vars,ps) {continue;}
                    let physical = physical_function(cell, &v.pin_names);
                    let actual = input_phase(physical, k, usize::from(v.neg_mask));
                    let inputs: Vec<_> = boundary
                        .iter()
                        .enumerate()
                        .map(|(i, c)| c * 2 + usize::from((v.neg_mask >> i) & 1))
                        .collect();
                    let matched: Vec<_> = targets
                        .iter()
                        .filter(|t| t.function == actual && !inputs.contains(&t.signal))
                        .map(|t| t.signal)
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .collect();
                    if !matched.is_empty() {
                        let mut generated=Instance {
                            cell: v.cell_name.clone(),
                            area: v.area,
                            inputs,
                            input_pins: v.pin_names.clone(),
                            outputs: vec![OutputPin {sources:Vec::new(),
                                name: v.out_pin.clone(),
                                delays: v.pin_delay.clone(),
                                function: physical,
                                targets: matched,
                            }],
                        };
 if ps.protected_exchange {exchange::record_matches(&mut generated,boundary,targets,mask(k),physical_aliases);}
 instances.push(generated);
                    }
                }
            }
        }
    }
    if let Some(inv) = &tech.inv_variant {
        for c in 0..cuts.len() {
            if deadline.is_some_and(|d|Instant::now()>=d){return instances;}
            if cuts[c].is_empty(){continue;}
            for phase in 0..2 {
                if free[c * 2 + phase] {
                    continue;
                }
                let mut generated=Instance {
                    cell: inv.cell_name.clone(),
                    area: inv.area,
                    inputs: vec![c * 2 + (1 - phase)],
                    input_pins: inv.pin_names.clone(),
                    outputs: vec![OutputPin {sources:Vec::new(),
                        name: inv.out_pin.clone(),
                        delays: inv.pin_delay.clone(),
                        function: 1,
                        targets: vec![c * 2 + phase],
                    }],
                };
 if ps.protected_exchange {exchange::record_inverter(&mut generated,physical_aliases);}
 instances.push(generated);
            }
        }
    }
    // Cells with no inputs can realize a class whose precise expression is constant.
    for cell in tech
        .cells
        .iter()
        .filter(|c| c.vars.is_empty() && !multi_names.contains(c.name.as_str()))
    {
        let value = cell.tt[0] & 1 != 0;
        let targets = cuts
            .iter()
            .enumerate()
            .flat_map(|(c, cs)| {
                (0..2).filter_map(move |phase| {
                    if !free[c * 2 + phase]
                        && constants[c * 2 + phase].is_none()
                        && cs.iter().any(|cut| {
                            cut.leaves.is_empty()
                                && ((cut.function & 1 != 0) ^ (phase == 1)) == value
                        })
                    {
                        Some(c * 2 + phase)
                    } else {
                        None
                    }
                })
            })
            .collect();
        let mut generated=Instance {
            cell: cell.name.clone(),
            area: cell.area,
            inputs: vec![],
            input_pins: vec![],
            outputs: vec![OutputPin {sources:Vec::new(),
                name: cell.out_pin.clone(),
                delays: vec![],
                function: u64::from(value),
                targets,
            }],
        };
 if ps.protected_exchange {let matched:Vec<_>=generated.outputs[0].targets.iter().map(|&signal|Target{signal,function:u64::from(value)}).collect();exchange::record_matches(&mut generated,&[],&matched,1,physical_aliases);}
 instances.push(generated);
    }
    let mut index = HashMap::new();
    let base = std::mem::take(&mut instances);
    for (i,mut instance) in base.into_iter().enumerate() {
        if i%128==0 && deadline.is_some_and(|d|Instant::now()>=d){return instances;}
        if let Some((aliases,free))=physical_aliases {
            if !normalize_physical_instance(&mut instance,aliases,free){continue;}
        }
        if ps.protected_exchange {let key=exchange::recorded_key(&instance).unwrap_or_else(||instance_key(&instance));insert_instance_key(&mut instances,&mut index,instance,key);} else {insert_instance(&mut instances, &mut index, instance);}
    }
    let base_count = instances.len();
    stats.instance_limit_hit |= base_count > ps.max_instances;
    for &(boundary, targets) in &ordered_boundaries {
        if deadline.is_some_and(|d|Instant::now()>=d){return instances;}
        let k = boundary.len();
        if k == 0 {
            continue;
        }
        let care = reach.care(boundary, ps, stats);
        for (&name, cells) in &groups {
            let multi = cells.len() > 1;
            if (multi && !ps.multioutput) || (!multi && care == mask(k)) {
                continue;
            }
            let pins: Vec<_> = cells
                .iter()
                .flat_map(|c| c.vars.iter().cloned())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            let n = pins.len();
            if n > k {
                continue;
            }
            // Reject inconsistent descriptions rather than charging one of two different areas.
            if cells.iter().any(|c| (c.area - cells[0].area).abs() > 1e-6) {
                continue;
            }
            for selection in subsets(k, n) {
                let selected: Vec<_> = selection.iter().map(|&i| boundary[i]).collect();
                for perm in permutations(n) {
                    let input_pins: Vec<_> = perm.iter().map(|&i| pins[i].clone()).collect();
                    let physical: Vec<_> = cells
                        .iter()
                        .map(|c| physical_function(c, &input_pins))
                        .collect();
                    for phase in 0..1usize << n {
                        if !admit_input_assignment(phase,&input_pins,&pins,ps) {continue;}
                        let inputs: Vec<_> = selected
                            .iter()
                            .enumerate()
                            .map(|(i, c)| c * 2 + ((phase >> i) & 1))
                            .collect();
                        let outputs: Vec<_> = cells
                            .iter()
                            .zip(&physical)
                            .map(|(cell, &function)| {
                                let actual =
                                    expand(input_phase(function, n, phase), &selected, boundary);
                                let mut matched = targets_for(actual, targets, care, stats);
                                matched.retain(|s| !inputs.contains(s));
                                OutputPin {sources:Vec::new(),
                                    name: cell.out_pin.clone(),
                                    function,
                                    targets: matched,
                                    delays: input_pins
                                        .iter()
                                        .map(|p| cell.pin_delay.get(p).copied().unwrap_or(0.0))
                                        .collect(),
                                }
                            })
                            .collect();
                        if outputs.iter().any(|o| !o.targets.is_empty()) {
                            let mut instance = Instance {
                                cell: name.into(),
                                area: cells[0].area,
                                inputs,
                                input_pins: input_pins.clone(),
                                outputs,
                            };
                            if ps.protected_exchange {exchange::record_matches(&mut instance,boundary,targets,care,physical_aliases);}
                            if let Some((aliases,free))=physical_aliases {
                                if !normalize_physical_instance(&mut instance,aliases,free){continue;}
                            }
                            let key = instance_key(&instance);
                            if !index.contains_key(&key)
                                && instances.len() >= ps.max_instances.max(base_count)
                            {
                                stats.instance_limit_hit = true;
                                return instances;
                            }
                            if ps.protected_exchange {let key=exchange::recorded_key(&instance).unwrap_or_else(||instance_key(&instance));insert_instance_key(&mut instances,&mut index,instance,key);} else {insert_instance(&mut instances, &mut index, instance);}
                        }
                    }
                }
            }
        }
    }
    instances
}

fn finite_choices(instances: &[Instance], free: &[bool]) -> Vec<Option<Binding>> {
    let mut available = free.to_vec();
    let mut chosen = vec![None; free.len()];
    let mut order: Vec<_> = (0..instances.len()).collect();
    order.sort_by(|&a, &b| instances[a].area.total_cmp(&instances[b].area));
    loop {
        let mut changed = false;
        for &i in &order {
            let m = &instances[i];
            if !m.inputs.iter().all(|&s| available[s]) {
                continue;
            }
            for (pin, output) in m.outputs.iter().enumerate() {
                for &s in &output.targets {
                    if !available[s] {
                        chosen[s] = Some(Binding { instance: i, pin });
                        available[s] = true;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    chosen
}

fn finite_choices_queued(instances: &[Instance], free: &[bool]) -> Vec<Option<Binding>> {
    let mut available = free.to_vec();
    let mut chosen = vec![None; free.len()];
    let mut order: Vec<_> = (0..instances.len()).collect();
    order.sort_by(|&a, &b| instances[a].area.total_cmp(&instances[b].area));
    let mut rank = vec![0; instances.len()];
    for (r, &i) in order.iter().enumerate() {
        rank[i] = r;
    }
    let mut users = vec![Vec::new(); free.len()];
    let mut remaining = vec![0; instances.len()];
    let mut current = BTreeSet::new();
    let mut next = BTreeSet::new();
    for (i, m) in instances.iter().enumerate() {
        for input in m.inputs.iter().copied().collect::<BTreeSet<_>>() {
            if !available[input] {
                users[input].push(i);
                remaining[i] += 1;
            }
        }
        if remaining[i] == 0 {
            current.insert(rank[i]);
        }
    }
    loop {
        while let Some(r) = current.pop_first() {
            let i = order[r];
            for (pin, output) in instances[i].outputs.iter().enumerate() {
                for &signal in &output.targets {
                    if available[signal] {
                        continue;
                    }
                    available[signal] = true;
                    chosen[signal] = Some(Binding { instance: i, pin });
                    for &consumer in &users[signal] {
                        remaining[consumer] -= 1;
                        if remaining[consumer] == 0 {
                            if rank[consumer] > r {
                                current.insert(rank[consumer]);
                            } else {
                                next.insert(rank[consumer]);
                            }
                        }
                    }
                }
            }
        }
        if next.is_empty() {
            break;
        }
        std::mem::swap(&mut current, &mut next);
    }
    chosen
}

#[cfg(test)]
fn fallback(
    instances: &[Instance],
    free: &[bool],
    roots: &[usize],
) -> Result<Vec<Option<Binding>>, String> {
    let mut chosen = finite_choices(instances, free);
    prune(instances, &mut chosen, free, roots)?;
    Ok(chosen)
}

fn fill_from_witness(
    instances: &[Instance],
    chosen: &mut [Option<Binding>],
    witness: &[Option<Binding>],
    free: &[bool],
    roots: &[usize],
) -> Result<(), String> {
    let mut seen = vec![false; free.len()];
    let mut stack = roots.to_vec();
    while let Some(s) = stack.pop() {
        if free[s] || seen[s] {
            continue;
        }
        seen[s] = true;
        let binding = chosen[s]
            .or(witness[s])
            .ok_or_else(|| format!("unrealizable dependency {s}"))?;
        chosen[s] = Some(binding);
        stack.extend(instances[binding.instance].inputs.iter().copied());
    }
    prune(instances, chosen, free, roots)
}

fn better_cover(candidate: (f64, f64), current: (f64, f64), ps: &MapParams) -> bool {
    let feasible = |delay| ps.max_delay.is_none_or(|limit| delay <= limit + 1e-6);
    match (feasible(candidate.1), feasible(current.1)) {
        (true, false) => true,
        (false, true) => false,
        (false, false) => candidate.1 < current.1 - 1e-6,
        (true, true) => {
            let a = ps.objective.key(candidate, ps.delay_weight);
            let b = ps.objective.key(current, ps.delay_weight);
            a.0 < b.0 - 1e-6 || ((a.0 - b.0).abs() <= 1e-6 && a.1 < b.1 - 1e-6)
        }
    }
}

/// Estimate reclaimable physical area when producers of the listed signals are replaced.
/// Shared inputs and other live pins keep their physical instance alive. Sparse counters avoid
/// copying the entire graph for every ranked proposal. New-input repair is evaluated separately.
fn reclaimable_area(
    targets: &[usize],
    instances: &[Instance],
    chosen: &[Option<Binding>],
    free: &[bool],
    signal_refs: &[usize],
    output_users: &[usize],
) -> f64 {
    let mut refs = HashMap::new();
    let mut users = HashMap::new();
    let mut released = HashSet::new();
    let mut pending = targets.to_vec();
    let mut area = 0.0;
    while let Some(s) = pending.pop() {
        if free[s] || !released.insert(s) {
            continue;
        }
        let Some(binding) = chosen[s] else { continue };
        let i = binding.instance;
        let count = users.entry(i).or_insert(output_users[i]);
        *count -= 1;
        if *count != 0 {
            continue;
        }
        area += f64::from(instances[i].area);
        for &input in instances[i].inputs.iter().collect::<BTreeSet<_>>() {
            let count = refs.entry(input).or_insert(signal_refs[input]);
            *count -= 1;
            if *count == 0 {
                pending.push(input);
            }
        }
    }
    area
}

/// Relax positive tree/flow costs over all e-class alternatives. Depth breaks zero-cost ties;
/// the resulting graph is validated before it can become either an incumbent or a repair witness.
fn flow_choices(
    instances: &[Instance],
    free: &[bool],
    refs: &[f64],
    weight: f64,
    divide_inputs: bool,
    delay_ties: bool,
    objective: MapObjective,
    deadline: Instant,
) -> (Vec<Option<Binding>>, Vec<f64>) {
    let mut costs = vec![f64::INFINITY; free.len()];
    let mut arrival = vec![f64::INFINITY; free.len()];
    let mut depth = vec![usize::MAX; free.len()];
    let mut chosen = vec![None; free.len()];
    for (s, &yes) in free.iter().enumerate() {
        if yes {
            costs[s] = 0.0;
            arrival[s] = 0.0;
            depth[s] = 0;
        }
    }
    for _ in 0..32 {
        let mut changed = false;
        for (i, m) in instances.iter().enumerate() {
            if !m.inputs.iter().all(|&s| costs[s].is_finite()) {
                continue;
            }
            let input_cost: f64 = m
                .inputs
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .map(|s| costs[s])
                .sum();
            let level = m
                .inputs
                .iter()
                .map(|&s| depth[s])
                .max()
                .unwrap_or(0)
                .saturating_add(1);
            for (pin, output) in m.outputs.iter().enumerate() {
                let time = m
                    .inputs
                    .iter()
                    .enumerate()
                    .filter(|(p, _)| output.has_arc(*p))
                    .map(|(p, &s)| arrival[s] + f64::from(output.delays[p]))
                    .fold(0.0, f64::max);
                for &s in &output.targets {
                    if free[s] || m.inputs.contains(&s) {
                        continue;
                    }
                    let area = if divide_inputs {
                        (f64::from(m.area) + input_cost) / refs[s]
                    } else {
                        f64::from(m.area) / refs[s] + input_cost
                    };
                    let score = area + weight * time;
                    let old = if weight == 0.0 {
                        costs[s]
                    } else {
                        costs[s] + weight * arrival[s]
                    };
                    // Equal area-flow covers still differ in arrival. Prefer the faster
                    // realization before depth, even in pure area mode; otherwise candidate
                    // insertion order can lock in a slower pin with exactly the same cost.
                    let tied = (score - old).abs() <= 1e-9;
                    let better_tie = if delay_ties {
                        time < arrival[s] - 1e-9
                            || ((time - arrival[s]).abs() <= 1e-9 && level < depth[s])
                    } else {
                        level < depth[s]
                    };
                    let prefer = if objective == MapObjective::Weighted {
                        score < old - 1e-9 || (tied && better_tie)
                    } else {
                        let a = objective.key((area, time), weight);
                        let b = objective.key((costs[s], arrival[s]), weight);
                        a.0 < b.0 - 1e-9
                            || ((a.0 - b.0).abs() <= 1e-9
                                && (a.1 < b.1 - 1e-9
                                    || ((a.1 - b.1).abs() <= 1e-9 && level < depth[s])))
                    };
                    if prefer {
                        costs[s] = area;
                        arrival[s] = time;
                        depth[s] = level;
                        chosen[s] = Some(Binding { instance: i, pin });
                        changed = true;
                    }
                }
            }
            if Instant::now() >= deadline {
                return (chosen, costs);
            }
        }
        if !changed {
            break;
        }
    }
    (chosen, costs)
}

fn flow_choices_worklist(
    instances: &[Instance],
    free: &[bool],
    refs: &[f64],
    weight: f64,
    divide_inputs: bool,
    delay_ties: bool,
    objective: MapObjective,
    deadline: Instant,
) -> (Vec<Option<Binding>>, Vec<f64>) {
    flow_choices_worklist_impl(
        instances,
        free,
        refs,
        weight,
        divide_inputs,
        delay_ties,
        objective,
        deadline,
        None,
        None,
    )
}

// Preserve ascending unique-ID summation exactly, without a per-candidate tree allocation.
fn unique_input_cost(inputs: &[usize], costs: &[f64]) -> f64 {
    if inputs.len() <= 8 {
        let mut ids = [0usize; 8];
        let ids = &mut ids[..inputs.len()];
        ids.copy_from_slice(inputs);
        ids.sort_unstable();
        let mut last = None;
        ids.iter().copied().filter(|&s| {
            let novel = last != Some(s); last = Some(s); novel
        }).map(|s| costs[s]).sum()
    } else {
        let mut ids = inputs.to_vec(); ids.sort_unstable(); ids.dedup();
        ids.into_iter().map(|s| costs[s]).sum()
    }
}

// Tree area is only a flow proxy: a finite reconvergent DAG can have an
// exponentially large unfolded tree. Keep DA reachability distinct from this
// overflow. Infinity from an unavailable input remains unavailable; only sums
// of already finite nonnegative labels are capped. Actual DAG area/delay are
// never capped, and complete-cover acceptance still uses the original objective.
fn flow_input_cost(inputs: &[usize], costs: &[f64], bounded: bool) -> f64 {
    let sum = unique_input_cost(inputs, costs);
    if bounded && sum.is_infinite() && inputs.iter().all(|&s| costs[s].is_finite()) {
        f64::MAX
    } else { sum }
}

fn flow_choices_worklist_impl(
    instances: &[Instance],
    free: &[bool],
    refs: &[f64],
    weight: f64,
    divide_inputs: bool,
    delay_ties: bool,
    objective: MapObjective,
    deadline: Instant,
    required: Option<&[f64]>,
    seed: Option<&[Option<Binding>]>,
) -> (Vec<Option<Binding>>, Vec<f64>) {
    let mut costs = vec![f64::INFINITY; free.len()];
    let mut arrival = vec![f64::INFINITY; free.len()];
    let mut depth = vec![usize::MAX; free.len()];
    let mut chosen = vec![None; free.len()];
    for (s, &yes) in free.iter().enumerate() {
        if yes {
            costs[s] = 0.0;
            arrival[s] = 0.0;
            depth[s] = 0;
        }
    }
    if let Some(seed) = seed {
        fn init(
            s: usize,
            instances: &[Instance],
            free: &[bool],
            seed: &[Option<Binding>],
            refs: &[f64],
            bounded: bool,
            costs: &mut [f64],
            arrival: &mut [f64],
            depth: &mut [usize],
        ) {
            if free[s] || costs[s].is_finite() {
                return;
            }
            let Some(b) = seed[s] else {
                return;
            };
            let m = &instances[b.instance];
            for &input in &m.inputs {
                init(input, instances, free, seed, refs, bounded, costs, arrival, depth);
            }
            costs[s] = (f64::from(m.area)
                + flow_input_cost(&m.inputs, costs, bounded))
                / refs[s].max(1.0);
            arrival[s] = m
                .inputs
                .iter()
                .enumerate()
                .filter(|(p, _)| m.outputs[b.pin].has_arc(*p))
                .map(|(p, &i)| arrival[i] + f64::from(m.outputs[b.pin].delays[p]))
                .fold(0.0, f64::max);
            depth[s] = m
                .inputs
                .iter()
                .map(|&i| depth[i])
                .max()
                .unwrap_or(0)
                .saturating_add(1);
        }
        for s in 0..free.len() {
            init(
                s,
                instances,
                free,
                seed,
                refs,
                objective == MapObjective::DelayArea,
                &mut costs,
                &mut arrival,
                &mut depth,
            );
        }
        chosen.clone_from_slice(seed);
    }
    // A changed signal wakes precisely its physical consumers. Unlike a fixed
    // number of full sweeps, this can settle arbitrarily deep finite cones.
    let mut consumers = vec![Vec::new(); free.len()];
    for (i, m) in instances.iter().enumerate() {
        for &input in &m.inputs {
            consumers[input].push(i);
        }
    }
    let mut queue: std::collections::VecDeque<_> = (0..instances.len()).collect();
    let mut pending = vec![true; instances.len()];
    while let Some(i) = queue.pop_front() {
        if Instant::now() >= deadline {
            break;
        }
        pending[i] = false;
        let m = &instances[i];
        if !m.inputs.iter().all(|&s| costs[s].is_finite()) {
            continue;
        }
        let input_cost: f64 = flow_input_cost(&m.inputs, &costs, objective == MapObjective::DelayArea);
        let level = m
            .inputs
            .iter()
            .map(|&s| depth[s])
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        for (pin, output) in m.outputs.iter().enumerate() {
            let time = m
                .inputs
                .iter()
                .enumerate()
                .filter(|(p, _)| output.has_arc(*p))
                .map(|(p, &s)| arrival[s] + f64::from(output.delays[p]))
                .fold(0.0, f64::max);
            for &s in &output.targets {
                if free[s] || m.inputs.contains(&s) || required.is_some_and(|r| time > r[s] + 1e-6)
                {
                    continue;
                }
                let area = if divide_inputs {
                    (f64::from(m.area) + input_cost) / refs[s]
                } else {
                    f64::from(m.area) / refs[s] + input_cost
                };
                let score = area + weight * time;
                let old = if weight == 0.0 {
                    costs[s]
                } else {
                    costs[s] + weight * arrival[s]
                };
                // Equal area-flow covers still differ in arrival. Prefer the faster
                // realization before depth, even in pure area mode; otherwise candidate
                // insertion order can lock in a slower pin with exactly the same cost.
                let tied = (score - old).abs() <= 1e-9;
                let better_tie = if delay_ties {
                    time < arrival[s] - 1e-9
                        || ((time - arrival[s]).abs() <= 1e-9 && level < depth[s])
                } else {
                    level < depth[s]
                };
                let prefer = if objective == MapObjective::Weighted {
                    score < old - 1e-9 || (tied && better_tie)
                } else {
                    let a = objective.key((area, time), weight);
                    let b = objective.key((costs[s], arrival[s]), weight);
                    a.0 < b.0 - 1e-9
                        || ((a.0 - b.0).abs() <= 1e-9
                            && (a.1 < b.1 - 1e-9
                                || ((a.1 - b.1).abs() <= 1e-9 && level < depth[s])))
                };
                if prefer
                    && !(required.is_some()
                        && introduces_cycle(s, &m.inputs, &chosen, instances, free))
                {
                    costs[s] = area;
                    arrival[s] = time;
                    depth[s] = level;
                    chosen[s] = Some(Binding { instance: i, pin });
                    for &consumer in &consumers[s] {
                        if !pending[consumer] {
                            pending[consumer] = true;
                            queue.push_back(consumer);
                        }
                    }
                }
            }
        }
        if Instant::now() >= deadline {
            return (chosen, costs);
        }
    }
    (chosen, costs)
}


#[cfg(test)]
thread_local!{static TEST_SKIP_SCORED_APPLICATION:std::cell::Cell<bool>=const{std::cell::Cell::new(false)};}

// Preserve only whole covers already validated by the existing score oracle.
// The area trajectory may use slack; its DA return cannot lose a faster legal
// point visited earlier. No candidate is replayed or evaluated a second time.
fn remember_feasible_trial(trial:&[Option<Binding>],value:(f64,f64),ps:&MapParams,saved:&mut Option<(Vec<Option<Binding>>,(f64,f64))>){
    let Some(limit)=ps.protected_area_limit else{return;};
    if value.0>limit || !value.0.is_finite() || !value.1.is_finite() || ps.max_delay.is_some_and(|d|value.1>d){return;}
    if saved.as_ref().is_none_or(|(_,v)|value.1<v.1-1e-6 || ((value.1-v.1).abs()<=1e-6 && value.0<v.0-1e-6)){*saved=Some((trial.to_vec(),value));}
}

// Score and apply actual-area substitutions under the caller's existing deadline.
fn slack_area_pass(instances:&[Instance],free:&[bool],roots:&[usize],witness:&[Option<Binding>],mut chosen:Vec<Option<Binding>>,mut best:(f64,f64),ps:&MapParams,stats:&mut MapStats,deadline:Instant,preserve_scored:bool)->(Vec<Option<Binding>>,(f64,f64)){
    let pass_start=Instant::now();
    let stage_bound=if ps.fixed_area_bound && ps.objective==MapObjective::AreaDelay {ps.max_delay.unwrap_or(best.1)} else {best.1};
    let mut stage_ps=ps.clone();if ps.fixed_area_bound && ps.objective==MapObjective::AreaDelay {stage_ps.max_delay=Some(stage_bound);}let ps=&stage_ps;
    let timing_bound=|current:f64|if ps.fixed_area_bound && ps.objective==MapObjective::AreaDelay {stage_bound}else{current};
    if ps.no_actual_area_tail {return (chosen,best);}
    let mut feasible_incumbent=None;remember_feasible_trial(&chosen,best,ps,&mut feasible_incumbent);
    // Retain ownership of one already validated cover. Do not change the working
    // trajectory, re-score a candidate or bypass the final global deadline check.
    let mut scored_incumbent:Option<(Vec<Option<Binding>>,(f64,f64))>=None;
    let mut scored_updates=0usize;
    let mut bound_skipped=0usize;
    let mut bound_refreshes=0usize;
    let mut delta_queries=0usize;let mut delta_skips=0usize;let mut timing_refreshes=0usize;
        // Try timing-feasible physical substitutions before global area flow.
        // Candidate timing is only a filter; actual complete-cover metrics decide.
        for _ in 0..ps.refine_rounds {
            if Instant::now() >= deadline {
                break;
            }
            stats.slack_rounds_started += 1;
            let (mut arrival, _) = critical_arrivals(instances, &chosen, free, roots);
            let witness_roots: Vec<_> = witness
                .iter()
                .enumerate()
                .filter_map(|(s, b)| b.map(|_| s))
                .collect();
            let (witness_arrival, _) = critical_arrivals(instances, &witness, free, &witness_roots);
            for s in 0..arrival.len() {
                if arrival[s].is_none() {
                    arrival[s] = witness_arrival[s];
                }
            }
            let mut required = required_times(instances, &chosen, free, roots, timing_bound(best.1));
            let mut online_improved=false;
            let mut online_batch:OnlineAreaBatch=None;
            let online_batch_size=(instances.len()/1024).clamp(1,128);
            let mut active:BTreeSet<_>=chosen.iter().flatten().map(|b|b.instance).collect();
            let mut bounds=if (ps.shared_area_bound || ps.delta_area_filter) && (!ps.online_area || ps.online_shared_bound || ps.delta_area_filter){bound_refreshes+=1;Some(shared_area_counters(instances,&chosen,roots))}else{None};
            let mut proposals = Vec::new();
            for (i, m) in instances.iter().enumerate() {
                if Instant::now() >= deadline {
                    break;
                }
                if ps.online_area && i%online_batch_size==0 {
                    let committed=commit_online_area(&mut online_batch,&mut chosen,&mut best,ps,stats);
                    online_improved|=committed;
                    if committed && ps.refresh_area_timing {
                        // The committed complete DAG, not the round's old snapshot,
                        // determines feasibility for all subsequent substitutions.
                        arrival=critical_arrivals(instances,&chosen,free,roots).0;
                        for s in 0..arrival.len(){if arrival[s].is_none(){arrival[s]=witness_arrival[s];}}
                        required=required_times(instances,&chosen,free,roots,timing_bound(best.1));
                        timing_refreshes+=1;
                    }
                    if committed && bounds.is_some(){
                        // All ref counts AND active physical-instance membership must
                        // describe this validated cover before another pruning query.
                        bounds=Some(shared_area_counters(instances,&chosen,roots));
                        active=chosen.iter().flatten().map(|b|b.instance).collect();
                        bound_refreshes+=1;
                    }
                }
                if m.inputs.iter().any(|&input| arrival[input].is_none()) {
                    continue;
                }
                let feasible = m.outputs.iter().any(|o| {
                    let time = m
                        .inputs
                        .iter()
                        .enumerate()
                        .filter(|(p, _)| o.has_arc(*p))
                        .map(|(p, &input)| arrival[input].unwrap() + f64::from(o.delays[p]))
                        .fold(0.0, f64::max);
                    o.targets.iter().any(|&s| {
                        chosen[s].is_some_and(|b| b.instance != i) && time <= required[s] + 1e-6
                    })
                });
                if !feasible {
                    continue;
                }
                if let Some((refs,users))=&bounds {
                    if ps.delta_area_filter {
                        delta_queries+=1;
                        if let Some(area)=selected_input_area(instances,&chosen,free,i,best.0,refs,users){
                            // Subtract a conservative summation envelope. A near
                            // tie falls back to the original complete oracle.
                            let margin=8.*f64::EPSILON*(instances.len()+chosen.len()+1)as f64*(best.0.abs()+area.abs()+1.);
                            if area-margin>=best.0-1e-6 {delta_skips+=1;continue;}
                        }
                    }
                    let targets:Vec<_>=m.outputs.iter().flat_map(|o|o.targets.iter().copied()).filter(|&s|chosen[s].is_some_and(|b|b.instance!=i)).collect::<BTreeSet<_>>().into_iter().collect();
                    let lower=shared_area_lower_bound(instances,&chosen,free,&targets,i,best.0,refs,users,&active);
                    if lower>=best.0-1e-6 {bound_skipped+=1;continue;}
                }
                stats.slack_local_scored += 1;
                if ps.fast_area_scan && instances.len()>16384 {
                // Cheap ordering only; do not spend the stage on full-cover scoring
                // before its first trial. Joint-pin/alias sharing is checked at acceptance.
                let affected:BTreeSet<_>=m.outputs.iter().flat_map(|o|&o.targets)
                    .filter_map(|&s|chosen[s]).map(|b|b.instance).filter(|&old|old!=i).collect();
                let removed:f64=affected.iter().map(|&old|f64::from(instances[old].area)).sum();
                let added=if active.contains(&i){0.0}else{f64::from(m.area)};
                proposals.push((added-removed,i));
                    continue;
                }
                let mut trial = chosen.clone();
                for (pin, o) in m.outputs.iter().enumerate() {
                    for &target in &o.targets {
                        if chosen[target].is_some() {
                            trial[target] = Some(Binding { instance: i, pin });
                        }
                    }
                }
                let value=match validate_score_trial(instances,&mut trial,witness,free,roots,ps.fused_cover_scoring){Ok(v)=>v,Err(_)=>continue};
                remember_feasible_trial(&trial,value,ps,&mut feasible_incumbent);
                if value.0 < best.0 - 1e-6 {
                    proposals.push((value.0, i));
                    // This trial already passed full prune/physical-metric scoring.
                    // Reuse it now; no replay, extra score or extra search window.
                    // Required/arrival labels remain filters only: every subsequent
                    // acceptance still checks current whole-cover delay and area.
                    if ps.online_area && value.1<=timing_bound(best.1)+1e-6 && better_cover(value,best,ps) {
                        let replace=online_batch.as_ref().is_none_or(|(_,v,area,index)|better_cover(value,*v,ps) || ((value.0-v.0).abs()<1e-6 && (value.1-v.1).abs()<1e-6 && m.area.total_cmp(area).then(i.cmp(index)).is_lt()));
                        if replace {online_batch=Some((trial,value,m.area,i));}
                        continue;
                    }
                    if preserve_scored && (!ps.fixed_area_bound || value.1<=timing_bound(best.1)+1e-6) && better_cover(value,best,ps)
                        && scored_incumbent.as_ref().is_none_or(|(_,saved)|better_cover(value,*saved,ps)) {
                        scored_incumbent=Some((trial,value));scored_updates+=1;
                    }
                }
            }
            if ps.online_area {online_improved|=commit_online_area(&mut online_batch,&mut chosen,&mut best,ps,stats);}
            proposals.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let mut improved = online_improved;
            for (_, i) in proposals.into_iter().take(ps.refine_trials) {
                #[cfg(test)]
                if TEST_SKIP_SCORED_APPLICATION.with(|v|v.get()){break;}
                if Instant::now() >= deadline {
                    break;
                }
                stats.slack_local_trials += 1;
                stats.refinement_trials += 1;
                let m = &instances[i];
                let mut trial = chosen.clone();
                for (pin, o) in m.outputs.iter().enumerate() {
                    for &target in &o.targets {
                        if chosen[target].is_some() {
                            trial[target] = Some(Binding { instance: i, pin });
                        }
                    }
                }
                let value=match validate_score_trial(instances,&mut trial,witness,free,roots,ps.fused_cover_scoring){Ok(v)=>v,Err(_)=>continue};
                remember_feasible_trial(&trial,value,ps,&mut feasible_incumbent);
                if (!ps.fixed_area_bound || value.1<=timing_bound(best.1)+1e-6) && (!ps.fast_area_scan || (value.0 < best.0 - 1e-6 && value.1 <= timing_bound(best.1) + 1e-6)) && better_cover(value, best, ps) {
                    chosen = trial;
                    best = value;
                    improved = true;
                    stats.refinement_accepted += 1;
                    stats.slack_local_accepted += 1;
                }
            }
            if !improved {
                break;
            }
        }
    let mut used=false;
    if let Some((candidate,value))=scored_incumbent {
        if better_cover(value,best,ps){chosen=candidate;best=value;used=true;stats.refinement_accepted+=1;}
    }
    if let Some((candidate,value))=feasible_incumbent {
        let invalid=ps.protected_area_limit.is_some_and(|a|best.0>a) || ps.max_delay.is_some_and(|d|best.1>d);
        let restore=invalid || value.1<best.1-1e-6 || ((value.1-best.1).abs()<=1e-6 && value.0<best.0-1e-6);
        if restore{chosen=candidate;best=value;}
        recovery_event("feasible_visited_incumbent",pass_start,serde_json::json!({"restored":restore,"area":best.0,"delay":best.1,"area_limit":ps.protected_area_limit,"fixed_delay_limit":ps.max_delay,"additional_evaluations":0,"whole_cover_only":true}));
    }
    if preserve_scored {recovery_event("scored_incumbent",pass_start,serde_json::json!({"updates":scored_updates,"used":used,"area":best.0,"delay":best.1,"additional_evaluations":0}));}
    recovery_event("shared_area_bound",pass_start,serde_json::json!({"enabled":ps.shared_area_bound&&(!ps.online_area||ps.online_shared_bound),"online_enabled":ps.online_shared_bound,"refreshes":bound_refreshes,"skipped":bound_skipped,"scope":"optimistic reclaimable-area bound; unchanged complete trial metrics/order for potential improvements; no extra trial budget"}));
    recovery_event("delta_area_filter",pass_start,serde_json::json!({"enabled":ps.delta_area_filter,"queries":delta_queries,"skipped":delta_skips,"acceptance":"unchanged full-cover cycle/prune/physical area/pin delay oracle"}));
    recovery_event("area_timing_refresh",pass_start,serde_json::json!({"enabled":ps.refresh_area_timing,"refreshes":timing_refreshes,"deadline_extended":false}));
    recovery_event("online_area",pass_start,serde_json::json!({"enabled":ps.online_area,"updates":stats.online_area_updates,"extra_evaluations":0,"area":best.0,"delay":best.1}));
    (chosen,best)
}


fn diagnostic_producer_index(instances:&[Instance],signals:usize)->Vec<Vec<(usize,usize)>>{
    let mut index=vec![Vec::new();signals];
    for (i,m) in instances.iter().enumerate(){for (pin,o) in m.outputs.iter().enumerate(){
        for &s in o.targets.iter().collect::<BTreeSet<_>>(){if let Some(row)=index.get_mut(s){row.push((i,pin));}}
    }}index
}
fn diagnostic_cell_names(tech:&TechLib,k:usize,function:u64)->BTreeSet<String>{
    (0..2).flat_map(|phase|tech.candidates(k as u8,tt9_from_u64(function ^ if phase==1{mask(k)}else{0})).into_iter().flatten().filter(|v|!v.out_neg).map(|v|v.cell_name.clone())).collect()
}

fn physical_refs(instances: &[Instance], chosen: &[Option<Binding>], roots: &[usize]) -> Vec<f64> {
    let mut refs = vec![0.0f64; chosen.len()];
    for &s in roots.iter().collect::<BTreeSet<_>>() {
        refs[s] += 1.0;
    }
    let live: BTreeSet<_> = chosen.iter().flatten().map(|b| b.instance).collect();
    for i in live {
        for &s in instances[i].inputs.iter().collect::<BTreeSet<_>>() {
            refs[s] += 1.0;
        }
    }
    for count in &mut refs {
        *count = count.max(1.0);
    }
    refs
}

// A finite acyclic seed from a physical readiness frontier. Nondependent pins
// use a conservative readiness bound, NOT a minimum loaded/unloaded-delay claim.
#[derive(Clone,Copy,Debug)]struct DelayReady(f64,f64,usize,usize,usize);
impl PartialEq for DelayReady{fn eq(&self,b:&Self)->bool{self.cmp(b).is_eq()}}
impl Eq for DelayReady{}
impl PartialOrd for DelayReady{fn partial_cmp(&self,b:&Self)->Option<std::cmp::Ordering>{Some(self.cmp(b))}}
impl Ord for DelayReady{fn cmp(&self,b:&Self)->std::cmp::Ordering{b.0.total_cmp(&self.0).then(b.1.total_cmp(&self.1)).then(b.2.cmp(&self.2)).then(b.3.cmp(&self.3)).then(b.4.cmp(&self.4))}}
fn delay_frontier_seed(instances:&[Instance],free:&[bool],deadline:Instant)->Option<(Vec<Option<Binding>>,usize)>{
    let mut known=free.to_vec();let mut chosen=vec![None;free.len()];let mut bound=vec![0.;free.len()];let mut cost=vec![0.;free.len()];
    let mut edges=Vec::new();let mut remaining=Vec::new();let mut users=vec![Vec::new();free.len()];let mut queue=std::collections::BinaryHeap::new();let mut conservative=0;
    fn ready(i:usize,pin:usize,m:&Instance,known:&[bool],bound:&[f64],cost:&[f64],queue:&mut std::collections::BinaryHeap<DelayReady>){
        let o=&m.outputs[pin];let time=m.inputs.iter().enumerate().map(|(p,&x)|bound[x]+if o.has_arc(p){f64::from(o.delays[p])}else{0.}).fold(0.,f64::max);
        let area=(f64::from(m.area)+unique_input_cost(&m.inputs,cost)).min(f64::MAX);
        for &target in &o.targets{if !known[target]{queue.push(DelayReady(time,area,target,i,pin));}}
    }
    for (i,m) in instances.iter().enumerate(){
        if i%256==0 && Instant::now()>=deadline{return None}
        let inputs:BTreeSet<_>=m.inputs.iter().copied().collect();
        for (pin,o) in m.outputs.iter().enumerate(){if o.targets.is_empty(){continue}if (0..m.inputs.len()).any(|p|!o.has_arc(p)){conservative+=1;}
            let edge=edges.len();edges.push((i,pin));let mut wait=0;
            for &x in &inputs{if !known[x]{wait+=1;users[x].push(edge);}}
            remaining.push(wait);if wait==0{ready(i,pin,m,&known,&bound,&cost,&mut queue);}
        }
    }
    let mut visits=0;
    while let Some(DelayReady(time,area,target,i,pin))=queue.pop(){
        visits+=1;if visits%256==0 && Instant::now()>=deadline{return None}
        if known[target]{continue}known[target]=true;bound[target]=time;cost[target]=area;chosen[target]=Some(Binding{instance:i,pin});
        for &edge in &users[target]{remaining[edge]-=1;if remaining[edge]==0{let (i,p)=edges[edge];ready(i,p,&instances[i],&known,&bound,&cost,&mut queue);}}
    }
    if Instant::now()>=deadline{return None}Some((chosen,conservative))
}
// One constrained area-flow sweep in an explicitly acyclic physical order.
// Area labels are proxies; only a validated full-cover lex improvement commits.
fn required_area_dag(instances:&[Instance],free:&[bool],roots:&[usize],incumbent:&[Option<Binding>],witness:&[Option<Binding>],deadline:Instant)->Option<Vec<Option<Binding>>>{
    let mut base=witness.to_vec();for (i,b) in incumbent.iter().enumerate(){if b.is_some(){base[i]=*b;}}
    let all:Vec<_>=base.iter().enumerate().filter_map(|(s,b)|b.map(|_|s)).collect();if prune(instances,&mut base,free,&all).is_err(){return None}
    let bound=metrics(instances,incumbent,free,roots).1;let required=required_times(instances,incumbent,free,roots,bound);let refs=physical_refs(instances,incumbent,roots);
    let mut users=vec![Vec::new();free.len()];let mut wait=vec![0;free.len()];let mut ready=std::collections::VecDeque::new();
    for (target,b) in base.iter().enumerate(){if free[target]{continue}if let Some(b)=b{let input:BTreeSet<_>=instances[b.instance].inputs.iter().copied().filter(|&x|!free[x]).collect();wait[target]=input.len();if input.is_empty(){ready.push_back(target)}for x in input{users[x].push(target);}}}
    let mut producer=vec![Vec::new();free.len()];for (i,m) in instances.iter().enumerate(){if i%256==0 && Instant::now()>=deadline{return None}for (pin,o) in m.outputs.iter().enumerate(){for &target in &o.targets{if !free[target] && base[target].is_some(){producer[target].push(Binding{instance:i,pin});}}}}
    let mut chosen=vec![None;free.len()];let mut done=free.to_vec();let mut arrival=vec![0.;free.len()];let mut cost=vec![0.;free.len()];let mut count=0;
    while let Some(target)=ready.pop_front(){if count%128==0 && Instant::now()>=deadline{return None}count+=1;let mut best:Option<(f64,f64,Binding)>=None;
        for &b in &producer[target]{let m=&instances[b.instance];if !m.inputs.iter().all(|&x|done[x]){continue}let o=&m.outputs[b.pin];let time=m.inputs.iter().enumerate().filter(|(p,_)|o.has_arc(*p)).map(|(p,&x)|arrival[x]+f64::from(o.delays[p])).fold(0.,f64::max);if time>required[target]+1e-7{continue}
            let area=(f64::from(m.area)+unique_input_cost(&m.inputs,&cost)).min(f64::MAX)/refs[target].max(1.0);
            if best.as_ref().is_none_or(|old|area.total_cmp(&old.0).then(time.total_cmp(&old.1)).then(b.instance.cmp(&old.2.instance)).then(b.pin.cmp(&old.2.pin)).is_lt()){best=Some((area,time,b));}}
        let (area,time,b)=best?;chosen[target]=Some(b);cost[target]=area;arrival[target]=time;done[target]=true;
        for &user in &users[target]{wait[user]-=1;if wait[user]==0{ready.push_back(user);}}
    }
    if all.iter().any(|&s|!done[s])||Instant::now()>=deadline{return None}Some(chosen)
}
// Isolated recovery scheduling experiment; trace writing is diagnostic reporting.
thread_local! { static RECOVERY_EVENTS: std::cell::RefCell<Vec<serde_json::Value>> = const { std::cell::RefCell::new(Vec::new()) }; }
fn recovery_event(name:&str,start:Instant,data:serde_json::Value){
    if std::env::var_os("EGM_RECOVERY_TRACE").is_some(){RECOVERY_EVENTS.with(|events|events.borrow_mut().push(serde_json::json!({"stage":name,"seconds":start.elapsed().as_secs_f64(),"data":data})));}
}
struct RecoveryTrace;
impl Drop for RecoveryTrace{fn drop(&mut self){if let Some(path)=std::env::var_os("EGM_RECOVERY_TRACE"){RECOVERY_EVENTS.with(|events|{let value=serde_json::json!({"events":*events.borrow(),"scope":"one global recovery window; report serialization excluded; nonpreemptive stage checks"});if let Ok(text)=serde_json::to_vec_pretty(&value){let _=std::fs::write(path,text);}});}}}
#[derive(Debug,Clone,Copy)] struct RecoveryProposal(f64,f64,usize);
impl PartialEq for RecoveryProposal{fn eq(&self,b:&Self)->bool{self.cmp(b).is_eq()}}
impl Eq for RecoveryProposal{}
impl PartialOrd for RecoveryProposal{fn partial_cmp(&self,b:&Self)->Option<std::cmp::Ordering>{Some(self.cmp(b))}}
impl Ord for RecoveryProposal{fn cmp(&self,b:&Self)->std::cmp::Ordering{self.0.total_cmp(&b.0).then(self.1.total_cmp(&b.1)).then(self.2.cmp(&b.2))}}
fn best_recovery_proposals(proposals:impl Iterator<Item=(f64,f64,usize)>,limit:usize)->Vec<(f64,f64,usize)>{
    let mut heap=std::collections::BinaryHeap::new();
    for (timing,area,i) in proposals{let p=RecoveryProposal(timing,area,i);if limit==0{break}if heap.len()<limit{heap.push(p)}else if heap.peek().is_some_and(|worst|p<*worst){heap.pop();heap.push(p);}}
    heap.into_sorted_vec().into_iter().map(|p|(p.0,p.1,p.2)).collect()
}
fn heuristic_cover(
    instances: &[Instance],
    free: &[bool],
    roots: &[usize],
    ps: &MapParams,
    stats: &mut MapStats,
) -> Result<Vec<Option<Binding>>, String> {
    let _trace=RecoveryTrace; RECOVERY_EVENTS.with(|x|x.borrow_mut().clear());
    let start = Instant::now();
    let _metric_epochs=MetricEpochGuard::enter(ps.metric_epochs);
    let deadline = start + std::time::Duration::from_secs_f64(ps.refine_time_secs);
    // Reserve time within the original budget for final incumbent accounting.
    let search_deadline = if ps.strict_budgets {
        deadline - std::time::Duration::from_secs_f64(ps.refine_time_secs.min(ps.return_reserve))
    } else {
        deadline
    };
    let mut chosen = heuristic_cover_until(instances, free, roots, ps, stats, search_deadline)?;
    if ps.flow_compare && !ps.flow_worklist && ps.refine_rounds > 0 && Instant::now() < deadline {
        stats.flow_compare_attempted = true;
        let mut alternative_params = ps.clone();
        alternative_params.flow_worklist = true;
        let mut alternative_stats = MapStats::default();
        if let Ok(alternative) = heuristic_cover_until(
            instances,
            free,
            roots,
            &alternative_params,
            &mut alternative_stats,
            deadline,
        ) {
            if better_cover(
                metrics(instances, &alternative, free, roots),
                metrics(instances, &chosen, free, roots),
                ps,
            ) {
                chosen = alternative;
                stats.flow_compare_accepted = true;
            }
        }
        stats.refinement_trials += alternative_stats.refinement_trials;
        stats.refinement_accepted += alternative_stats.refinement_accepted;
    }
    let validate_start=Instant::now();
    stats.heuristic_area = metrics(instances, &chosen, free, roots).0;
    recovery_event("final_validation",validate_start,serde_json::json!({}));
    stats.heuristic_secs = start.elapsed().as_secs_f64();
    recovery_event("global_total",start,serde_json::json!({"budget":ps.refine_time_secs}));
    if ps.strict_budgets && Instant::now() > deadline {
        return Err(format!(
            "budget: recovery {:.6}s exceeds {:.6}s including setup and final validation",
            stats.heuristic_secs, ps.refine_time_secs
        ));
    }
    Ok(chosen)
}

// Candidates here have already passed complete prune/metrics in this immutable
// physical pool. Save ownership of the entire predecessor, not flow labels.
fn activate_saved_tail(chosen:&mut Vec<Option<Binding>>,best:&mut(f64,f64),candidate:Option<&(Vec<Option<Binding>>,(f64,f64))>,ps:&MapParams,deadline:Instant)->Option<(Vec<Option<Binding>>,(f64,f64))>{
    if ps.objective!=MapObjective::DelayArea || Instant::now()>=deadline{return None;}
    let(cover,value)=candidate?;
    if value.0.is_finite() && value.1.is_finite() && value.1<=best.1 && better_cover(*value,*best,ps){let saved=Some((std::mem::replace(chosen,cover.clone()),*best));*best=*value;return saved;}
    None
}

fn heuristic_cover_until(
    instances: &[Instance],
    free: &[bool],
    roots: &[usize],
    ps: &MapParams,
    stats: &mut MapStats,
    deadline: Instant,
) -> Result<Vec<Option<Binding>>, String> {
    let start = Instant::now();
    let seed_fraction=ps.seed_window_fraction;
    if !seed_fraction.is_finite() || !(0.0..=1.0).contains(&seed_fraction){return Err("invalid seed window fraction".into())}
    let bounded=ps.bounded_proposals;
    let area_dag=ps.required_area_dag;
    let defer_area=ps.defer_area_dag;
    let mut deferred_area=None;
    let mut legacy_area_backup=None;
    let mut witness = if ps.strict_budgets {
        finite_choices_queued(instances, free)
    } else {
        finite_choices(instances, free)
    };
    let mut chosen = witness.clone();
    prune(instances, &mut chosen, free, roots)?;
    let mut best = metrics(instances, &chosen, free, roots);
    stats.initial_area = best.0;
    stats.initial_delay = best.1;
    stats.seed_delay = best.1;
    recovery_event("finite_witness_and_initial_validation",start,serde_json::json!({"area":best.0,"delay":best.1}));
    let preserve_legacy=ps.preserve_legacy_seeds && ps.objective==MapObjective::DelayArea;
    let legacy_state=preserve_legacy.then(||(chosen.clone(),best,witness.clone()));
    let mut frontier_state=None;
    let mut frontier_active=false;
    if ps.fast_delay_seed{
        let frontier_start=Instant::now();let mut result="deadline";let mut conservative=0;
        if let Some((full,c))=delay_frontier_seed(instances,free,deadline){
            conservative=c;let mut checked=full.clone();let all:Vec<_>=full.iter().enumerate().filter_map(|(s,b)|b.map(|_|s)).collect();
            if prune(instances,&mut checked,free,&all).is_ok(){
                let mut cover=full.clone();if prune(instances,&mut cover,free,roots).is_ok(){let value=metrics(instances,&cover,free,roots);if better_cover(value,best,ps){chosen=cover;best=value;}stats.seed_delay=stats.seed_delay.min(value.1);witness=full;frontier_active=true;result="validated";}else{result="PO validation failed";}
            }else{result="all-signal validation failed";}
        }
        recovery_event("fast_delay_frontier",frontier_start,serde_json::json!({"result":result,"conservative_nonarc_edges":conservative,"best_delay":best.1,"best_area":best.0}));
    }
    if let Some((cover,value,full))=legacy_state {
        if frontier_active { frontier_state=Some((chosen.clone(),best,witness.clone())); }
        chosen=cover;best=value;witness=full;
    }
    let fanout_start=Instant::now();
    if ps.refine_rounds == 0
        || ps.refine_time_secs == 0.0
        || (ps.strict_budgets && Instant::now() >= deadline)
    {
        stats.heuristic_area = best.0;
        stats.heuristic_secs = start.elapsed().as_secs_f64();
        return Ok(chosen);
    }
    let mut refs = vec![1.0; free.len()];
    let mut edges = HashSet::new();
    for m in instances {
        for output in &m.outputs {
            for &s in &output.targets {
                for &input in &m.inputs {
                    edges.insert((input, s));
                }
            }
        }
    }
    for (input, _) in edges {
        refs[input] += 1.0;
    }
    let mut tree_cost = vec![0.0; free.len()];
    let mut seeds = vec![(vec![1.0; free.len()], false), (refs, false)];
    if ps.structural_refs && (preserve_legacy || !(frontier_active && area_dag && ps.objective==MapObjective::DelayArea)) {
        seeds.push((physical_refs(instances, &chosen, roots), true));
    }
    recovery_event("fanout_and_seed_setup",fanout_start,serde_json::json!({"seed_count":seeds.len()}));
    let seed_deadline=regions::phase_deadline(Instant::now(),deadline,if preserve_legacy {seed_fraction.min(0.25)} else {seed_fraction});
    for (seed_index,(divisor, divide_inputs)) in seeds.into_iter().enumerate() {
        if Instant::now() >= seed_deadline {
            break;
        }
        let seed_start=Instant::now();
        let flow = if ps.flow_worklist {
            flow_choices_worklist
        } else {
            flow_choices
        };
        let (candidate,costs)=if frontier_active && ps.flow_worklist && !preserve_legacy{
            flow_choices_worklist_impl(
            instances,
            free,
            &divisor,
            if ps.objective == MapObjective::Weighted {
                ps.delay_weight
            } else {
                0.0
            },
            divide_inputs,
            ps.flow_delay_ties,
            // Area-flow ties can change physical sharing. Keep the legacy seed for
            // area-first; complete-cover acceptance still breaks area ties by delay.
            if ps.objective == MapObjective::AreaDelay {
                MapObjective::Weighted
            } else {
                ps.objective
            },
            seed_deadline,
            None,Some(&witness),
            )
        }else{flow(
            instances,
            free,
            &divisor,
            if ps.objective == MapObjective::Weighted {
                ps.delay_weight
            } else {
                0.0
            },
            divide_inputs,
            ps.flow_delay_ties,
            // Area-flow ties can change physical sharing. Keep the legacy seed for
            // area-first; complete-cover acceptance still breaks area ties by delay.
            if ps.objective == MapObjective::AreaDelay {
                MapObjective::Weighted
            } else {
                ps.objective
            },
            seed_deadline,
        )};
        // A cycle in an unused alternative must not reject an otherwise valid PO cover.
        let mut cover = candidate.clone();
        if fill_from_witness(instances, &mut cover, &witness, free, roots).is_ok() {
            let value = metrics(instances, &cover, free, roots);
            stats.seed_delay = stats.seed_delay.min(value.1);
            if better_cover(value, best, ps) {
                chosen = cover;
                best = value;
            }
        }
        recovery_event("flow_seed_and_PO_validation",seed_start,serde_json::json!({"seed_index":seed_index,"best_area":best.0,"best_delay":best.1,"candidate_slots":candidate.iter().filter(|x|x.is_some()).count()}));
        let repair_start=Instant::now();
        // The repair witness needs the stricter all-signal check because later replacements can
        // introduce dependencies outside the current PO cone. Keep the old witness if this fails.
        let mut full = witness.clone();
        for (s, b) in candidate.iter().enumerate() {
            if b.is_some() {
                full[s] = *b;
            }
        }
        let all: Vec<_> = full
            .iter()
            .enumerate()
            .filter(|(_, b)| b.is_some())
            .map(|(s, _)| s)
            .collect();
        let repair=prune(instances, &mut full, free, &all);
        recovery_event("seed_repair_witness_validation",repair_start,serde_json::json!({"seed_index":seed_index,"valid":repair.is_ok()}));
        if repair.is_err(){continue;}
        witness = full;
        tree_cost = costs;

    }
    if let Some((cover,value,_))=&frontier_state {
        deferred_area=Some((cover.clone(),*value));
        recovery_event("protected_legacy_seed",Instant::now(),serde_json::json!({"legacy_area":best.0,"legacy_delay":best.1,"frontier_area":value.0,"frontier_delay":value.1,"legacy_seed_window_fraction":0.25}));
    }
    if frontier_active && area_dag && ps.objective==MapObjective::DelayArea && Instant::now()<deadline{
        let area_start=Instant::now();let stage_deadline=regions::phase_deadline(area_start,deadline,0.35);let mut status="no finite timed sweep";
        if let Some(mut trial)=required_area_dag(instances,free,roots,frontier_state.as_ref().map_or(&chosen,|(c,_,_)|c),frontier_state.as_ref().map_or(&witness,|(_,_,w)|w),stage_deadline){
            if prune(instances,&mut trial,free,roots).is_ok(){let value=metrics(instances,&trial,free,roots);status="validated unchanged";if better_cover(value,best,ps) && Instant::now()<deadline{if defer_area { if deferred_area.as_ref().is_none_or(|(_,v)|better_cover(value,*v,ps)){deferred_area=Some((trial,value));}status="validated deferred incumbent"; } else { chosen=trial;best=value;status="accepted"; }}}
        }
        recovery_event("required_area_dag",area_start,serde_json::json!({"status":status,"best_area":best.0,"best_delay":best.1,"time_fraction_of_remaining":0.35}));
    }
    if ps.activate_deferred && ps.objective==MapObjective::DelayArea {
        let activation_start=Instant::now();let before=best;let mut status="no strictly faster deferred cover";
        if let Some((trial,value))=&deferred_area {
            if value.1+1e-7<best.1 && Instant::now()<deadline {
                // Timed DAG was built in this frontier witness order. Never graft
                // a PO-only cover into an unchecked all-signal repair graph.
                let mut full=frontier_state.as_ref().map_or(&witness,|(_,_,w)|w).clone();
                for (s,b) in trial.iter().enumerate(){if b.is_some(){full[s]=*b;}}
                let all:Vec<_>=full.iter().enumerate().filter_map(|(s,b)|b.map(|_|s)).collect();
                if prune(instances,&mut full,free,&all).is_ok() && Instant::now()<deadline {
                    legacy_area_backup=Some((std::mem::replace(&mut chosen,trial.clone()),best));
                    best=*value;witness=full;status="activated validated faster reference";
                } else {status="repair witness invalid or deadline";}
            }
        }
        recovery_event("deferred_working_reference",activation_start,serde_json::json!({"status":status,"before_area":before.0,"before_delay":before.1,"working_area":best.0,"working_delay":best.1}));
    }
    let area_params=shared_area_params(ps,legacy_area_backup.is_some());
    let ps=&area_params;
    #[cfg(test)] TEST_LOCAL_START.with(|v|v.set(Some(best)));
    let joint_reference=if ps.joint_cones {frontier_state.as_ref().map(|(_,_,w)|w.as_slice())}else{None};
    let local_deadline=if ps.slack_flow && ps.objective==MapObjective::DelayArea {
        regions::phase_deadline(Instant::now(),deadline,1.0-if ps.online_area && legacy_area_backup.is_some(){ps.area_time_fraction.max(0.25)}else{ps.area_time_fraction})
    } else {deadline};
    if ps.phase_recovery {
        let phase_start = Instant::now();
        let phase_local_deadline = regions::phase_deadline(phase_start, local_deadline, ps.phase_time_fraction);
        regions::recover_phases(
            instances,
            &mut chosen,
            &witness,
            joint_reference,
            free,
            roots,
            ps,
            stats,
            phase_local_deadline,
        )?;
        stats.phase_secs = phase_start.elapsed().as_secs_f64();
        recovery_event("phase",phase_start,serde_json::json!({"trials":stats.phase_trials,"accepted":stats.phase_accepted}));
        best = metrics(instances, &chosen, free, roots);
    }
    for cone_phase in [false, true] {
        if cone_phase && !ps.recover_cones {
            continue;
        }
        // Preserve the original local recovery first; the cone-ranked pass can only improve it.
        for _ in 0..if cone_phase { 1 } else { ps.refine_rounds } {
            if Instant::now() >= local_deadline {
                break;
            }
            let rank_setup_start=Instant::now();
            let mut output_users = vec![0; instances.len()];
            let mut signal_refs = vec![0; free.len()];
            if cone_phase {
                for b in chosen.iter().flatten() {
                    output_users[b.instance] += 1;
                }
                for &s in roots.iter().collect::<BTreeSet<_>>() {
                    signal_refs[s] += 1;
                }
                for (i, m) in instances.iter().enumerate() {
                    if output_users[i] > 0 {
                        for &s in m.inputs.iter().collect::<BTreeSet<_>>() {
                            signal_refs[s] += 1;
                        }
                    }
                }
            }
            let timing = if ps.critical_recovery
                && (ps.objective == MapObjective::DelayArea
                    || ps.max_delay.is_some_and(|limit| best.1 > limit + 1e-6))
            {
                Some(critical_arrivals(instances, &chosen, free, roots))
            } else {
                None
            };
            recovery_event("critical_and_cone_rank_setup",rank_setup_start,serde_json::json!({"cone":cone_phase}));
            let prepare_start=Instant::now();
            let proposal_iter = instances
                .iter()
                .enumerate()
                .take_while(|_|!bounded || Instant::now()<local_deadline)
                .filter_map(|(i, m)| {
                    if (cone_phase || bounded) && Instant::now() >= local_deadline {
                        return None;
                    }
                    let affected: BTreeSet<_> = m
                        .outputs
                        .iter()
                        .flat_map(|o| &o.targets)
                        .filter_map(|&s| chosen[s].map(|b| b.instance))
                        .collect();
                    if affected.is_empty() {
                        return None;
                    }
                    let removed: f64 = if cone_phase {
                        let targets: Vec<_> = m
                            .outputs
                            .iter()
                            .flat_map(|o| &o.targets)
                            .filter(|&&s| chosen[s].is_some_and(|b| b.instance != i))
                            .copied()
                            .collect();
                        reclaimable_area(
                            &targets,
                            instances,
                            &chosen,
                            free,
                            &signal_refs,
                            &output_users,
                        )
                    } else {
                        affected
                            .into_iter()
                            .map(|old| f64::from(instances[old].area))
                            .sum()
                    };
                    let added: f64 = m
                        .inputs
                        .iter()
                        .filter(|&&s| !free[s] && chosen[s].is_none())
                        .map(|&s| tree_cost[s])
                        .sum();
                    let timing_gain = timing.as_ref().map_or(0.0, |(arrival, critical)| {
                        critical_gain(m, &chosen, arrival, critical)
                    });
                    Some((timing_gain, f64::from(m.area) + added - removed, i))
                });
            let mut proposals:Vec<_>=if bounded {best_recovery_proposals(proposal_iter,ps.refine_trials)}else{proposal_iter.collect()};
            if !bounded {proposals.sort_by(|a,b|a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)).then(a.2.cmp(&b.2)));}
            recovery_event("proposal_prepare_and_order",prepare_start,serde_json::json!({"cone":cone_phase,"bounded_topk":bounded,"kept":proposals.len(),"critical_first":ps.critical_recovery}));
            let mut improved = false;
            let before_trials = stats.refinement_trials;
            let trial_start=Instant::now();let before_accepted=stats.refinement_accepted;
            'proposals: for (_, _, i) in proposals.into_iter().take(ps.refine_trials) {
                if Instant::now() >= local_deadline {
                    break;
                }
                let m = &instances[i];
                let modes: Vec<_> = std::iter::once(None)
                    .chain(
                        (0..m.outputs.len())
                            .filter(|_| m.outputs.len() > 1)
                            .map(Some),
                    )
                    .collect();
                for mode in modes {
                    if Instant::now() >= local_deadline
                        || stats.refinement_trials - before_trials >= ps.refine_trials
                    {
                        break 'proposals;
                    }
                    let mut trial = chosen.clone();
                    let mut changed = false;
                    for (pin, output) in m.outputs.iter().enumerate() {
                        if mode.is_some_and(|p| p != pin) {
                            continue;
                        }
                        for &s in &output.targets {
                            if chosen[s].is_some()
                                && !chosen[s].is_some_and(|b| b.instance == i && b.pin == pin)
                            {
                                trial[s] = Some(Binding { instance: i, pin });
                                changed = true;
                            }
                        }
                    }
                    if !changed {
                        continue;
                    }
                    stats.refinement_trials += 1;
                    if joint_input_cone(instances, &mut trial, &witness, free, roots,joint_reference,&chosen,stats,local_deadline).is_err() {
                        continue;
                    }
                    let value = metrics(instances, &trial, free, roots);
                    if better_cover(value, best, ps) {
                        chosen = trial;
                        best = value;
                        improved = true;
                        stats.refinement_accepted += 1;
                    }
                }
            }
            recovery_event("local_trials",trial_start,serde_json::json!({"cone":cone_phase,"trials":stats.refinement_trials-before_trials,"accepted":stats.refinement_accepted-before_accepted,"best_area":best.0,"best_delay":best.1}));
            if !improved {
                break;
            }
        }
    }
    recovery_event("joint_repair",Instant::now(),serde_json::json!({"attempts":stats.joint_attempts,"valid":stats.joint_valid,"extra_physical_candidates":0,"new_deadline":false}));
    let tail_start=Instant::now();
    let mut tail_backup=None;
    let tail_params=ps.clone();
    let ps=&tail_params;
    // Re-estimate fanouts from the physical incumbent, rather than all alternative candidates.
    // Divide the entire cone flow by its observed reference count. Keep the previous cover unless
    // the rebuilt finite DAG improves the actual objective (including a hard delay constraint).
    for _ in 0..if ps.recover_live_refs {
        ps.refine_rounds
    } else {
        0
    } {
        if Instant::now() >= local_deadline {
            break;
        }
        let live_refs = physical_refs(instances, &chosen, roots);
        let flow = if ps.flow_worklist {
            flow_choices_worklist
        } else {
            flow_choices
        };
        let (mut candidate, _) = flow(
            instances,
            free,
            &live_refs,
            if ps.objective == MapObjective::Weighted {
                ps.delay_weight
            } else {
                0.0
            },
            true,
            ps.flow_delay_ties,
            // Area-flow ties can change physical sharing. Keep the legacy seed for
            // area-first; complete-cover acceptance still breaks area ties by delay.
            if ps.objective == MapObjective::AreaDelay {
                MapObjective::Weighted
            } else {
                ps.objective
            },
            local_deadline,
        );
        if fill_from_witness(instances, &mut candidate, &witness, free, roots).is_err() {
            break;
        }
        let value = metrics(instances, &candidate, free, roots);
        if !better_cover(value, best, ps) {
            break;
        }
        chosen = candidate;
        best = value;
        stats.refinement_accepted += 1;
    }
    let flow_deadline=if ps.fast_area_scan || ps.online_area {
        regions::phase_deadline(Instant::now(),deadline,0.30)
    } else {deadline};
    if ps.objective == MapObjective::DelayArea && ps.slack_flow {
        for _ in 0..ps.refine_rounds {
            if Instant::now() >= flow_deadline {
                break;
            }
            let required = required_times(instances, &chosen, free, roots, best.1);
            let refs = physical_refs(instances, &chosen, roots);
            let (mut trial, _) = flow_choices_worklist_impl(
                instances,
                free,
                &refs,
                0.0,
                true,
                true,
                MapObjective::AreaDelay,
                flow_deadline,
                Some(&required),
                Some(&chosen),
            );
            stats.slack_attempts += 1;
            if fill_from_witness(instances, &mut trial, &witness, free, roots).is_err() {
                break;
            }
            stats.slack_valid += 1;
            let value = metrics(instances, &trial, free, roots);
            if !better_cover(value, best, ps) {
                break;
            }
            chosen = trial;
            best = value;
            stats.refinement_accepted += 1;
            stats.slack_accepted += 1;
        }
        // Preserve the existing bounded flow trajectory before activating any
        // deferred reference. No extra flow pass, pool or recovery window.
        let mut actual_params=ps.clone();
        if ps.tail_shared_area && ps.objective==MapObjective::DelayArea {
            tail_backup=activate_saved_tail(&mut chosen,&mut best,deferred_area.as_ref(),ps,deadline);
            if tail_backup.is_some() {
                actual_params.online_area=true;
                actual_params.max_delay=Some(actual_params.max_delay.map_or(best.1,|d|d.min(best.1)));
            }
            recovery_event("tail_shared_reference",tail_start,serde_json::json!({"saved_predecessor":tail_backup.is_some(),"area":best.0,"delay":best.1,"after_original_flow":true,"policy_changed":tail_backup.is_some(),"new_deadline":false}));
        }
        let ps=&actual_params;
        let preserve_scored=ps.scored_incumbent;
        (chosen,best)=slack_area_pass(instances,free,roots,&witness,chosen,best,ps,stats,deadline,preserve_scored);
    }
    // Both covers have already passed whole-cover validation in this immutable pool.
    // Delaying this choice preserves the original phase/local trajectory; no new
    // search, additional iteration/trial allowance or extra recovery deadline.
    if let Some((trial,value))=deferred_area {
        let accepted=better_cover(value,best,ps);
        if accepted {chosen=trial;best=value;}
        recovery_event("deferred_incumbent_selection",Instant::now(),serde_json::json!({"accepted":accepted,"area":best.0,"delay":best.1}));
    }
    if let Some((cover,value))=legacy_area_backup {if better_cover(value,best,ps){chosen=cover;best=value;}}
    if let Some((cover,value))=tail_backup {if better_cover(value,best,ps){chosen=cover;best=value;}}
    recovery_event("live_refs_and_slack",tail_start,serde_json::json!({"slack_attempts":stats.slack_attempts}));
    if ps.terminal_area {chosen=terminal_area_continuation(instances,free,roots,&witness,chosen,ps,stats,deadline);best=metrics(instances,&chosen,free,roots);}
    stats.heuristic_area = best.0;
    stats.heuristic_secs = start.elapsed().as_secs_f64();
    Ok(chosen)
}

// Called only after every legacy/deferred final-cover comparison in this pool.
// No flow rerun or independent budget: old actual-area rounds remain charged.
fn terminal_area_continuation(instances:&[Instance],free:&[bool],roots:&[usize],witness:&[Option<Binding>],chosen:Vec<Option<Binding>>,ps:&MapParams,stats:&mut MapStats,deadline:Instant)->Vec<Option<Binding>> {
    let start=Instant::now();
    let remaining=ps.refine_rounds.saturating_sub(stats.slack_rounds_started);
    if !ps.terminal_area || ps.objective!=MapObjective::DelayArea || ps.refine_trials==0 || remaining==0 || start>=deadline {return chosen;}
    let before=metrics(instances,&chosen,free,roots);
    if Instant::now()>=deadline {return chosen;}
    let old_rounds=stats.slack_rounds_started;let old_trials=stats.slack_local_trials;
    let mut recovery=ps.clone();recovery.terminal_area=false;recovery.objective=MapObjective::AreaDelay;
    recovery.refine_rounds=remaining;recovery.max_delay=Some(ps.max_delay.map_or(before.1,|d|d.min(before.1)));
    recovery.refresh_area_timing=true;recovery.online_area=true;
    stats.terminal_area_attempted=true;
    let(mut trial,_)=slack_area_pass(instances,free,roots,witness,chosen.clone(),before,&recovery,stats,deadline,true);
    let mut accepted=false;let mut after=before;
    // Revalidate the returned complete physical DAG, rather than trusting proxy
    // flow values or an intermediate score. Expired work cannot replace old.
    if Instant::now()<deadline && prune(instances,&mut trial,free,roots).is_ok(){
        let value=metrics(instances,&trial,free,roots);
        if Instant::now()<deadline && value.0.is_finite() && value.1.is_finite()
            && value.0<=before.0 && value.1<=before.1
            && (value.0<before.0 || value.1<before.1){accepted=true;after=value;}
    }
    stats.terminal_area_accepted=accepted;
    recovery_event("terminal_area",start,serde_json::json!({"old_area":before.0,"old_delay":before.1,"area":after.0,"delay":after.1,"accepted":accepted,"old_area_rounds":old_rounds,"total_area_rounds":stats.slack_rounds_started,"remaining_rounds":remaining,"old_area_trials":old_trials,"total_area_trials":stats.slack_local_trials,"configured_rounds":ps.refine_rounds,"configured_trials_per_round":ps.refine_trials,"after_all_legacy_cover_comparisons":true,"no_new_deadline":true,"scope":"strict raw area AND delay nonincrease against complete old final cover in this run; no claim about another deadline-sensitive run"}));
    if accepted {trial}else{chosen}
}

// Preserve every finite witness binding, then allocate extra producers in
// rounds across target signals. This avoids spending the whole hard cap on
// cheap gate types or early classes. All accepted covers still require a DAG.
fn introduces_cycle(
    target: usize,
    inputs: &[usize],
    chosen: &[Option<Binding>],
    instances: &[Instance],
    free: &[bool],
) -> bool {
    let mut stack = inputs.to_vec();
    let mut seen = HashSet::new();
    while let Some(s) = stack.pop() {
        if s == target {
            return true;
        }
        if free[s] || !seen.insert(s) {
            continue;
        }
        if let Some(b) = chosen[s] {
            stack.extend(instances[b.instance].inputs.iter().copied());
        }
    }
    false
}

fn required_times(
    instances: &[Instance],
    chosen: &[Option<Binding>],
    free: &[bool],
    roots: &[usize],
    delay: f64,
) -> Vec<f64> {
    fn visit(
        s: usize,
        instances: &[Instance],
        chosen: &[Option<Binding>],
        free: &[bool],
        seen: &mut [bool],
        order: &mut Vec<usize>,
    ) {
        if free[s] || seen[s] {
            return;
        }
        seen[s] = true;
        let b = chosen[s].expect("validated cover");
        for &input in &instances[b.instance].inputs {
            visit(input, instances, chosen, free, seen, order);
        }
        order.push(s);
    }
    let mut order = Vec::new();
    let mut seen = vec![false; free.len()];
    for &s in roots {
        visit(s, instances, chosen, free, &mut seen, &mut order);
    }
    let mut required = vec![f64::INFINITY; free.len()];
    for &s in roots {
        required[s] = delay;
    }
    for s in order.into_iter().rev() {
        let b = chosen[s].expect("validated cover");
        let m = &instances[b.instance];
        for (p, &input) in m.inputs.iter().enumerate() {
            if m.outputs[b.pin].has_arc(p) {
                required[input] =
                    required[input].min(required[s] - f64::from(m.outputs[b.pin].delays[p]));
            }
        }
    }
    required
}

fn cap_instances(
    instances: &mut Vec<Instance>,
    free: &[bool],
    roots: &[usize],
    budget: usize,
    timing_cap: bool,
) -> Result<usize, String> {
    cap_instances_impl(instances, free, roots, budget, timing_cap, false)
}

fn interleave_ranks(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut seen=BTreeSet::new(); let mut out=Vec::new();
    for i in 0..a.len().max(b.len()) {
        for list in [a,b] {
            if let Some(&v)=list.get(i) {if seen.insert(v){out.push(v);}}
        }
    }
    out
}

fn cap_instances_impl(instances:&mut Vec<Instance>,free:&[bool],roots:&[usize],budget:usize,timing_cap:bool,queued:bool)->Result<usize,String>{cap_instances_joint(instances,free,roots,budget,timing_cap,queued,false)}
fn cap_instances_joint(instances:&mut Vec<Instance>,free:&[bool],roots:&[usize],budget:usize,timing_cap:bool,queued:bool,joint:bool)->Result<usize,String>{cap_instances_guarded(instances,free,roots,budget,timing_cap,queued,joint,false)}
// A frontier can be carried only when it was required for the actual
// retention decision. Reporting-only frontiers never cross this interface.
#[derive(Debug)]
struct PoolRetention {pruned:usize,frontier:Option<GenerationFrontier>}
#[derive(Debug)]
enum FrontierIndexSpace {Retained,Raw}
#[derive(Debug)]
struct GenerationFrontier {chosen:Vec<Option<Binding>>,area_witness:Vec<Option<Binding>>,raw_keep:BTreeSet<usize>,indices:FrontierIndexSpace,origin:&'static str}
impl GenerationFrontier {
 fn resolve(mut self,instances:&[Instance],free:&[bool],roots:&[usize])->Result<(Vec<Option<Binding>>,Vec<Option<Binding>>,&'static str),String>{
  if self.chosen.len()!=free.len() || self.area_witness.len()!=free.len(){return Err("generation frontier signal space mismatch".into());}
  let ids:Vec<_>=self.raw_keep.into_iter().collect();
  let remap=|cover:&mut Vec<Option<Binding>>|->Result<(),String>{for b in cover.iter_mut().flatten(){b.instance=ids.binary_search(&b.instance).map_err(|_|"required generation witness evicted by cap")?;}Ok(())};
  if matches!(self.indices,FrontierIndexSpace::Raw){remap(&mut self.chosen)?;}
  remap(&mut self.area_witness)?;
  for cover in [&self.chosen,&self.area_witness]{for (signal,b)in cover.iter().enumerate(){if let Some(b)=b {
   if b.instance>=instances.len() || b.pin>=instances[b.instance].outputs.len(){return Err("invalid retained generation frontier index".into());}
   let targets=&instances[b.instance].outputs[b.pin].targets;
   if targets.binary_search(&signal).is_err()&&!targets.contains(&signal){return Err("generation witness target changed on retention".into());}
  }}}
  // The full area witness was produced by the existing availability queue, and
  // all of its physical producers are mandatory in the retained pool. Ordered
  // compaction only renames instance indices; it does not change that DAG.
  prune(instances,&mut self.chosen,free,roots)?;
  Ok((self.chosen,self.area_witness,self.origin))
 }
}

fn cap_instances_guarded(instances:&mut Vec<Instance>,free:&[bool],roots:&[usize],budget:usize,timing_cap:bool,queued_witness:bool,joint:bool,guard:bool)->Result<usize,String>{
 Ok(cap_instances_with_frontier(instances,free,roots,budget,timing_cap,queued_witness,joint,guard,false)?.pruned)
}
fn cap_instances_with_frontier(
    instances: &mut Vec<Instance>,
    free: &[bool],
    roots: &[usize],
    budget: usize,
    timing_cap: bool,
    queued_witness: bool,
    joint: bool,
    guard: bool,
    capture_frontier: bool,
) -> Result<PoolRetention, String> {
    let algorithmic_frontier=joint && instances.len()>budget;
    let mut raw_reference=None;let mut retained_reference=None;
    let diagnose = std::env::var("EGM_POOL_DIAG_PATH").ok();
    let audit=std::env::var("EGM_POOL_FRONTIER_AUDIT").as_deref()==Ok("1");
    if instances.len() <= budget && diagnose.is_none() && !audit {
        return Ok(PoolRetention{pruned:0,frontier:None});
    }
    let witness = if queued_witness {
        finite_choices_queued(instances, free)
    } else {
        finite_choices(instances, free)
    };
    let mut check = witness.clone();
    prune(instances, &mut check, free, roots)?;
    let mut keep: BTreeSet<usize> = witness.iter().flatten().map(|b| b.instance).collect();
    let area_witness_count=keep.len();let frontier_start=Instant::now();
    // Candidate-generation ranking, inside the unchanged native/mapping guards;
    // NOT a second recovery run or extra two-second recovery window. It returns
    // bindings already present in generated instances, never new instances.
    let mut fast=if algorithmic_frontier||audit {
        Some(delay_frontier_seed(instances,free,frontier_start+std::time::Duration::from_secs(5)).ok_or("budget: candidate-retention frontier unfinished within native subguard")?.0)
    }else{None};
    let before=if audit {Some(pool_delay_certificate(instances,free,roots,fast.as_ref().unwrap())?)}else{None};
    let mut legacy_keep=None;
    let mut adopt_fast=joint && instances.len()>budget;
    if adopt_fast && guard {
        let original=fill_optional_pool(instances,free,budget,timing_cap,&witness,keep.clone());
        let old_pool:Vec<_>=original.iter().map(|&i|instances[i].clone()).collect();
        let (mut old_fast,_)=delay_frontier_seed(&old_pool,free,Instant::now()+std::time::Duration::from_secs(5)).ok_or("budget: old-pool root-delay guard incomplete")?;
        prune(&old_pool,&mut old_fast,free,roots)?;
        let mut full=fast.as_ref().unwrap().clone();prune(instances,&mut full,free,roots)?;
        let new_delay=metrics(instances,&full,free,roots).1;let old_delay=metrics(&old_pool,&old_fast,free,roots).1;
        adopt_fast=new_delay+1e-6<old_delay;
        if let Some(path)=&diagnose {std::fs::write(std::path::Path::new(path).with_extension("pool-guard.json"),serde_json::json!({"guard":true,"old_pool_instances":original.len(),"old_root_delay":old_delay,"full_root_delay":new_delay,"adopt_fast":adopt_fast,"scope":"generation ranking inside unchanged native/mapping guard; old pool retained exactly when no root-delay gain; not old final area certificate"}).to_string()).map_err(|e|e.to_string())?;}
        if !adopt_fast {legacy_keep=Some(original);if capture_frontier {retained_reference=Some(old_fast);}}
        else if capture_frontier {raw_reference=Some(full);}
    }
    if adopt_fast {if let Some(full)=&fast {keep.extend(full.iter().flatten().map(|b|b.instance));}}
    let union_witness_count=keep.len();let frontier_secs=frontier_start.elapsed().as_secs_f64();
    if keep.len() > budget {
        return Err(format!(
            "instance budget {budget} below finite witness {}",
            keep.len()
        ));
    }
    let witness_kept = keep.len();
    if instances.len() <= budget {
        if let Some(ref path) = diagnose {
            let all = (0..instances.len()).collect();
            write_pool_diagnostic(instances, free, roots, &witness, &all, witness_kept, budget, &path)?;
        }
        if audit {write_frontier_audit(diagnose.as_deref(),joint,area_witness_count,union_witness_count,frontier_secs,before.clone().unwrap(),before.unwrap())?;}
        return Ok(PoolRetention{pruned:0,frontier:None});
    }
    keep=legacy_keep.unwrap_or_else(||fill_optional_pool(instances,free,budget,timing_cap,&witness,keep));
    if let Some(ref path) = diagnose {
        write_pool_diagnostic(instances, free, roots, &witness, &keep, witness_kept, budget, &path)?;
    }
    let original = instances.len();
    let mut i = 0;
    instances.retain(|_| {
        let yes = keep.contains(&i);
        i += 1;
        yes
    });
    if audit {
        let full=delay_frontier_seed(instances,free,Instant::now()+std::time::Duration::from_secs(5)).ok_or("budget: retained-pool diagnostic unfinished")?.0;
        let after=pool_delay_certificate(instances,free,roots,&full)?;
        write_frontier_audit(diagnose.as_deref(),joint,area_witness_count,union_witness_count,frontier_secs,before.unwrap(),after)?;
    }
    // Only data already required by the real retention decision is returned.
    // The separate reporting-only `full` above is deliberately not considered.
    let frontier=if capture_frontier && algorithmic_frontier {
        if let Some(chosen)=retained_reference {Some(GenerationFrontier{chosen,area_witness:witness,raw_keep:keep,indices:FrontierIndexSpace::Retained,origin:"legacy-pool-guard"})}
        else if adopt_fast {Some(GenerationFrontier{chosen:raw_reference.or_else(||fast.take()).ok_or("missing necessary raw retention frontier")?,area_witness:witness,raw_keep:keep,indices:FrontierIndexSpace::Raw,origin:"full-pool-retention"})}
        else {None}
    }else{None};
    Ok(PoolRetention{pruned:original-instances.len(),frontier})
}

// Read-only opt-in diagnostic. Scores use ONE common area-ranked witness;
// they are local estimates, not minimum-delay bounds or alternative mapped covers.
fn write_pool_diagnostic(
    instances: &[Instance], free: &[bool], roots: &[usize],
    witness: &[Option<Binding>], keep: &BTreeSet<usize>,
    witness_kept: usize, budget: usize, path: &str,
) -> Result<(), String> {
    write_pool_diagnostic_detail(instances,free,roots,witness,keep,witness_kept,budget,path,std::env::var("EGM_POOL_DIAG_DETAIL").as_deref()!=Ok("0"))
}
fn write_pool_diagnostic_detail(
    instances: &[Instance], free: &[bool], roots: &[usize],
    witness: &[Option<Binding>], keep: &BTreeSet<usize>,
    witness_kept: usize, budget: usize, path: &str, detail:bool,
) -> Result<(), String> {
    if !detail {
        let data=serde_json::json!({"instances_before":instances.len(),"instances_after":keep.len(),"witness_reserved_instances":witness_kept,"budget":budget,"roots":roots,"signals":[],"detailed":false,"interpretation":"counts only; optional area-proxy labels and rankings not computed; mapping decisions unchanged"});
        return std::fs::write(path,serde_json::to_vec(&data).map_err(|e|e.to_string())?).map_err(|e|e.to_string());
    }
    let all_roots: Vec<_> = witness.iter().enumerate()
        .filter_map(|(s,b)| b.map(|_| s)).collect();
    let arrivals = critical_arrivals(instances, witness, free, &all_roots).0;
    let mut before = vec![0usize; free.len()];
    let mut after = before.clone();
    let mut best_before: Vec<Option<(f64,usize,usize)>> = vec![None; free.len()];
    let mut best_after = best_before.clone();
    for (i,m) in instances.iter().enumerate() {
        for (pin,o) in m.outputs.iter().enumerate() {
            let score = m.inputs.iter().enumerate().filter(|(p,_)| o.has_arc(*p))
                .map(|(p,&s)| arrivals[s].unwrap_or(f64::INFINITY) + f64::from(o.delays[p]))
                .fold(0.0, f64::max);
            for &s in &o.targets {
                before[s] += 1;
                if best_before[s].is_none_or(|v| score < v.0) { best_before[s]=Some((score,i,pin)); }
                if keep.contains(&i) {
                    after[s] += 1;
                    if best_after[s].is_none_or(|v| score < v.0) { best_after[s]=Some((score,i,pin)); }
                }
            }
        }
    }
    let describe = |x: Option<(f64,usize,usize)>| x.map(|(score,i,pin)| {
        let m=&instances[i]; let o=&m.outputs[pin];
        serde_json::json!({"area_witness_arrival_estimate":score,"cell":m.cell,
            "inputs":m.inputs,"input_pins":m.input_pins,"output_pin":o.name,
            "delays":o.delays,"area":m.area,"kept":keep.contains(&i)})
    });
    let rows: Vec<_> = (0..free.len()).filter(|&s| before[s]>0).map(|s|
        serde_json::json!({"signal":s,"before":before[s],"after":after[s],
            "witness_arrival":arrivals[s],"best_before":describe(best_before[s]),
            "best_after":describe(best_after[s])})).collect();
    let data=serde_json::json!({"instances_before":instances.len(),"instances_after":keep.len(),
        "witness_reserved_instances":witness_kept,"budget":budget,"roots":roots,
        "interpretation":"common area-witness local estimates; not delay lower bounds or PPA",
        "signals":rows});
    std::fs::write(path,serde_json::to_vec(&data).map_err(|e|e.to_string())?)
        .map_err(|e|e.to_string())
}

fn retain_possible_cone(instances: &mut Vec<Instance>, free: &[bool], roots: &[usize]) {
    let mut producers: Vec<Vec<usize>> = vec![Vec::new(); free.len()];
    for (i, m) in instances.iter().enumerate() {
        for output in &m.outputs {
            for &s in &output.targets {
                producers[s].push(i);
            }
        }
    }
    let mut needed = vec![false; free.len()];
    let mut used = vec![false; instances.len()];
    let mut stack = roots.to_vec();
    while let Some(s) = stack.pop() {
        if free[s] || needed[s] {
            continue;
        }
        needed[s] = true;
        for &i in &producers[s] {
            if !used[i] {
                used[i] = true;
                stack.extend(instances[i].inputs.iter().copied());
            }
        }
    }
    for m in instances.iter_mut() {
        for output in &mut m.outputs {
            output.targets.retain(|&s| needed[s]);
        }
    }
    instances.retain(|m| m.outputs.iter().any(|o| !o.targets.is_empty()));
}

fn prune(
    instances: &[Instance],
    chosen: &mut [Option<Binding>],
    free: &[bool],
    roots: &[usize],
) -> Result<(), String> {
    fn visit(
        s: usize,
        instances: &[Instance],
        chosen: &[Option<Binding>],
        free: &[bool],
        state: &mut [u8],
    ) -> Result<(), String> {
        if free[s] || state[s] == 2 {
            return Ok(());
        }
        if state[s] == 1 {
            return Err(format!("cycle through signal {s}"));
        }
        state[s] = 1;
        let b = chosen[s].ok_or_else(|| format!("undriven signal {s}"))?;
        let m = &instances[b.instance];
        if !m.outputs[b.pin].targets.contains(&s) {
            return Err(format!("invalid binding for signal {s}"));
        }
        for &input in &m.inputs {
            visit(input, instances, chosen, free, state)?;
        }
        state[s] = 2;
        Ok(())
    }
    let mut state = vec![0; free.len()];
    for &root in roots {
        visit(root, instances, chosen, free, &mut state)?;
    }
    for (s, choice) in chosen.iter_mut().enumerate() {
        if state[s] != 2 {
            *choice = None;
        }
    }
    Ok(())
}

// The cover is a validated DAG. Mark all timing arcs on any worst-delay path,
// including ties and demanded pins of a multioutput instance. Absent inputs have
// no predicted arrival and are left to the original area ordering.
fn critical_arrivals(
    instances: &[Instance],
    chosen: &[Option<Binding>],
    free: &[bool],
    roots: &[usize],
) -> (Vec<Option<f64>>, Vec<bool>) {
    fn visit(
        s: usize,
        instances: &[Instance],
        chosen: &[Option<Binding>],
        free: &[bool],
        cache: &mut [Option<f64>],
    ) -> f64 {
        if let Some(v) = cache[s] {
            return v;
        }
        let v = if free[s] {
            0.0
        } else {
            let b = chosen[s].expect("validated cover");
            let m = &instances[b.instance];
            m.inputs
                .iter()
                .enumerate()
                .filter(|(p, _)| m.outputs[b.pin].has_arc(*p))
                .map(|(p, &input)| {
                    visit(input, instances, chosen, free, cache)
                        + f64::from(m.outputs[b.pin].delays[p])
                })
                .fold(0.0, f64::max)
        };
        cache[s] = Some(v);
        v
    }
    let mut arrival = vec![None; free.len()];
    for (s, &is_free) in free.iter().enumerate() {
        if is_free {
            arrival[s] = Some(0.0);
        }
    }
    let worst = roots
        .iter()
        .map(|&s| visit(s, instances, chosen, free, &mut arrival))
        .fold(0.0, f64::max);
    let mut critical = vec![false; free.len()];
    let mut stack: Vec<_> = roots
        .iter()
        .copied()
        .filter(|&s| arrival[s].is_some_and(|t| (t - worst).abs() <= 1e-6))
        .collect();
    while let Some(s) = stack.pop() {
        if critical[s] {
            continue;
        }
        critical[s] = true;
        if free[s] {
            continue;
        }
        let b = chosen[s].expect("validated cover");
        let m = &instances[b.instance];
        for (p, &input) in m.inputs.iter().enumerate() {
            if m.outputs[b.pin].has_arc(p)
                && arrival[input].is_some_and(|t| {
                    (t + f64::from(m.outputs[b.pin].delays[p]) - arrival[s].unwrap()).abs() <= 1e-6
                })
            {
                stack.push(input);
            }
        }
    }
    (arrival, critical)
}

fn critical_gain(
    m: &Instance,
    chosen: &[Option<Binding>],
    arrival: &[Option<f64>],
    critical: &[bool],
) -> f64 {
    m.outputs
        .iter()
        .filter_map(|output| {
            let mut predicted: f64 = 0.0;
            for (p, &input) in m.inputs.iter().enumerate() {
                if output.has_arc(p) {
                    predicted = predicted.max(arrival[input]? + f64::from(output.delays[p]));
                }
            }
            output
                .targets
                .iter()
                .filter(|&&s| critical[s] && chosen[s].is_some())
                .filter_map(|&s| arrival[s].map(|old| (predicted - old).min(0.0)))
                .min_by(f64::total_cmp)
        })
        .min_by(f64::total_cmp)
        .unwrap_or(0.0)
}

#[derive(Default)]
struct MetricWorkspace {live:Vec<usize>, values:Vec<f64>, stamps:Vec<u32>, epoch:u32}
thread_local! {
    static METRIC_EPOCHS:std::cell::Cell<bool>=const{std::cell::Cell::new(false)};
    static METRIC_WORKSPACE:std::cell::RefCell<MetricWorkspace>=std::cell::RefCell::new(MetricWorkspace::default());
}
struct MetricEpochGuard(bool);
impl MetricEpochGuard {fn enter(enabled:bool)->Self{Self(METRIC_EPOCHS.with(|v|v.replace(enabled)))}}
impl Drop for MetricEpochGuard {fn drop(&mut self){METRIC_EPOCHS.with(|v|v.set(self.0));}}

fn metrics(
    instances: &[Instance],
    chosen: &[Option<Binding>],
    free: &[bool],
    roots: &[usize],
) -> (f64, f64) {
    if METRIC_EPOCHS.with(|v|v.get()){return metrics_epoch(instances,chosen,free,roots)}
    fn arrival(
        s: usize,
        instances: &[Instance],
        chosen: &[Option<Binding>],
        free: &[bool],
        cache: &mut [Option<f64>],
    ) -> f64 {
        if free[s] {
            return 0.0;
        }
        if let Some(v) = cache[s] {
            return v;
        }
        let b = chosen[s].expect("cover was validated");
        let m = &instances[b.instance];
        let value = m
            .inputs
            .iter()
            .zip(&m.outputs[b.pin].delays)
            .enumerate()
            .filter(|(i, _)| m.outputs[b.pin].has_arc(*i))
            .map(|(_, (&input, &delay))| {
                arrival(input, instances, chosen, free, cache) + f64::from(delay)
            })
            .fold(0.0, f64::max);
        cache[s] = Some(value);
        value
    }
    // Preserve ascending physical-ID summation and single-instance charging,
    // without allocating a tree node for every bound signal on every trial.
    let mut live: Vec<_> = chosen.iter().flatten().map(|b| b.instance).collect();
    live.sort_unstable();live.dedup();
    let area = live.into_iter().map(|i| f64::from(instances[i].area)).sum();
    let mut cache = vec![None; free.len()];
    let delay = roots
        .iter()
        .map(|&s| arrival(s, instances, chosen, free, &mut cache))
        .fold(0.0, f64::max);
    (area, delay)
}

fn metrics_epoch(
    instances: &[Instance],
    chosen: &[Option<Binding>],
    free: &[bool],
    roots: &[usize],
) -> (f64, f64) {
    fn arrival(
        s: usize,
        instances: &[Instance],
        chosen: &[Option<Binding>],
        free: &[bool],
        values: &mut [f64], stamps: &mut [u32], epoch:u32,
    ) -> f64 {
        if free[s] {
            return 0.0;
        }
        if stamps[s]==epoch {return values[s];}
        let b = chosen[s].expect("cover was validated");
        let m = &instances[b.instance];
        let value = m
            .inputs
            .iter()
            .zip(&m.outputs[b.pin].delays)
            .enumerate()
            .filter(|(i, _)| m.outputs[b.pin].has_arc(*i))
            .map(|(_, (&input, &delay))| {
                arrival(input, instances, chosen, free, values,stamps,epoch) + f64::from(delay)
            })
            .fold(0.0, f64::max);
        values[s]=value;stamps[s]=epoch;
        value
    }
    METRIC_WORKSPACE.with(|workspace| {
        let mut workspace=workspace.borrow_mut();
        workspace.values.resize(free.len(),0.0);
        workspace.stamps.resize(free.len(),0);
        workspace.epoch=workspace.epoch.wrapping_add(1);
        if workspace.epoch==0 {workspace.stamps.fill(0);workspace.epoch=1;}
        let MetricWorkspace{live,values,stamps,epoch}=&mut *workspace;
        let epoch=*epoch;
        live.clear();live.extend(chosen.iter().flatten().map(|b|b.instance));
        live.sort_unstable();live.dedup();
        // Exact same physical-ID summation order as the original implementation.
        let area=live.iter().map(|&i|f64::from(instances[i].area)).sum();
        let delay=roots.iter().map(|&s|arrival(s,instances,chosen,free,values,stamps,epoch)).fold(0.0,f64::max);
        (area,delay)
    })
}

fn extract(
    instances: &[Instance],
    free: &[bool],
    roots: &[usize],
    ps: &MapParams,
    stats: &mut MapStats,
) -> Result<Vec<Option<Binding>>, String> {
    let seed = heuristic_cover(instances, free, roots, ps, stats)?;
    let seed_metrics = metrics(instances, &seed, free, roots);
    let feasible_delay = |delay: f64| ps.max_delay.is_none_or(|limit| delay <= limit + 1e-6);
    if ps.time_limit_secs <= 0.0 {
        if !feasible_delay(seed_metrics.1) {
            return Err("fallback exceeds the requested delay bound".into());
        }
        stats.used_fallback = true;
        stats.solver_status = "heuristic-only".into();
        return Ok(seed);
    }
    let n = free.len();
    let mut model = Model::default();
    model.set_obj_sense(Sense::Minimize);
    model.set_parameter("log", "0");
    model.set_parameter("seconds", &ps.time_limit_secs.to_string());
    let x: Vec<_> = instances
        .iter()
        .map(|m| {
            let col = model.add_binary();
            model.set_obj_coeff(
                col,
                if ps.objective == MapObjective::DelayArea {
                    0.0
                } else {
                    f64::from(m.area)
                },
            );
            col
        })
        .collect();
    let mut bindings: Vec<(usize, Binding, Col)> = Vec::new();
    let mut producers: Vec<Vec<Col>> = vec![Vec::new(); n];
    for (i, m) in instances.iter().enumerate() {
        let row = model.add_row();
        model.set_row_upper(row, 0.0);
        model.set_weight(row, x[i], 1.0);
        for (pin, output) in m.outputs.iter().enumerate() {
            for &s in &output.targets {
                let z = model.add_binary();
                bindings.push((s, Binding { instance: i, pin }, z));
                producers[s].push(z);
                model.set_weight(row, z, -1.0); // x implies at least one used output binding
                let activation = model.add_row();
                model.set_row_upper(activation, 0.0);
                model.set_weight(activation, z, 1.0);
                model.set_weight(activation, x[i], -1.0);
            }
        }
    }
    for (s, prod) in producers.iter().enumerate() {
        let row = model.add_row();
        model.set_row_upper(row, if free[s] { 0.0 } else { 1.0 });
        if roots.contains(&s) && !free[s] {
            model.set_row_lower(row, 1.0);
        }
        for &z in prod {
            model.set_weight(row, z, 1.0);
        }
    }
    for (i, m) in instances.iter().enumerate() {
        for &s in &m.inputs {
            if free[s] {
                continue;
            }
            let row = model.add_row();
            model.set_row_upper(row, 0.0);
            model.set_weight(row, x[i], 1.0);
            for &z in &producers[s] {
                model.set_weight(row, z, -1.0);
            }
        }
    }
    // Each used pin, including secondary pins, must be above all its physical input signals.
    let big_level = n as f64 + 1.0;
    let level: Vec<_> = free
        .iter()
        .map(|&is_free| {
            let col = model.add_col();
            model.set_col_lower(col, 0.0);
            model.set_col_upper(col, if is_free { 0.0 } else { big_level });
            col
        })
        .collect();
    let timed =
        ps.objective != MapObjective::Weighted || ps.delay_weight > 0.0 || ps.max_delay.is_some();
    let max_arc = instances
        .iter()
        .flat_map(|m| &m.outputs)
        .flat_map(|o| &o.delays)
        .copied()
        .fold(0.0f32, f32::max);
    let big_time = f64::from(max_arc) * big_level + 1.0;
    let arrivals: Vec<_> = if timed {
        free.iter()
            .map(|&is_free| {
                let col = model.add_col();
                model.set_col_lower(col, 0.0);
                model.set_col_upper(col, if is_free { 0.0 } else { big_time });
                col
            })
            .collect()
    } else {
        Vec::new()
    };
    for &(s, b, z) in &bindings {
        let m = &instances[b.instance];
        for (input_index, &input) in m.inputs.iter().enumerate() {
            let row = model.add_row();
            model.set_row_lower(row, 1.0 - big_level);
            model.set_weight(row, level[s], 1.0);
            model.set_weight(row, level[input], -1.0);
            model.set_weight(row, z, -big_level);
            if timed && m.outputs[b.pin].has_arc(input_index) {
                let row = model.add_row();
                model.set_row_lower(
                    row,
                    f64::from(m.outputs[b.pin].delays[input_index]) - big_time,
                );
                model.set_weight(row, arrivals[s], 1.0);
                model.set_weight(row, arrivals[input], -1.0);
                model.set_weight(row, z, -big_time);
            }
        }
    }
    let worst = if timed {
        let worst = model.add_col();
        model.set_col_lower(worst, 0.0);
        model.set_col_upper(worst, ps.max_delay.unwrap_or(big_time).min(big_time));
        model.set_obj_coeff(
            worst,
            match ps.objective {
                MapObjective::DelayArea => 1.0,
                MapObjective::AreaDelay => 0.0,
                MapObjective::Weighted => ps.delay_weight,
            },
        );
        for &root in roots {
            let row = model.add_row();
            model.set_row_lower(row, 0.0);
            model.set_weight(row, worst, 1.0);
            model.set_weight(row, arrivals[root], -1.0);
        }
        Some(worst)
    } else {
        None
    };
    // Supply feasible binary assignments; CBC completes continuous level/arrival columns.
    if feasible_delay(seed_metrics.1) {
        if let Some(worst) = worst {
            // A zero initial worst-arrival gives CBC an incorrect warm-start objective
            // in delay-first mode even though the supplied binary cover has nonzero delay.
            model.set_col_initial_solution(worst, seed_metrics.1);
        }
        let selected: HashSet<_> = seed.iter().flatten().map(|b| b.instance).collect();
        for (i, &col) in x.iter().enumerate() {
            model.set_col_initial_solution(col, f64::from(selected.contains(&i)));
        }
        for &(s, b, col) in &bindings {
            let yes = seed[s].is_some_and(|old| old.instance == b.instance && old.pin == b.pin);
            model.set_col_initial_solution(col, f64::from(yes));
        }
    }
    // With lexicographic objectives and tightened arrival bounds, the installed CBC
    // can return fractional LP columns after a continuous warm start. Solve these
    // phases without that warm start; retain the independently checked seed in Rust.
    if ps.objective != MapObjective::Weighted {
        model.remove_initial_solution();
    }
    let start = Instant::now();
    let decode = |solution: &coin_cbc::Solution| -> Option<Vec<Option<Binding>>> {
        let integral = x
            .iter()
            .chain(bindings.iter().map(|(_, _, z)| z))
            .all(|&col| {
                let v = solution.col(col);
                v.is_finite() && (v.abs() < 1e-6 || (v - 1.0).abs() < 1e-6)
            });
        if !integral {
            return None;
        }
        let mut chosen = vec![None; n];
        for &(s, b, col) in &bindings {
            if solution.col(col) > 0.5 {
                if chosen[s].is_some() || solution.col(x[b.instance]) <= 0.5 || free[s] {
                    return None;
                }
                chosen[s] = Some(b);
            }
        }
        if prune(instances, &mut chosen, free, roots).is_err() {
            return None;
        }
        if !feasible_delay(metrics(instances, &chosen, free, roots).1) {
            return None;
        }
        Some(chosen)
    };
    let solution = model.solve();
    stats.solver_status = format!("{:?}", solution.raw().status());
    let mut result = decode(&solution);
    let mut proven = solution.raw().is_proven_optimal();
    if ps.objective != MapObjective::Weighted {
        if let Some(first) = &result {
            // Sequential optimization, not an epsilon-weight approximation. Preserve the
            // best known primary value; only two proven solves certify lexicographic optimality.
            let value = metrics(instances, first, free, roots);
            let remaining = ps.time_limit_secs - start.elapsed().as_secs_f64();
            if remaining > 0.01 {
                for (i, &col) in x.iter().enumerate() {
                    model.set_obj_coeff(
                        col,
                        if ps.objective == MapObjective::DelayArea {
                            f64::from(instances[i].area)
                        } else {
                            0.0
                        },
                    );
                }
                let worst = worst.expect("lexicographic timing variable");
                if ps.objective == MapObjective::AreaDelay {
                    let row = model.add_row();
                    model.set_row_upper(row, value.0 + 1e-7);
                    for (i, &col) in x.iter().enumerate() {
                        model.set_weight(row, col, f64::from(instances[i].area));
                    }
                    model.set_obj_coeff(worst, 1.0);
                } else {
                    model.set_col_upper(
                        worst,
                        (value.1 + 1e-7).min(ps.max_delay.unwrap_or(big_time)),
                    );
                    model.set_obj_coeff(worst, 0.0);
                }
                // Do not carry CBC's continuous first-stage warm start into a changed
                // objective/bound: the installed libcbc can expose the LP relaxation as the
                // returned column solution. The validated first cover remains our fallback.
                model.remove_initial_solution();
                model.set_parameter("seconds", &remaining.to_string());
                let second = model.solve();
                stats.solver_status = format!(
                    "primary={:?}, secondary={:?}",
                    solution.raw().status(),
                    second.raw().status()
                );
                let secondary_proven = second.raw().is_proven_optimal();
                if let Some(cover) = decode(&second) {
                    let after = metrics(instances, &cover, free, roots);
                    if better_cover(after, value, ps) || !better_cover(value, after, ps) {
                        result = Some(cover);
                        proven &= secondary_proven;
                    } else {
                        proven = false;
                    }
                } else {
                    proven = false;
                }
            } else {
                proven = false;
            }
        }
    }
    stats.solve_secs = start.elapsed().as_secs_f64();
    if let Some(chosen) = result {
        let value = metrics(instances, &chosen, free, roots);
        if !feasible_delay(seed_metrics.1) || !better_cover(seed_metrics, value, ps) {
            stats.proven_optimal = proven;
            return Ok(chosen);
        }
    }
    if !feasible_delay(seed_metrics.1) {
        return Err(format!(
            "{}: no valid cover within delay bound",
            stats.solver_status
        ));
    }
    stats.used_fallback = true;
    Ok(seed)
}

/// Saturate, match and extract without first collapsing the e-graph to one AIG.
pub fn map(aig: &Aig, tech: &TechLib, ps: &MapParams) -> Result<MappedGraph, String> {
    if !(2..=6).contains(&ps.max_inputs)
        || ps.dc_support > 20
        || ps.cuts_per_class == 0
        || ps.max_instances == 0
        || ps.boolean_rules > 2
        || !ps.time_limit_secs.is_finite()
        || !ps.phase_time_fraction.is_finite()
        || !(0.0..=1.0).contains(&ps.phase_time_fraction)
        || !ps.refine_time_secs.is_finite()
        || !(0.0..=3600.0).contains(&ps.refine_time_secs)
        || !ps.delay_weight.is_finite()
        || ps.delay_weight < 0.0
        || ps.max_delay.is_some_and(|v| !v.is_finite() || v < 0.0)
    {
        return Err("invalid mapping budgets, input limit, or timing objective".into());
    }
    if ps.protected_exchange && (!ps.strict_budgets || ps.objective!=MapObjective::DelayArea || ps.multioutput || !ps.functional_cuts || !ps.canonical_wire_only || ps.time_limit_secs>0. || ps.cut_rounds==0 || ps.flow_compare) {return Err("protected exchange requires strict single-output DA, functional cuts, raw alias wires, nonzero cut rounds, heuristic-only and no flow-compare".into());}
    if ps.terminal_area && (!ps.strict_budgets || ps.objective!=MapObjective::DelayArea || ps.tail_shared_area || ps.flow_compare || !ps.slack_flow || ps.no_actual_area_tail){return Err("terminal area requires strict delay-area, slack flow, old tail0 and flow-compare0".into());}
    if ps.cone_import && (!ps.strict_budgets || ps.max_inputs<3){return Err("bounded cone import requires strict budgets and at least three-input cuts".into())}
    let mut library_pins: HashMap<&str, (f32, HashSet<&str>)> = HashMap::new();
    for cell in &tech.cells {
        if !cell.area.is_finite()
            || cell.area < 0.0
            || cell.pin_delay.values().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(format!("invalid area/delay in {}", cell.name));
        }
        let group = library_pins
            .entry(&cell.name)
            .or_insert((cell.area, HashSet::new()));
        if (group.0 - cell.area).abs() > 1e-6 || !group.1.insert(&cell.out_pin) {
            return Err(format!(
                "inconsistent area or duplicate output pin in {}",
                cell.name
            ));
        }
    }
    let start = Instant::now();
    if ps.proof_source && !ps.strict_budgets {return Err("source proof requires strict global budgets".into());}
    if ps.shared_majority && (!ps.strict_budgets || !ps.cone_import){return Err("library source planning requires strict budgets and cone import".into());}
    let compressed=ps.proof_source.then(||crate::source_proof::compress_with_propagation(aig,ps.source_propagation));
    let aig=compressed.as_ref().unwrap_or(aig);
    let literal_import=ps.literal_import || (ps.literal_auto && regions::ordinary_node_upper(aig).ok_or("cannot plan AIG")? > ps.saturation.node_limit);
    let cone_requested=ps.cone_import;
    let compact_upper=regions::compact_node_upper(aig).ok_or("cannot plan compact import")?;
    let xor_import=cone_requested && compact_upper>regions::initial_node_ceiling(ps.saturation.node_limit);
    if let Some(path)=std::env::var_os("EGM_POOL_DIAG_PATH"){let path=std::path::Path::new(&path).with_extension("import-plan.json");let _=std::fs::write(path,serde_json::json!({"compact_upper":compact_upper,"node_limit":ps.saturation.node_limit,"threshold_percent":90,"cone_requested":cone_requested,"xor_import":xor_import,"policy":"Fixed 90%-of-node-limit compact-size threshold, unified with10% optional-helper rewrite reserve uses exact XOR/TT3 contraction; no benchmark names or PPA dispatch"}).to_string());}
    let mut admission = crate::egraph_budget::NodeAdmission::new(if ps.strict_budgets || xor_import { ps.saturation.node_limit } else { usize::MAX });
    let (graph, roots, source_pairs) = if xor_import {
        if ps.shared_majority {
            source_import::import_source_admitted(aig,2048,ps.saturation.node_limit,true,ps.max_inputs.clamp(3,4),Some(tech),&mut admission)
        } else {
            regions::import_helpers_admitted(aig,2048,ps.saturation.node_limit,ps.canonical_source_inputs,ps.import_helper_fill,&mut admission)
        }
    } else if literal_import {
        regions::import_literals_admitted(aig,&mut admission).map(|(g,r)|(g,r,Vec::new()))
    } else if ps.region_classes > 0 {
        regions::import_demanded_admitted(aig,&mut admission).map(|(g,r)|(g,r,Vec::new()))
    } else {
        crate::egraph_sat::import_aig_admitted(aig,&mut admission).map(|(g,r)|(g,r,Vec::new()))
    }
    .ok_or_else(||format!("cannot import AIG within node allowance {}; see import diagnostics",ps.saturation.node_limit))?;
    if ps.strict_budgets && graph.total_number_of_nodes() > ps.saturation.node_limit {
        return Err(format!(
            "budget: imported graph {} exceeds node allowance {}",
            graph.total_number_of_nodes(),
            ps.saturation.node_limit
        ));
    }
    // A fixed representation policy, not a design/QoR-specific winner selection.
    // Leave the ordinary profile's return window intact; compact large covers
    // reserve more of the same global two seconds for validation.
    let mut effective=ps.clone();if ps.literal_auto && literal_import {effective.return_reserve=effective.return_reserve.max(0.15);}
    let ps=&effective;
    let imported_enodes = graph.total_number_of_nodes();
    let cumulative_imported_enodes = admission.admitted();
    let admission = std::sync::Arc::new(std::sync::Mutex::new(admission));
    let mut runner = Runner::<AigLang, (), ()>::default()
        .with_egraph(graph)
        .with_iter_limit(ps.saturation.iter_limit)
        .with_node_limit(ps.saturation.node_limit);
    runner.roots = roots.iter().map(|(_, id)| *id).collect();
    let mut equalities = rules(ps.saturation.use_factoring, ps.saturation.use_absorption);
    if literal_import {
        // Exact Boolean identities, never conditional don't-care unions. The
        // original masked AND remains a counted enode in the merged class.
        equalities.push(egg::rewrite!("literal-np-compl"; "(@egm_and_np ?a ?a)" => "0"));
        equalities.push(egg::rewrite!("literal-pn-compl"; "(@egm_and_pn ?a ?a)" => "0"));
    }
    if ps.boolean_rules >= 1 {
        // (a & b) | (a & c) = a & (b | c), expressed entirely in AIG operators.
        equalities.push(egg::rewrite!("or-factor";
            "(~ (& (~ (& ?a ?b)) (~ (& ?a ?c))))" => "(& ?a (~ (& (~ ?b) (~ ?c))))"));
    }
    if ps.boolean_rules >= 2 {
        equalities.push(egg::rewrite!("and-associate";
            "(& (& ?a ?b) ?c)" => "(& ?a (& ?b ?c))"));
    }
    if ps.library_timing && ps.objective == MapObjective::DelayArea {
        let pattern: egg::Pattern<AigLang> = "(& (& ?a ?b) ?c)".parse().unwrap();
        equalities.push(
            egg::Rewrite::new(
                "bounded-delay-associate",
                pattern,
                BoundedAssociativity {
                    rhs: "(& ?a (& ?b ?c))".parse::<egg::Pattern<AigLang>>().unwrap().ast,
                    original_classes: runner.egraph.total_size(),
                    nodes: ps.saturation.node_limit,
                    per_class: ps.cuts_per_class,
                },
            )
            .unwrap(),
        );
    }
    let skipped = if ps.strict_budgets {
        Some(regions::reserve_rewrites(
            &mut equalities,
            admission.clone(),
        ))
    } else {
        None
    };
    let runner = runner.run(&equalities);
    let graph = &runner.egraph;
    if let Some(path)=std::env::var_os("EGM_POOL_DIAG_PATH") {
        let path=std::path::Path::new(&path).with_extension("admission.json");
        let _=std::fs::write(path,serde_json::json!({"schema_version":1,"import_complete":true,"strict":ps.strict_budgets,"cumulative_imported_enodes":cumulative_imported_enodes,"cumulative_admitted_enodes":admission.lock().unwrap().admitted(),"resident_imported_enodes":imported_enodes,"resident_enodes":graph.total_number_of_nodes(),"node_allowance":ps.saturation.node_limit,"refund_on_rebuild":false,"all_strict_rewrites_preflighted":ps.strict_budgets}).to_string());
    }
    if ps.strict_budgets && graph.total_number_of_nodes() > ps.saturation.node_limit {
        return Err(format!(
            "budget: saturated graph {} exceeds node allowance {}",
            graph.total_number_of_nodes(),
            ps.saturation.node_limit
        ));
    }
    let mut classes: Vec<_> = graph.classes().map(|c| c.id).collect();
    classes.sort_by_key(|&id| usize::from(id));
    let ids: HashMap<_, _> = classes.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let nodes: Vec<_> = classes.iter().map(|&id| graph[id].nodes.clone()).collect();
    if !ps.return_reserve.is_finite() || !(0.0..=3600.0).contains(&ps.return_reserve) {return Err("invalid return reserve".into());}
    if !ps.area_time_fraction.is_finite() || !(0.0..=1.0).contains(&ps.area_time_fraction) {return Err("invalid area time fraction".into());}
    if ps.literal_auto && !ps.strict_budgets {return Err("literal auto requires strict global budgets".into());}
    if (ps.literal_import || ps.literal_auto) && ps.canonical_phases && !ps.canonical_wire_only {return Err("literal import requires raw cut interfaces when canonical phases enabled".into());}
    if ps.canonical_wire_only && !ps.canonical_phases {return Err("canonical wires requires canonical phases".into());}
    let canonical=if ps.canonical_phases {phase_aliases(&nodes,|id|ids[&graph.find(id)])?} else {(0..nodes.len()*2).collect()};
    if ps.canonical_phases && std::env::var("EGM_AREA_CUT_QUOTA").as_deref().unwrap_or("0")!="0" {
        return Err("canonical-phase prototype requires AREA_CUT_QUOTA=0".into());
    }
    let mut stats = MapStats {
        imported_enodes,
        cumulative_imported_enodes,
        cumulative_admitted_enodes: admission.lock().unwrap().admitted(),
        saturation_budget_skips: skipped
            .as_ref()
            .map_or(0, |s| s.load(std::sync::atomic::Ordering::Relaxed)),
        eclasses: classes.len(),
        enodes: graph.total_number_of_nodes(),
        saturation_stop: format!("{:?}", runner.stop_reason),
        ..Default::default()
    };
    let witnesses = if ps.strict_budgets {
        cheapest_nodes_queued(graph)
    } else {
        cheapest_nodes(graph)
    };
    let representatives: Vec<_> = classes
        .iter()
        .map(|&id| {
            witnesses
                .get(&id)
                .cloned()
                .map(|node| node.map_children(|child| graph.find(child)))
                .ok_or_else(|| format!("no finite reachability witness for eclass {id}"))
        })
        .collect::<Result<_, _>>()?;
    let mut reach = Reachability {
        nodes: representatives,
        ids: ids.clone(),
        cache: HashMap::new(),
    };
    let n = classes.len() * 2;
    let mut constants = vec![None; n];
    let mut free = vec![false; n];
    let mut inputs = Vec::new();
    for (c, enodes) in nodes.iter().enumerate() {
        for node in enodes {
            match node {
                AigLang::Zero | AigLang::One => {
                    let value = matches!(node, AigLang::One);
                    for phase in 0..2 {
                        constants[c * 2 + phase] = Some(value ^ (phase == 1));
                        free[c * 2 + phase] = true;
                    }
                }
                AigLang::Pi(var) => {
                    free[c * 2] = true;
                    let name = aig
                        .symbol_input(*var)
                        .cloned()
                        .unwrap_or_else(|| format!("$i{var}"));
                    inputs.push((c * 2, *var, name));
                }
                _ => {}
            }
        }
    }
    inputs.sort_by_key(|(_, var, _)| *var);
    let original_interface=if ps.canonical_wire_only {Some((free.clone(),constants.clone()))} else {None};
    if ps.canonical_phases {
        let mut new_free=vec![false;free.len()];let mut new_constants=vec![None;constants.len()];
        for s in 0..free.len() {
            if free[s]{new_free[canonical[s]]=true;}
            if let Some(v)=constants[s] {
                if new_constants[canonical[s]].is_some_and(|old|old!=v){return Err("conflicting canonical constant".into());}
                new_constants[canonical[s]]=Some(v);
            }
        }
        for (s,_,_) in &mut inputs {*s=canonical[*s];}
        free=new_free;constants=new_constants;
    }
    let outputs: Vec<_> = roots
        .iter()
        .map(|&(pos, root)| {
            (
                canonical[ids[&graph.find(root)] * 2],
                aig.symbol_output(pos)
                    .cloned()
                    .unwrap_or_else(|| format!("$o{pos}")),
            )
        })
        .collect();
    let required: Vec<_> = outputs.iter().map(|(s, _)| *s).collect();
    let mut cut_params=ps.clone();if ps.canonical_wire_only {cut_params.canonical_phases=false;}
    // Full graph still participates in saturation, counting, aliases and care.
    // The ablated optional cut constructor sees one acyclic witness per class;
    // both settings retain the same mandatory original-source physical cuts.
    let witness_nodes: Vec<_> = reach.nodes.iter().cloned().map(|n|vec![n]).collect();
    if ps.protected_exchange {cut_params.functional_cuts=ps.exchange_functional_seed;cut_params.cut_rounds=cut_params.cut_rounds.saturating_sub(1);}
    let mut cuts = enumerate_mapping_cuts(&nodes,&witness_nodes,&ids,graph,tech,&cut_params,&mut stats,&source_pairs);
    if ps.canonical_phases && !ps.canonical_wire_only {cuts=canonical_cut_banks(&cuts,&nodes,|id|ids[&graph.find(id)],&canonical,ps,&mut stats)?;}
    stats.max_cuts_per_class = cuts.iter().map(|c| c.len()).max().unwrap_or(0);
    if ps.strict_budgets && stats.max_cuts_per_class > ps.cuts_per_class {
        return Err(format!(
            "budget: mandatory cuts {} exceed per-class allowance {}",
            stats.max_cuts_per_class, ps.cuts_per_class
        ));
    }
    let (interface_free,interface_constants)=original_interface.as_ref().map(|(f,c)|(&f[..],&c[..])).unwrap_or((&free[..],&constants[..]));
    let physical_aliases=if ps.canonical_wire_only {Some((&canonical[..],&free[..]))} else {None};
    let mut instances = generate(&cuts, interface_free, interface_constants, tech, ps, &mut reach, &mut stats,physical_aliases,None);
    let raw_generated=instances.len();
    if ps.canonical_wire_only {
        let mut merged=Vec::new();let mut index=HashMap::new();
        for mut m in instances {
            for s in &mut m.inputs {*s=canonical[*s];}
            for o in &mut m.outputs {
                for s in &mut o.targets {*s=canonical[*s];}
                o.targets.retain(|s|!free[*s] && !m.inputs.contains(s));
                o.targets.sort_unstable();o.targets.dedup();
                exchange::normalize_sources(o,&canonical);
            }
            if m.outputs.iter().any(|o|!o.targets.is_empty()){
                if ps.protected_exchange {let key=exchange::recorded_key(&m).unwrap_or_else(||instance_key(&m));insert_instance_key(&mut merged,&mut index,m,key);}else{insert_instance(&mut merged,&mut index,m);}
            }
        }
        instances=merged;
    }
    retain_possible_cone(&mut instances, &free, &required);
    stats.generated_instances = instances.len();
    let mut generation_frontier=None;
    if ps.library_timing || ps.strict_budgets {
        let retention = cap_instances_with_frontier(
            &mut instances,
            &free,
            &required,
            ps.max_instances,
            ps.timing_cap,
            ps.strict_budgets,
            ps.timing_witness_pool,
            ps.preserve_shared_trajectory,
            ps.protected_exchange && ps.retain_generation_frontier,
        )?;
        stats.instance_budget_pruned=retention.pruned;generation_frontier=retention.frontier;
        stats.instance_limit_hit |= stats.instance_budget_pruned > 0;
    }

    stats.instances = instances.len();
    stats.bindings = instances
        .iter()
        .flat_map(|m| &m.outputs)
        .map(|o| o.targets.len())
        .sum();
    stats.build_secs = start.elapsed().as_secs_f64();
    let chosen = if ps.protected_exchange {
        exchange::recover(&mut instances,&mut cuts,&free,&required,interface_free,interface_constants,physical_aliases,tech,&mut reach,ps,&mut stats,generation_frontier)?
    } else {extract(&instances, &free, &required, ps, &mut stats)?};
    stats.instances=instances.len();
    stats.cuts=cuts.iter().map(BTreeSet::len).sum();
    stats.max_cuts_per_class=cuts.iter().map(BTreeSet::len).max().unwrap_or(0);
    stats.bindings=instances.iter().flat_map(|m|&m.outputs).map(|o|o.targets.len()).sum();
    stats.exchange_source_bindings=instances.iter().flat_map(|m|&m.outputs).map(|o|o.sources.len()).sum();
    let (area, delay) = metrics(&instances, &chosen, &free, &required);
    if let Ok(path) = std::env::var("EGM_DIAG_PATH") {
        let (arrival, critical) = critical_arrivals(&instances, &chosen, &free, &required);
        let diag_start=Instant::now();
        let profile=std::path::Path::new(&path).with_extension("profile.json");
        let note=|stage:&str,unique:usize|{let _=std::fs::write(&profile,serde_json::json!({"stage":stage,"elapsed":diag_start.elapsed().as_secs_f64(),"cut_functions":unique,"scope":"read-only reporting after recovery; native process guard unchanged"}).to_string());};
        note("cut_rows",0);
        let mut name_cache=std::collections::BTreeMap::new();let mut cut_rows=Vec::new();
        for (c,cs) in cuts.iter().enumerate(){for cut in cs {
            let k=cut.leaves.len();
            let names=name_cache.entry((k,cut.function)).or_insert_with(||diagnostic_cell_names(tech,k,cut.function));
            cut_rows.push(serde_json::json!({"class":c,"leaves":cut.leaves,"function":cut.function,"direct_cells":&*names,"trivial":cut.leaves==[c]}));
        }}
        note("producer_index",name_cache.len());
        let producer_index=diagnostic_producer_index(&instances,free.len());
        use std::hash::{Hash, Hasher};
        let mut pool_hash=std::collections::hash_map::DefaultHasher::new();
        for m in &instances {instance_key(m).hash(&mut pool_hash);m.area.to_bits().hash(&mut pool_hash);
            for o in &m.outputs {o.targets.hash(&mut pool_hash);}}
        let live: BTreeSet<_> = chosen.iter().flatten().map(|b|b.instance).collect();
        let selected_instances: Vec<_> = live.iter().map(|&i| {
            let m=&instances[i];serde_json::json!({"instance":i,"cell":m.cell,"area":m.area,
                "inputs":m.inputs,"input_pins":m.input_pins,"outputs":m.outputs.iter().enumerate()
                .map(|(pin,o)|serde_json::json!({"pin":pin,"name":o.name,"function":o.function,
                    "delays":o.delays,"binding_sources":o.sources.iter().map(|s|s.diagnostic()).collect::<Vec<_>>(),"selected_targets":o.targets.iter().filter(|&&s|chosen[s].is_some_and(|b|b.instance==i && b.pin==pin)).collect::<Vec<_>>()})).collect::<Vec<_>>()})
        }).collect();
        let graph_rows: Vec<_> = nodes.iter().enumerate().map(|(c,ns)|serde_json::json!({"class":c,"nodes":format!("{:?}",ns)})).collect();
        let critical_rows: Vec<_> = critical.iter().enumerate().filter(|(_, yes)| **yes)
            .map(|(s,_)| {
                let producers:Vec<_>=producer_index[s].iter().map(|&(i,pin)|{
                    let m=&instances[i];let o=&m.outputs[pin];
                    serde_json::json!({"instance":i,"pin":pin,"cell":m.cell,"inputs":m.inputs,"delays":o.delays,"function":o.function,"area":m.area})
                }).collect();
                serde_json::json!({"signal":s,"arrival":arrival[s],"selected":chosen[s].map(|b| (b.instance,b.pin)),"producers":producers})
            }).collect();
        note("serialize",name_cache.len());
        // Move already built JSON arrays into the report once. json!(&rows)
        // would clone every Value, and to_vec would allocate the full byte copy.
        let mut report=serde_json::json!({"roots":required,
            "inputs":inputs,"canonical_signals":canonical,"raw_generated_instances":raw_generated,"canonical_wire_only":ps.canonical_wire_only,"retained_pool_fingerprint":format!("{:016x}",pool_hash.finish()),
            "arrivals":arrival,"free":free,"area":area,"delay":delay,"slack_local_scored":stats.slack_local_scored,"slack_local_trials":stats.slack_local_trials,"slack_local_accepted":stats.slack_local_accepted,"slack_attempts":stats.slack_attempts,"slack_valid":stats.slack_valid,"slack_accepted":stats.slack_accepted,"timing_model":"block pin; unloaded"});
        for (key,rows) in [("cuts",cut_rows),("selected_instances",selected_instances),("egraph_nodes",graph_rows),("critical_signals",critical_rows)] {
            report.as_object_mut().unwrap().insert(key.into(),serde_json::Value::Array(rows));
        }
        let file=std::fs::File::create(&path).map_err(|e|e.to_string())?;
        let mut writer=std::io::BufWriter::new(file);
        serde_json::to_writer(&mut writer,&report).map_err(|e|e.to_string())?;
        writer.flush().map_err(|e|e.to_string())?;
        note("finished",name_cache.len());
    }

    let live: BTreeSet<_> = chosen.iter().flatten().map(|b| b.instance).collect();
    for i in live {
        if instances[i].outputs.len() > 1 {
            stats.selected_multioutput += 1;
            let pins: HashSet<_> = chosen
                .iter()
                .flatten()
                .filter(|b| b.instance == i)
                .map(|b| b.pin)
                .collect();
            stats.selected_joint_multioutput += usize::from(pins.len() > 1);
        }
    }
    Ok(MappedGraph {
        instances,
        chosen,
        inputs,
        outputs,
        constants,
        area,
        delay,
        stats,
    })
}

fn identifier(name: &str) -> String {
    let valid = name.bytes().enumerate().all(|(i, b)| {
        b.is_ascii_alphabetic() || b == b'_' || (i > 0 && (b.is_ascii_digit() || b == b'$'))
    });
    let keyword = matches!(
        name,
        "module"
            | "endmodule"
            | "input"
            | "output"
            | "wire"
            | "assign"
            | "reg"
            | "always"
            | "and"
            | "or"
            | "not"
            | "buf"
            | "xor"
            | "nand"
            | "nor"
    );
    if !name.is_empty() && valid && !keyword {
        name.into()
    } else {
        format!("\\{name} ")
    }
}

/// Functional library models preserve all outputs and their common physical input namespace.
/// ABC's single-output genlib reader is insufficient for verifying grouped multi-output cells.
pub fn write_library_verilog<W: Write>(tech: &TechLib, mut w: W) -> std::io::Result<()> {
    fn expression(expr: &Expr) -> String {
        match expr {
            Expr::Const(v) => format!("1'b{}", u8::from(*v)),
            Expr::Var(v) if v == "CONST0" => "1'b0".into(),
            Expr::Var(v) if v == "CONST1" => "1'b1".into(),
            Expr::Var(v) => identifier(v),
            Expr::Not(a) => format!("~({})", expression(a)),
            Expr::And(a, b) => format!("({}) & ({})", expression(a), expression(b)),
            Expr::Or(a, b) => format!("({}) | ({})", expression(a), expression(b)),
        }
    }
    let mut groups: BTreeMap<&str, Vec<&Cell>> = BTreeMap::new();
    for cell in &tech.cells {
        groups.entry(&cell.name).or_default().push(cell);
    }
    for (name, cells) in groups {
        let pins: BTreeSet<_> = cells.iter().flat_map(|c| &c.vars).collect();
        let ports: Vec<_> = pins
            .iter()
            .map(|p| identifier(p))
            .chain(cells.iter().map(|c| identifier(&c.out_pin)))
            .collect();
        writeln!(w, "module {}({});", identifier(name), ports.join(", "))?;
        for pin in &pins {
            writeln!(w, "  input {};", identifier(pin))?;
        }
        for cell in &cells {
            writeln!(w, "  output {};", identifier(&cell.out_pin))?;
        }
        for cell in &cells {
            writeln!(
                w,
                "  assign {} = {};",
                identifier(&cell.out_pin),
                expression(&cell.expr)
            )?;
        }
        writeln!(w, "endmodule")?;
    }
    Ok(())
}

/// Flatten explicit cell models with Yosys and prove equivalence against the original AIG in ABC.
/// An unavailable tool, failed synthesis, or unknown checker outcome is an error, never success.
pub fn verify_with_models(
    tech: &TechLib,
    golden: &std::path::Path,
    netlist: &std::path::Path,
) -> Result<bool, String> {
    let mut models = Vec::new();
    write_library_verilog(tech, &mut models).map_err(|e| e.to_string())?;
    verify_with_verilog_models(
        &String::from_utf8(models).map_err(|e| e.to_string())?,
        golden,
        netlist,
    )
}

/// Check a netlist against a golden AIG using explicit functional Verilog models.
/// This also supports parameterized LUT models for cross-target baseline adapters.
pub fn verify_with_verilog_models(
    model_source: &str,
    golden: &std::path::Path,
    netlist: &std::path::Path,
) -> Result<bool, String> {
    use std::process::Command;
    use std::time::{SystemTime, UNIX_EPOCH};
    fn quoted(path: &std::path::Path) -> String {
        format!(
            "\"{}\"",
            path.to_string_lossy()
                .replace('\\', "\\\\")
                .replace('"', "\\\"")
        )
    }
    fn token(name: &str) -> String {
        name.trim_end().to_string()
    }
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let directory = std::env::temp_dir().join(format!("egm_cec_{}_{stamp}", std::process::id()));
    std::fs::create_dir(&directory).map_err(|e| e.to_string())?;
    let result = (|| {
        let models = directory.join("cells.v");
        let implementation = directory.join("implementation.aig");
        let normalized_golden = directory.join("golden.aig");
        std::fs::write(&models, model_source).map_err(|e| e.to_string())?;
        let mapped = std::fs::canonicalize(netlist).map_err(|e| e.to_string())?;
        let golden = std::fs::canonicalize(golden).map_err(|e| e.to_string())?;
        let original = Aig::from_aiger(golden.to_str().ok_or("invalid golden path")?);
        let mut source_renames = String::new();
        let mut mapped_renames = String::new();
        for node in original.node_indices() {
            let (name, fallback, canonical) = match original.node_type(node) {
                crate::aig_graph::AigNode::Input(var) => (
                    original.symbol_input(var),
                    format!("$i{var}"),
                    format!("cec_{stamp}_i{var}"),
                ),
                crate::aig_graph::AigNode::Output(pos) => (
                    original.symbol_output(pos),
                    format!("$o{pos}"),
                    format!("cec_{stamp}_o{pos}"),
                ),
                _ => continue,
            };
            let name = name.unwrap_or(&fallback);
            // rename keeps quote characters. Restrict data tokens so a source symbol cannot
            // become a Yosys script delimiter; unsupported names fail verification explicitly.
            if !name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "_$[].:/-".contains(c))
            {
                return Err("unsupported port identifier in Yosys verification".into());
            }
            // AIGER's unnamed ports are Yosys private identifiers; Verilog ports are public.
            let source_name = if name.starts_with('$') {
                name.clone()
            } else {
                identifier(name)
            };
            source_renames.push_str(&format!("rename {} {canonical}; ", token(&source_name)));
            mapped_renames.push_str(&format!(
                "rename {} {canonical}; ",
                token(&identifier(name))
            ));
        }
        let yosys = std::env::var("EGM_YOSYS").unwrap_or_else(|_| "yosys".into());
        // Canonical public names survive AIGER export even for originally unnamed ports. CEC then
        // matches by name, so Yosys port reordering cannot create spurious functional mismatches.
        let script = format!(
            "read_aiger -module_name golden {}; cd golden; {source_renames}cd ..; write_aiger -symbols {}; design -reset; read_verilog {} {}; hierarchy -check -top mapped; cd mapped; {mapped_renames}cd ..; proc; flatten; techmap; opt; aigmap; write_aiger -symbols {}",
            quoted(&golden),
            quoted(&normalized_golden),
            quoted(&models),
            quoted(&mapped),
            quoted(&implementation)
        );
        // Preserve the exact Yosys command text while avoiding OS per-argument
        // limits for circuits with thousands of named ports.
        let synthesis_script = directory.join("cec.ys");
        std::fs::write(&synthesis_script, &script).map_err(|e| e.to_string())?;
        let synthesis = Command::new(yosys)
            .args(["-Q", "-T", "-s"])
            .arg(&synthesis_script)
            .output()
            .map_err(|e| format!("cannot run Yosys: {e}"))?;
        if !synthesis.status.success() || !implementation.is_file() {
            return Err(format!(
                "Yosys failed: {}",
                String::from_utf8_lossy(&synthesis.stderr)
            ));
        }
        let abc = std::env::var("BENCH_RS_EMAP_ABC").unwrap_or_else(|_| "abc".into());
        let script = format!(
            "cec {} {}",
            quoted(&normalized_golden),
            quoted(&implementation)
        );
        let check = Command::new(abc)
            .args(["-c", &script])
            .output()
            .map_err(|e| format!("cannot run ABC: {e}"))?;
        let output = format!(
            "{}{}",
            String::from_utf8_lossy(&check.stdout),
            String::from_utf8_lossy(&check.stderr)
        );
        if output.contains("Networks are NOT EQUIVALENT")
            || output
                .to_lowercase()
                .contains("networks are not equivalent")
        {
            return Ok(false);
        }
        if check.status.success() && output.to_lowercase().contains("networks are equivalent") {
            return Ok(true);
        }
        Err(format!("ABC produced no equivalence result: {output}"))
    })();
    let _ = std::fs::remove_dir_all(directory);
    result
}

impl MappedGraph {
    /// Evaluate the actual selected physical netlist, useful for exhaustive small-circuit checks.
    pub fn evaluate(&self, inputs: &HashMap<u32, bool>) -> Result<Vec<bool>, String> {
        fn eval(
            s: usize,
            graph: &MappedGraph,
            values: &mut [Option<bool>],
            active: &mut [bool],
        ) -> Result<bool, String> {
            if let Some(value) = values[s] {
                return Ok(value);
            }
            if active[s] {
                return Err("cycle in selected netlist".into());
            }
            active[s] = true;
            let binding = graph.chosen[s].ok_or_else(|| format!("undriven signal {s}"))?;
            let m = &graph.instances[binding.instance];
            let mut assignment = 0;
            for (i, &input) in m.inputs.iter().enumerate() {
                assignment |= usize::from(eval(input, graph, values, active)?) << i;
            }
            let value = (m.outputs[binding.pin].function >> assignment) & 1 != 0;
            values[s] = Some(value);
            active[s] = false;
            Ok(value)
        }
        let mut values = self.constants.clone();
        for &(s, var, _) in &self.inputs {
            values[s] = Some(
                *inputs
                    .get(&var)
                    .ok_or_else(|| format!("missing input {var}"))?,
            );
        }
        let mut active = vec![false; values.len()];
        self.outputs
            .iter()
            .map(|(s, _)| eval(*s, self, &mut values, &mut active))
            .collect()
    }

    /// Unused physical outputs are explicitly left open. Equivalent bound signals share one net.
    pub fn write_verilog<W: Write>(&self, mut w: W, module: &str) -> std::io::Result<()> {
        let ports: Vec<_> = self
            .inputs
            .iter()
            .map(|(_, _, name)| identifier(name))
            .chain(self.outputs.iter().map(|(_, name)| identifier(name)))
            .collect();
        if ports.iter().collect::<HashSet<_>>().len() != ports.len()
            || self
                .inputs
                .iter()
                .map(|(_, _, n)| n)
                .chain(self.outputs.iter().map(|(_, n)| n))
                .any(|n| n.is_empty() || n.chars().any(char::is_whitespace))
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "duplicate or unrepresentable Verilog port names",
            ));
        }
        let mut prefix = "__egm_".to_string();
        while ports.iter().any(|name| name.starts_with(&prefix)) {
            prefix.push('_');
        }
        let signal_name = |s| format!("{prefix}s{s}");
        writeln!(w, "module {}({});", identifier(module), ports.join(", "))?;
        for (_, _, name) in &self.inputs {
            writeln!(w, "  input {};", identifier(name))?;
        }
        for (_, name) in &self.outputs {
            writeln!(w, "  output {};", identifier(name))?;
        }
        let mut nets: Vec<_> = (0..self.chosen.len()).map(&signal_name).collect();
        for (s, value) in self.constants.iter().enumerate() {
            if let Some(value) = value {
                nets[s] = format!("1'b{}", u8::from(*value));
            }
        }
        for &(s, _, ref name) in &self.inputs {
            nets[s] = identifier(name);
        }
        let mut bound: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
        for (s, b) in self.chosen.iter().enumerate() {
            if let Some(b) = b {
                bound.entry((b.instance, b.pin)).or_default().push(s);
            }
        }
        let mut wires = BTreeSet::new();
        for signals in bound.values() {
            let net = signal_name(signals[0]);
            wires.insert(net.clone());
            for &s in signals {
                nets[s] = net.clone();
            }
        }
        for wire in wires {
            writeln!(w, "  wire {wire};")?;
        }
        let live: BTreeSet<_> = bound.keys().map(|(i, _)| *i).collect();
        for i in live {
            let m = &self.instances[i];
            let mut conns: Vec<_> = m
                .input_pins
                .iter()
                .zip(&m.inputs)
                .map(|(pin, &s)| format!(".{}({})", identifier(pin), nets[s]))
                .collect();
            for (pin, output) in m.outputs.iter().enumerate() {
                let net = bound
                    .get(&(i, pin))
                    .map(|signals| nets[signals[0]].as_str())
                    .unwrap_or("");
                conns.push(format!(".{}({net})", identifier(&output.name)));
            }
            writeln!(
                w,
                "  {} {prefix}g{i} ({});",
                identifier(&m.cell),
                conns.join(", ")
            )?;
        }
        for (s, name) in &self.outputs {
            writeln!(w, "  assign {} = {};", identifier(name), nets[*s])?;
        }
        writeln!(w, "endmodule")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig_graph::AigNode;
    use petgraph::graph::NodeIndex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn every_import_route_checks_lifetime_admission_before_add() {
        let mut aig=Aig::default();let a=input(&mut aig,0);let b=input(&mut aig,1);
        let unused=input(&mut aig,2);let ab=and(&mut aig,a,inv(b));
        let out=aig.add_node(AigNode::Output(0));aig.add_edge(ab.0,out,true);
        let _=unused;
        for route in 0..5 {
            for cap in 0..20 {
                let mut budget=crate::egraph_budget::NodeAdmission::new(cap);
                let result=match route {
                    0=>crate::egraph_sat::import_aig_admitted(&aig,&mut budget).map(|(g,r)|(g,r,Vec::new())),
                    1=>regions::import_demanded_admitted(&aig,&mut budget).map(|(g,r)|(g,r,Vec::new())),
                    2=>regions::import_literals_admitted(&aig,&mut budget).map(|(g,r)|(g,r,Vec::new())),
                    3=>regions::import_helpers_admitted(&aig,0,cap,false,false,&mut budget),
                    _=>source_import::import_source_admitted(&aig,0,cap,true,3,None,&mut budget),
                };
                assert!(budget.admitted()<=cap,"route{route},cap{cap}");
                if let Some((g,roots,_))=result {
                    assert!(g.total_number_of_nodes()<=budget.admitted());assert_eq!(roots.len(),1);
                    assert!(g.classes().flat_map(|c|&c.nodes).any(|n|*n==AigLang::Pi(2)),"unused PI must be counted");
                } else {assert_eq!(budget.admitted(),cap,"budget failure must not construct the next node");}
            }
        }
    }
    #[test]
    fn saturation_and_import_share_one_lifetime_ledger() {
        let mut g=egg::EGraph::default();let mut b=crate::egraph_budget::NodeAdmission::new(6);
        let a=b.add(&mut g,AigLang::Pi(0)).unwrap();let c=b.add(&mut g,AigLang::Pi(1)).unwrap();
        let na=b.add(&mut g,AigLang::Not(a)).unwrap();let nc=b.add(&mut g,AigLang::Not(c)).unwrap();
        g.union(a,c);g.rebuild();assert_eq!(g.total_number_of_nodes(),3);assert_eq!(b.admitted(),4);
        let root=b.add(&mut g,AigLang::And([na,nc])).unwrap();g.rebuild();
        let shared=std::sync::Arc::new(std::sync::Mutex::new(b));
        let mut r=vec![egg::rewrite!("expand-bounded"; "(~ ?a)"=>"(& ?a (~ (~ ?a)))")];
        let skipped=regions::reserve_rewrites(&mut r,shared.clone());
        let run=egg::Runner::default().with_egraph(g).with_iter_limit(8).run(&r);
        assert_eq!(shared.lock().unwrap().admitted(),5);
        assert!(skipped.load(std::sync::atomic::Ordering::Relaxed)>0);
        assert_eq!(run.egraph.total_number_of_nodes(),4);let _=root;
    }

    const BASIC: &str = "GATE INV 0.2 Y=!A;\nPIN A INV 1 99 0.2 0 0.2 0\n\
        GATE AND 1 Y=A*B;\nPIN A NONINV 1 99 1 0 1 0\nPIN B NONINV 1 99 1 0 1 0\n\
        GATE BUF 0.1 Y=A;\nPIN A NONINV 1 99 0.1 0 0.1 0\n";

    fn library(extra: &str) -> TechLib {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "egm_test_{}_{}.genlib",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, format!("{BASIC}{extra}")).unwrap();
        let library = TechLib::from_genlib(&path, 6).unwrap();
        std::fs::remove_file(path).unwrap();
        library
    }

    type Sig = (NodeIndex, bool);
    fn inv(s: Sig) -> Sig {
        (s.0, !s.1)
    }
    fn input(aig: &mut Aig, var: u32) -> Sig {
        (aig.add_node(AigNode::Input(var)), false)
    }
    fn and(aig: &mut Aig, a: Sig, b: Sig) -> Sig {
        let node = aig.add_node(AigNode::And);
        aig.add_edge(a.0, node, a.1);
        aig.add_edge(b.0, node, b.1);
        (node, false)
    }
    fn or(aig: &mut Aig, a: Sig, b: Sig) -> Sig {
        inv(and(aig, inv(a), inv(b)))
    }
    fn xor(aig: &mut Aig, a: Sig, b: Sig) -> Sig {
        let x = and(aig, a, inv(b));
        let y = and(aig, inv(a), b);
        or(aig, x, y)
    }
    fn output(aig: &mut Aig, s: Sig, pos: u32) {
        let node = aig.add_node(AigNode::Output(pos));
        aig.add_edge(s.0, node, s.1);
    }
    fn graph() -> Aig {
        let mut g = Aig::default();
        g.add_node(AigNode::Zero);
        g
    }
    fn params() -> MapParams {
        MapParams {
            time_limit_secs: 5.0,
            ..Default::default()
        }
    }
    fn assert_function(g: &MappedGraph, n: usize, f: impl Fn(usize) -> Vec<bool>) {
        for a in 0..1usize << n {
            let inputs = (0..n)
                .map(|i| ((i + 1) as u32, (a >> i) & 1 != 0))
                .collect();
            assert_eq!(g.evaluate(&inputs).unwrap(), f(a), "assignment {a}");
        }
    }



    #[test]
    fn shared_majority_folding_reduces_actual_import_and_preserves_shared_roots() {
      let mut aig=graph();let a=input(&mut aig,1);let b=input(&mut aig,2);let c=input(&mut aig,3);input(&mut aig,4);
      let ab=and(&mut aig,a,b);let neither=and(&mut aig,inv(a),inv(b));let propagate=and(&mut aig,c,inv(neither));let root=and(&mut aig,inv(ab),inv(propagate));output(&mut aig,root,0);
      let old=regions::import_with_helpers(&aig,0,20000).unwrap();let new=source_import::import_with_shared_helpers(&aig,0,6).unwrap();
      assert_eq!(new.0.total_number_of_nodes()+1,old.0.total_number_of_nodes(),"four-AND majority reconvergence should become one actual TT3, with no omitted functions stored elsewhere");
      let tech=library("GATE MAJ 1 Y=(A*B)+(A*C)+(B*C);\nPIN A UNKNOWN 1 99 1 0 1 0\nPIN B UNKNOWN 1 99 1 0 1 0\nPIN C UNKNOWN 1 99 1 0 1 0\n");
      let mut ps=MapParams{shared_majority:true,cone_import:true,strict_budgets:true,literal_auto:true,canonical_phases:true,canonical_wire_only:true,max_inputs:3,cuts_per_class:16,time_limit_secs:0.,..Default::default()};ps.saturation.node_limit=new.0.total_number_of_nodes();ps.saturation.iter_limit=0;
      let mapped=map(&aig,&tech,&ps).unwrap();assert_eq!(mapped.stats.imported_enodes,new.0.total_number_of_nodes());assert_eq!(mapped.inputs.len(),4);assert_function(&mapped,4,|v|vec![(v&7).count_ones()<2]);
      output(&mut aig,ab,1);ps.saturation.node_limit+=1;let shared=map(&aig,&tech,&ps).unwrap();assert_function(&shared,4,|v|vec![(v&7).count_ones()<2,v&3==3]);

    }
    #[test]
    fn exact_majority_folding_all_input_and_output_phases() {
      let tech=library("GATE MAJ 1 Y=(A*B)+(A*C)+(B*C);\nPIN A UNKNOWN 1 99 1 0 1 0\nPIN B UNKNOWN 1 99 1 0 1 0\nPIN C UNKNOWN 1 99 1 0 1 0\n");
      for xor_propagate in [false,true] {for phase in 0..16 {let mut aig=graph();let a=input(&mut aig,1);let b=input(&mut aig,2);let c=input(&mut aig,3);input(&mut aig,4);let a=if phase&1!=0{inv(a)}else{a};let b=if phase&2!=0{inv(b)}else{b};let c=if phase&4!=0{inv(c)}else{c};let ab=and(&mut aig,a,b);let propagate=if xor_propagate{let l=and(&mut aig,a,inv(b));let r=and(&mut aig,inv(a),b);or(&mut aig,l,r)}else{let neither=and(&mut aig,inv(a),inv(b));inv(neither)};let prop=and(&mut aig,c,propagate);let root=and(&mut aig,inv(ab),inv(prop));output(&mut aig,if phase&8!=0{inv(root)}else{root},0);
       let mut ps=MapParams{shared_majority:true,cone_import:true,strict_budgets:true,literal_auto:true,canonical_phases:true,canonical_wire_only:true,max_inputs:3,cuts_per_class:16,time_limit_secs:0.,..Default::default()};ps.saturation.node_limit=7;ps.saturation.iter_limit=0;let m=map(&aig,&tech,&ps).unwrap();assert!(m.stats.imported_enodes<=7);assert_function(&m,4,|v|vec![(((v^phase)&7).count_ones()<2)^(phase&8!=0)]);
      }}
    }
    #[test]
    fn xor_propagate_majority_uses_one_counted_truth_table() {
      let mut aig=graph();let a=input(&mut aig,1);let b=input(&mut aig,2);let c=input(&mut aig,3);input(&mut aig,4);let ab=and(&mut aig,a,b);let l=and(&mut aig,a,inv(b));let r=and(&mut aig,inv(a),b);let x=or(&mut aig,l,r);let prop=and(&mut aig,c,x);let root=and(&mut aig,inv(ab),inv(prop));output(&mut aig,root,0);
      let old=regions::import_with_helpers(&aig,0,20000).unwrap();let new=source_import::import_with_shared_helpers(&aig,0,6).unwrap();
      assert_eq!(new.0.total_number_of_nodes()+2,old.0.total_number_of_nodes(),"majority carry through XOR propagation must expose the exact common three-input boundary");
      let tech=library("GATE XOR 1 Y=(A*!B)+(!A*B);\nPIN A UNKNOWN 1 99 1 0 1 0\nPIN B UNKNOWN 1 99 1 0 1 0\nGATE MAJ 1 Y=(A*B)+(A*C)+(B*C);\nPIN A UNKNOWN 1 99 1 0 1 0\nPIN B UNKNOWN 1 99 1 0 1 0\nPIN C UNKNOWN 1 99 1 0 1 0\n");let mut ps=MapParams{shared_majority:true,cone_import:true,strict_budgets:true,literal_auto:true,canonical_phases:true,canonical_wire_only:true,max_inputs:3,cuts_per_class:16,time_limit_secs:0.,..Default::default()};ps.saturation.node_limit=new.0.total_number_of_nodes();ps.saturation.iter_limit=0;let m=map(&aig,&tech,&ps).unwrap();assert_function(&m,4,|v|vec![(v&7).count_ones()<2]);
      output(&mut aig,x,1);ps.saturation.node_limit=8;let shared=map(&aig,&tech,&ps).unwrap();assert_function(&shared,4,|v|vec![(v&7).count_ones()<2,(v&1!=0)^(v&2!=0)]);
    }
    #[test]
    fn roomy_import_keeps_old_structure_and_adds_exact_majority_alternative() {
      let mut aig=graph();let a=input(&mut aig,1);let b=input(&mut aig,2);let c=input(&mut aig,3);input(&mut aig,4);let ab=and(&mut aig,a,b);let l=and(&mut aig,a,inv(b));let r=and(&mut aig,inv(a),b);let x=or(&mut aig,l,r);let prop=and(&mut aig,c,x);let root=and(&mut aig,inv(ab),inv(prop));output(&mut aig,root,0);
      let old=regions::import_with_helpers(&aig,0,100).unwrap();let new=source_import::import_with_shared_helpers(&aig,0,100).unwrap();
      assert_eq!(new.0.total_number_of_nodes(),old.0.total_number_of_nodes()+1,"when old source fits, preserve its functions and charge one additional exact alternative instead of deleting its decomposition");
      assert!(new.0[new.1[0].1].nodes.len()>1);assert_eq!(new.0.classes().filter(|c|c.nodes.iter().any(|n|matches!(n,AigLang::Pi(_)))).count(),4);
    }
    #[test]
    fn broad_budget_compaction_preserves_oa_truth_and_shared_roots() {
      let mut aig=graph();let a=input(&mut aig,1);let b=input(&mut aig,2);let c=input(&mut aig,3);input(&mut aig,4);let ab=and(&mut aig,a,b);let ac=and(&mut aig,a,c);let root=and(&mut aig,inv(ab),inv(ac));output(&mut aig,root,0);
      let tech=library("GATE OA 1 Y=A*(B+C);\nPIN A UNKNOWN 1 99 1 0 1 0\nPIN B UNKNOWN 1 99 1 0 1 0\nPIN C UNKNOWN 1 99 1 0 1 0\n");let mut ps=MapParams{shared_majority:true,cone_import:true,strict_budgets:true,literal_auto:true,canonical_phases:true,canonical_wire_only:true,max_inputs:3,cuts_per_class:16,time_limit_secs:0.,..Default::default()};ps.saturation.node_limit=6;ps.saturation.iter_limit=0;let m=map(&aig,&tech,&ps).unwrap();assert!(m.stats.imported_enodes<=6);assert_function(&m,4,|v|vec![!((v&1!=0)&&((v&2!=0)||(v&4!=0)))]);
      output(&mut aig,ab,1);ps.saturation.node_limit=7;let m=map(&aig,&tech,&ps).unwrap();assert_function(&m,4,|v|vec![!((v&1!=0)&&((v&2!=0)||(v&4!=0))),v&3==3]);
      ps.saturation.node_limit=4;assert!(map(&aig,&tech,&ps).is_err(),"an incomplete import cannot masquerade as a bounded complete graph");
    }
    #[test]
    fn four_input_library_source_cut_fits_tight_global_budget() {
      let mut aig=graph();let a=input(&mut aig,1);let b=input(&mut aig,2);let c=input(&mut aig,3);let d=input(&mut aig,4);input(&mut aig,5);let p=and(&mut aig,inv(a),inv(b));let q=and(&mut aig,p,inv(c));let root=and(&mut aig,inv(q),d);output(&mut aig,root,0);
      let tech=library("GATE AO31 1 Y=(A+B+C)*D;\nPIN A UNKNOWN 1 99 1 0 1 0\nPIN B UNKNOWN 1 99 1 0 1 0\nPIN C UNKNOWN 1 99 1 0 1 0\nPIN D UNKNOWN 1 99 1 0 1 0\n");let mut ps=MapParams{shared_majority:true,cone_import:true,strict_budgets:true,literal_auto:true,canonical_phases:true,canonical_wire_only:true,max_inputs:4,cuts_per_class:16,time_limit_secs:0.,..Default::default()};ps.saturation.node_limit=7;ps.saturation.iter_limit=0;
      let m=map(&aig,&tech,&ps).expect("an exact four-input library cut needs one counted root plusfivePIs/zero, not an extra TT3 decomposition");assert_eq!(m.stats.imported_enodes,7);assert_eq!(m.inputs.len(),5);assert_function(&m,5,|v|vec![(v&7)!=0 && v&8!=0]);
      ps.max_inputs=3;assert!(map(&aig,&tech,&ps).is_err(),"three-input policy cannot silently use a four-input logical atom");
    }
    #[test]
    fn exact_four_input_source_tables_cover_all_input_output_phases() {
      let tech=library("GATE AO31 1 Y=(A+B+C)*D;\nPIN A UNKNOWN 1 99 1 0 1 0\nPIN B UNKNOWN 1 99 1 0 1 0\nPIN C UNKNOWN 1 99 1 0 1 0\nPIN D UNKNOWN 1 99 1 0 1 0\n");
      for phase in 0..32 {let mut aig=graph();let a=input(&mut aig,1);let b=input(&mut aig,2);let c=input(&mut aig,3);let d=input(&mut aig,4);input(&mut aig,5);let a=if phase&1!=0{inv(a)}else{a};let b=if phase&2!=0{inv(b)}else{b};let c=if phase&4!=0{inv(c)}else{c};let d=if phase&8!=0{inv(d)}else{d};let p=and(&mut aig,inv(a),inv(b));let q=and(&mut aig,p,inv(c));let r=and(&mut aig,inv(q),d);output(&mut aig,if phase&16!=0{inv(r)}else{r},0);
       let mut ps=MapParams{shared_majority:true,cone_import:true,strict_budgets:true,literal_auto:true,canonical_phases:true,canonical_wire_only:true,max_inputs:4,cuts_per_class:16,time_limit_secs:0.,..Default::default()};ps.saturation.node_limit=7+usize::from(phase&16!=0);ps.saturation.iter_limit=0;let m=map(&aig,&tech,&ps).unwrap();assert!(m.stats.imported_enodes<=ps.saturation.node_limit);assert_eq!(m.inputs.len(),5);assert_function(&m,5,|v|vec![((((v^phase)&7)!=0)&&((v^phase)&8!=0))^(phase&16!=0)]);
      }
    }
    #[test]
    fn exact_functional_cut_finds_existing_shared_non_syntactic_boundary() {
        let mut graph=egg::EGraph::<AigLang,()>::default();
        let a=graph.add(AigLang::Pi(1));let b=graph.add(AigLang::Pi(2));let c=graph.add(AigLang::Pi(3));
        let g=graph.add(AigLang::And([b,c]));
        let root=graph.add(AigLang::Cell("@egm_tt3_80".into(),vec![a,b,c]));graph.rebuild();
        let mut classes:Vec<_>=graph.classes().collect();classes.sort_by_key(|c|usize::from(c.id));
        let ids:HashMap<_,_>=classes.iter().enumerate().map(|(i,c)|(c.id,i)).collect();
        let nodes:Vec<_>=classes.iter().map(|c|c.nodes.clone()).collect();
        let tech=library("");let ps=MapParams{functional_cuts:true,cut_rounds:4,max_inputs:4,cuts_per_class:16,..params()};
        let mut stats=MapStats::default();let cuts=enumerate(&nodes,&ids,&graph,&tech,&ps,&mut stats);
        let expected=Cut{leaves:vec![ids[&a],ids[&g]],function:8};
        let mut control=ps.clone();control.functional_cuts=false;let before=graph.total_number_of_nodes();
        let old=enumerate(&nodes,&ids,&graph,&tech,&control,&mut MapStats::default());
        assert!(!old[ids[&root]].contains(&expected));assert_eq!(graph.total_number_of_nodes(),before);assert!(cuts.iter().all(|b|b.len()<=16));assert!(stats.functional_cut_admitted>0);
        assert!(cuts[ids[&root]].contains(&expected),"missing exact existing shared boundary (a,g=b&c) in compact TT3 root");
    }

    #[test]
    fn functional_cut_physical_shared_output_truth_and_area() {
        let mut a=graph();let x=input(&mut a,1);let y=input(&mut a,2);let z=input(&mut a,3);
        let g=and(&mut a,y,z);let xy=and(&mut a,x,y);let f=and(&mut a,xy,z);output(&mut a,g,0);output(&mut a,f,1);
        let tech=library("");let mut ps=MapParams{functional_cuts:true,cone_import:true,strict_budgets:true,literal_auto:true,canonical_phases:true,canonical_wire_only:true,max_inputs:3,cuts_per_class:16,objective:MapObjective::DelayArea,time_limit_secs:0.,refine_rounds:4,refine_trials:2000,..params()};ps.saturation.node_limit=7;ps.saturation.iter_limit=0;
        let mapped=map(&a,&tech,&ps).unwrap();assert_function(&mapped,3,|x|vec![x&6==6,x&7==7]);
        assert!(mapped.stats.functional_cut_admitted>0);assert!(mapped.stats.imported_enodes<=7);assert!(mapped.stats.max_cuts_per_class<=16);
    }

    #[test]
    fn pin_flow_labels_charge_inverse_and_keep_asymmetric_arcs() {
        let lib=library("GATE FAST 1 Y=A*B;\nPIN A NONINV 1 99 0.5 0 0.5 0\nPIN B NONINV 1 99 8 0 8 0\n");
        let nodes=vec![vec![AigLang::Pi(1)],vec![AigLang::Pi(2)]];
        let mandatory=vec![BTreeSet::new(),BTreeSet::new()];let aliases=vec![0,1,2,3];
        let labels=mandatory_pin_arrivals(&mandatory,&nodes,&aliases,&lib);
        assert_eq!(labels[0],0.);assert!((labels[1]-f64::from(lib.inv_delay)).abs()<1e-9);
        let cut=Cut{leaves:vec![0,1],function:8};let score=pin_arrival_cut_score(&cut,&[100.,f64::INFINITY,0.,f64::INFINITY],&aliases,&lib);
        assert!((score-100.5).abs()<1e-9,"late arrival must use fast real pin, not average/max arc: {score}");
    }
    #[test]
    fn pin_flow_ranking_preserves_fast_internal_four_leaf_cut() {
        let mut aig=graph();let inputs:Vec<_>=(1..=8).map(|v|input(&mut aig,v)).collect();
        let pairs:Vec<_>=inputs.chunks(2).map(|p|and(&mut aig,p[0],p[1])).collect();
        let l=and(&mut aig,pairs[0],pairs[1]);let r=or(&mut aig,pairs[2],pairs[3]);let root=or(&mut aig,l,r);output(&mut aig,root,0);
        let lib=library("GATE AO211 1.5 Y=A*B+C+D;\nPIN A NONINV 1 99 0.1 0 0.1 0\nPIN B NONINV 1 99 0.1 0 0.1 0\nPIN C NONINV 1 99 0.1 0 0.1 0\nPIN D NONINV 1 99 0.1 0 0.1 0\n");
        let ps=MapParams{cut_pin_flow:true,max_inputs:4,cuts_per_class:3,cut_rounds:8,objective:MapObjective::DelayArea,time_limit_secs:0.0,saturation:SatParams{iter_limit:0,..Default::default()},..params()};
        let mapped=map(&aig,&lib,&ps).unwrap();
        assert_function(&mapped,8,|x|vec![((x&3)==3 && (x&12)==12)||(x&48)==48||(x&192)==192]);
        assert!(mapped.delay<1.2,"internal four-leaf compound cut must beat shallow retained boundaries; got {}",mapped.delay);
    }
    #[test]
    fn flat_area_collection_matches_ordered_set_for_shared_pins_and_aliases() {
        let mut state=0x6a09e667f3bcc909u64;let n=251;
        let instances:Vec<_>=(0..33).map(|i|Instance{cell:format!("C{i}"),area:(i+1) as f32*0.13,inputs:vec![],input_pins:vec![],outputs:(0..2).map(|p|OutputPin{sources:Vec::new(),name:format!("Y{p}"),delays:vec![],function:1,targets:(0..n).collect()}).collect()}).collect();
        for _ in 0..300 {let mut chosen=vec![None;n];for v in &mut chosen{state=state.wrapping_mul(6364136223846793005).wrapping_add(1);if state&3!=0{*v=Some(Binding{instance:(state>>3) as usize%33,pin:(state>>9) as usize%2});}}
            let old:f64=chosen.iter().flatten().map(|b|b.instance).collect::<BTreeSet<_>>().into_iter().map(|i|f64::from(instances[i].area)).sum();
            let roots:Vec<_>=chosen.iter().enumerate().filter_map(|(s,b)|b.map(|_|s)).collect();let new=metrics(&instances,&chosen,&vec![false;n],&roots).0;assert_eq!(old.to_bits(),new.to_bits());
        }
    }
    #[test]
    fn import_admission_rejects_before_extra_node_and_counts_unused_pi_and_po_phase(){
        let mut aig=graph();let a=input(&mut aig,1);input(&mut aig,2);input(&mut aig,3);let d=input(&mut aig,4);output(&mut aig,d,0);
        assert!(regions::import_with_helpers(&aig,0,4).is_none(),"five actual nodes must not be materialized under four-node allowance");
        let g=regions::import_with_helpers(&aig,0,5).unwrap();assert_eq!(g.0.total_number_of_nodes(),5);
        output(&mut aig,inv(a),1);assert!(regions::import_with_helpers(&aig,0,5).is_none());assert_eq!(regions::import_with_helpers(&aig,0,6).unwrap().0.total_number_of_nodes(),6);
    }
    #[test]
    fn typed_cone_policy_counts_actual_nodes_and_unused_input_ports(){
        let mut aig=graph();let a=input(&mut aig,1);let b=input(&mut aig,2);let c=input(&mut aig,3);input(&mut aig,4);let ab=and(&mut aig,a,b);let root=and(&mut aig,ab,c);output(&mut aig,root,0);
        let tech=library("GATE AND3_TYPED 1.3 Y=A*B*C;\nPIN A UNKNOWN 1 99 1 0 1 0\nPIN B UNKNOWN 1 99 1 0 1 0\nPIN C UNKNOWN 1 99 1 0 1 0\n");
        let mut ps=MapParams{strict_budgets:true,literal_auto:true,canonical_phases:true,canonical_wire_only:true,max_inputs:3,cuts_per_class:16,time_limit_secs:0.,..Default::default()};ps.saturation.iter_limit=0;ps.saturation.node_limit=6;
        assert!(!ps.cone_import && !ps.scored_incumbent);assert!(map(&aig,&tech,&ps).err().unwrap().contains("node allowance"));
        ps.cone_import=true;let mapped=map(&aig,&tech,&ps).unwrap();assert_eq!(mapped.stats.imported_enodes,6);assert_eq!(mapped.inputs.len(),4);assert_function(&mapped,4,|v|vec![v&7==7]);
        ps.saturation.node_limit=5;assert!(map(&aig,&tech,&ps).err().unwrap().contains("node allowance"));
    }
    #[test]
    fn helper_groups_are_atomic_within_the_global_initial_ceiling(){
        assert_eq!(regions::initial_node_ceiling(20000),18000);let n=usize::MAX;assert_eq!(regions::initial_node_ceiling(n),(n/10)*9+(n%10)*9/10);
        let mut aig=graph();let a=input(&mut aig,1);let b=input(&mut aig,2);let u=and(&mut aig,a,inv(b));let v=and(&mut aig,inv(a),b);let x=or(&mut aig,u,v);output(&mut aig,x,0);
        let one=regions::import_with_helpers(&aig,1,10).unwrap();let two=regions::import_with_helpers(&aig,2,10).unwrap();let tight=regions::import_with_helpers(&aig,2,6).unwrap();
        assert!(one.2.is_empty());assert_eq!(two.0.total_number_of_nodes(),one.0.total_number_of_nodes()+2);assert_eq!(two.2.len(),1);assert_eq!(tight.0.total_number_of_nodes(),one.0.total_number_of_nodes());assert!(tight.2.is_empty());
    }
    #[test]
    fn typed_scored_incumbent_reuses_validation_without_application_replays(){
        let make=|target,function|Instance{cell:format!("single_{target}"),area:8.,inputs:vec![0,1],input_pins:vec!["A".into(),"B".into()],outputs:vec![OutputPin{sources:Vec::new(),name:"Y".into(),delays:vec![1.,1.],function,targets:vec![target]}]};
        let mut joint=make(2,8);joint.cell="joint".into();joint.area=9.;let mut o=joint.outputs[0].clone();o.name="Z".into();o.function=7;o.targets=vec![3];joint.outputs.push(o);let instances=vec![make(2,8),make(3,7),joint];
        let free=vec![true,true,false,false];let roots=vec![2,3];let mut ps=MapParams{objective:MapObjective::DelayArea,strict_budgets:true,flow_worklist:true,slack_flow:true,recover_live_refs:false,recover_cones:false,refine_rounds:1,refine_trials:0,..Default::default()};
        let mut old_stats=MapStats::default();let old=heuristic_cover(&instances,&free,&roots,&ps,&mut old_stats).unwrap();assert_eq!(metrics(&instances,&old,&free,&roots),(16.,1.));
        ps.scored_incumbent=true;let mut stats=MapStats::default();let new=heuristic_cover(&instances,&free,&roots,&ps,&mut stats).unwrap();assert_eq!(metrics(&instances,&new,&free,&roots),(9.,1.));assert_eq!(stats.slack_local_trials,0);assert_eq!(stats.slack_local_scored,old_stats.slack_local_scored);
    }
    #[test]
    fn scored_shared_cover_survives_exhausted_application_window(){
        let make=|target,function|Instance{cell:"fixture".into(),area:8.,inputs:vec![0,1],input_pins:vec!["A".into(),"B".into()],outputs:vec![OutputPin{sources:Vec::new(),name:"Y".into(),delays:vec![1.,1.],function,targets:vec![target]}]};
        let mut joint=make(2,8);joint.cell="joint".into();joint.area=9.;let mut second=joint.outputs[0].clone();second.name="Z".into();second.function=7;second.targets=vec![3];joint.outputs.push(second);
        let instances=vec![make(2,8),make(3,7),joint];
        let free=vec![true,true,false,false];let roots=vec![2,3];let seed=vec![None,None,Some(Binding{instance:0,pin:0}),Some(Binding{instance:1,pin:0})];
        let before=metrics(&instances,&seed,&free,&roots);assert_eq!(before,(16.,1.));
        struct Reset;impl Drop for Reset{fn drop(&mut self){TEST_SKIP_SCORED_APPLICATION.with(|v|v.set(false));}}
        TEST_SKIP_SCORED_APPLICATION.with(|v|v.set(true));let _reset=Reset;
        let ps=MapParams{objective:MapObjective::DelayArea,slack_flow:true,strict_budgets:true,refine_rounds:1,refine_trials:1,..Default::default()};
        let mut old_stats=MapStats::default();let (_,old_value)=slack_area_pass(&instances,&free,&roots,&seed,seed.clone(),before,&ps,&mut old_stats,Instant::now()+std::time::Duration::from_secs(1),false);assert_eq!(old_value,(16.,1.));
        let mut stats=MapStats::default();let (chosen,value)=slack_area_pass(&instances,&free,&roots,&seed,seed.clone(),before,&ps,&mut stats,Instant::now()+std::time::Duration::from_secs(1),true);
        assert!(stats.slack_local_scored>0);assert_eq!(stats.slack_local_trials,0);
        assert_eq!(value,(9.,1.),"a whole-cover validated joint producer was scored but lost when application had no budget");assert_eq!(metrics(&instances,&chosen,&free,&roots),value);
        assert_eq!(stats.slack_local_scored,old_stats.slack_local_scored,"retention adds no candidate scoring");
        let mut late_stats=MapStats::default();let (_,late)=slack_area_pass(&instances,&free,&roots,&seed,seed.clone(),before,&ps,&mut late_stats,Instant::now()-std::time::Duration::from_millis(1),true);assert_eq!(late,before);assert_eq!(late_stats.slack_local_scored,0);
        let mut slower=instances.clone();for pin in &mut slower[2].outputs{pin.delays=vec![2.,2.];}let mut slow_stats=MapStats::default();let (_,slow)=slack_area_pass(&slower,&free,&roots,&seed,seed.clone(),before,&ps,&mut slow_stats,Instant::now()+std::time::Duration::from_secs(1),true);assert_eq!(slow,before,"area may not buy a delay regression");

    }
    #[test]
    fn source_physical_certificates_name_only_counted_helper_functions(){
      let mut aig=graph();let a=input(&mut aig,1);let b=input(&mut aig,2);let c=input(&mut aig,3);input(&mut aig,4);let ab=and(&mut aig,a,b);let root=and(&mut aig,ab,c);output(&mut aig,root,0);
      let (small,_,missing)=regions::import_with_source(&aig,false).unwrap();assert!(missing.is_empty());
      let (g,_,cert)=regions::import_with_source(&aig,true).unwrap();assert_eq!(g.total_number_of_nodes(),small.total_number_of_nodes()+1);assert_eq!(cert.len(),1);assert_eq!(g[cert[0].root].nodes.len(),1);
      let classes:Vec<_>=g.classes().map(|c|c.id).collect();let ids:HashMap<_,_>=classes.iter().enumerate().map(|(i,&id)|(id,i)).collect();let choices=cheapest_nodes_queued(&g);let nodes:Vec<_>=classes.iter().map(|id|choices[id].clone().map_children(|id|g.find(id))).collect();let reach=Reachability{nodes,ids:ids.clone(),cache:HashMap::new()};
      for a in 0..16 {let mut memo=vec![None;classes.len()];for p in &cert{let root=reach.eval(ids[&p.root],&[1,2,3,4],a,&mut memo);let left=reach.eval(ids[&p.left],&[1,2,3,4],a,&mut memo)^p.left_neg;let right=reach.eval(ids[&p.right],&[1,2,3,4],a,&mut memo)^p.right_neg;assert_eq!(root,left&right);}}
      let banks:Vec<_>=g.classes().map(|c|c.nodes.clone()).collect();let mut ps=MapParams::default();ps.cuts_per_class=3;let mut stats=MapStats::default();let cuts=enumerate_with_source(&banks,&ids,&g,&library(""),&ps,&mut stats,&cert);assert_eq!(cuts[ids[&cert[0].root]].len(),3);assert!(cuts.iter().all(|c|c.len()<=3));
    }
    #[test]
    fn exact_tt3_import_truth_for_every_edge_phase_with_unused_inputs(){
      let tech=library("GATE AO 1.3 Y=(A*B)+C;\nPIN A UNKNOWN 1 99 1 0 1 0\nPIN B UNKNOWN 1 99 1 0 1 0\nPIN C UNKNOWN 1 99 1 0 1 0\nGATE AND3 1.3 Y=A*B*C;\nPIN A UNKNOWN 1 99 1 0 1 0\nPIN B UNKNOWN 1 99 1 0 1 0\nPIN C UNKNOWN 1 99 1 0 1 0\n");
      for phase in 0..16{let mut aig=graph();let a=input(&mut aig,1);let b=input(&mut aig,2);let c=input(&mut aig,3);input(&mut aig,4);let u=and(&mut aig,if phase&1!=0{inv(a)}else{a},if phase&2!=0{inv(b)}else{b});let v=and(&mut aig,if phase&4!=0{inv(u)}else{u},if phase&8!=0{inv(c)}else{c});output(&mut aig,v,0);
       let (g,_)=regions::import_xors(&aig).unwrap();assert!(g.classes().any(|c|c.nodes.iter().any(|n|tt3_inputs(n).is_some())));let mut ps=MapParams{cone_import:true,strict_budgets:true,literal_auto:true,canonical_phases:true,canonical_wire_only:true,max_inputs:3,cuts_per_class:16,time_limit_secs:0.,..Default::default()};ps.saturation.node_limit=6;ps.saturation.iter_limit=0;let mapped=map(&aig,&tech,&ps).unwrap();assert_function(&mapped,4,|a|{let x=(a&1!=0)^(phase&1!=0);let y=(a&2!=0)^(phase&2!=0);let z=(a&4!=0)^(phase&8!=0);vec![((x&y)^(phase&4!=0))&z]});
      }
    }
    #[test]
    fn exact_xor_contraction_keeps_shared_internal_outputs_and_all_input_phases(){
      let tech=library("GATE XOR 2 Y=(A*!B)+(!A*B);\nPIN A UNKNOWN 1 99 1 0 1 0\nPIN B UNKNOWN 1 99 1 0 1 0\n");
      for phase in 0..4 {let mut aig=graph();let a=input(&mut aig,1);let b=input(&mut aig,2);input(&mut aig,3);let x=if phase&1!=0{inv(a)}else{a};let y=if phase&2!=0{inv(b)}else{b};let u=and(&mut aig,x,inv(y));let v=and(&mut aig,inv(x),y);let root=or(&mut aig,u,v);output(&mut aig,root,0);output(&mut aig,u,1);
       let (g,_)=regions::import_xors(&aig).unwrap();assert!(g.classes().any(|c|c.nodes.iter().any(|n|xor_inputs(n).is_some())));assert_eq!(g.classes().filter(|c|c.nodes.iter().any(|n|matches!(n,AigLang::Pi(_)))).count(),3);
       let mut ps=MapParams{cone_import:true,strict_budgets:true,literal_auto:true,canonical_phases:true,canonical_wire_only:true,max_inputs:3,cuts_per_class:16,time_limit_secs:0.,..Default::default()};ps.saturation.node_limit=7;ps.saturation.iter_limit=0;let mapped=map(&aig,&tech,&ps).unwrap();assert_function(&mapped,3,|a|{let x=(a&1!=0)^(phase&1!=0);let y=(a&2!=0)^(phase&2!=0);vec![x^y,x&!y]});
      }
    }
    #[test]
    fn structural_import_policy_preserves_all_live_gates_and_global_node_cap() {
        let mut aig=graph();let a=input(&mut aig,1);let b=input(&mut aig,2);let c=input(&mut aig,3);input(&mut aig,4);
        let ab=and(&mut aig,a,inv(b));let abc=and(&mut aig,ab,c);output(&mut aig,inv(abc),0);
        let raw=regions::import_demanded(&aig).unwrap().0.total_number_of_nodes();
        let compact=regions::import_literals(&aig).unwrap().0.total_number_of_nodes();
        assert!(raw>compact);assert!(regions::ordinary_node_upper(&aig).unwrap()>=raw);
        let mut ps=MapParams{max_inputs:2,cuts_per_class:16,strict_budgets:true,literal_auto:true,canonical_phases:true,canonical_wire_only:true,time_limit_secs:0.0,..Default::default()};
        ps.saturation.node_limit=compact;ps.saturation.iter_limit=0;
        let mapped=map(&aig,&library(""),&ps).unwrap();assert!(mapped.stats.enodes<=compact);assert_eq!(mapped.inputs.len(),4);
        assert_function(&mapped,4,|v|vec![!(v&1!=0 && v&2==0 && v&4!=0)]);
        ps.saturation.node_limit=compact-1;assert!(map(&aig,&library(""),&ps).err().unwrap().contains("node allowance"));
    }
    #[test]
    fn bitwise_pin_dependence_matches_assignment_enumeration() {
        let mut state=0x243f6a8885a308d3u64;
        for n in 1..=6 {for _ in 0..10000 {
            state=state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let pin=OutputPin{sources:Vec::new(),name:"Y".into(),delays:vec![0.0;n],function:state & mask(n),targets:vec![]};
            for input in 0..n {let old=(0..1usize<<n).any(|a|((pin.function>>a)^(pin.function>>(a^(1<<input))))&1!=0);assert_eq!(pin.has_arc(input),old);}
        }}
    }
    #[test]
    fn fast_area_recovery_preserves_joint_outputs_truth_and_pin_timing() {
        let tech=library("GATE pair 0.3 Y=A*B;\nPIN A NONINV 1 999 1 0 1 0\nPIN B NONINV 1 999 1 0 1 0\nGATE pair 0.3 Z=!(A*B);\nPIN A INV 1 999 9 0 9 0\nPIN B INV 1 999 9 0 9 0\n");
        let mut aig=graph();let a=input(&mut aig,1);let b=input(&mut aig,2);let x=and(&mut aig,a,b);output(&mut aig,x,0);output(&mut aig,inv(x),1);
        for literal in [false,true] {
            let ps=MapParams{max_inputs:2,cuts_per_class:16,strict_budgets:true,canonical_phases:true,canonical_wire_only:true,literal_import:literal,objective:MapObjective::DelayArea,time_limit_secs:0.0,slack_flow:true,fast_area_scan:true,area_time_fraction:0.35,..Default::default()};
            let mapped=map(&aig,&tech,&ps).unwrap();
            assert_function(&mapped,2,|v|vec![v==3,v!=3]);
            assert!(mapped.delay<9.0,"slow secondary pin cannot enter the fastest cover");
        }
    }
    #[test]
    fn recovery_partition_reserves_area_without_resetting_global_time() {
        let now=Instant::now();let global=now+std::time::Duration::from_secs(2);
        let local=regions::phase_deadline(now,global,0.65);
        assert_eq!(global.duration_since(local),std::time::Duration::from_millis(700));
        assert_eq!(regions::phase_deadline(now,global,1.0),global);
        assert_eq!(regions::phase_deadline(now,global,0.0),now);
    }
    #[test]
    fn literal_input_phases_preserve_all_truth_rows_and_output_polarities() {
        let lib=library("");
        for phases in 0..4 {
            for same in [false,true] {
                let mut aig=graph();let a=input(&mut aig,1);let b=input(&mut aig,2);
                let b=if same {a} else {b};
                let lhs=if phases&1!=0 {inv(a)} else {a};
                let rhs=if phases&2!=0 {inv(b)} else {b};
                let root=and(&mut aig,lhs,rhs);output(&mut aig,root,0);output(&mut aig,inv(root),1);
                let ps=MapParams{literal_import:true,strict_budgets:true,region_classes:256,
                    time_limit_secs:0.,..Default::default()};
                let result=map(&aig,&lib,&ps).unwrap();
                assert_function(&result,2,|bits| {
                    let x=(bits&1!=0)^(phases&1!=0);
                    let y=(bits&(if same {1} else {2})!=0)^(phases&2!=0);
                    vec![x&y,!(x&y)]
                });
            }
        }
    }

    #[test]
    fn literal_import_counts_every_live_and_and_preserves_strict_node_limit() {
        let lib=library("");let mut aig=graph();let mut root=input(&mut aig,1);
        for v in 2..=41 {let b=input(&mut aig,v);root=and(&mut aig,inv(root),b);}
        output(&mut aig,root,0);
        let (compact,_)=regions::import_literals(&aig).unwrap();
        let (expanded,_)=regions::import_demanded(&aig).unwrap();
        assert_eq!(compact.total_number_of_nodes(),82); // 41 PI +40 live AND +zero.
        assert!(expanded.total_number_of_nodes()>100);
        let mut ps=MapParams{literal_import:false,strict_budgets:true,region_classes:256,
            time_limit_secs:0.,..Default::default()};ps.saturation.node_limit=100;
        assert!(map(&aig,&lib,&ps).err().unwrap().contains("cannot import AIG within node allowance"));
        ps.literal_import=true;let result=map(&aig,&lib,&ps).unwrap();
        assert!(result.stats.enodes<=100);
        for pattern in 0..8u32 {
            let inputs:HashMap<_,_>=(1..=41).map(|v|(v,((v*13+pattern*7)%11)<6)).collect();
            let mut expected=inputs[&1];for v in 2..=41 {expected=!expected & inputs[&v];}
            assert_eq!(result.evaluate(&inputs).unwrap(),vec![expected]);
        }
    }

    #[test]
    fn or_factoring_exposes_a_smaller_two_input_cover() {
        let lib = library("");
        let mut aig = graph();
        let a = input(&mut aig, 1);
        let b = input(&mut aig, 2);
        let c = input(&mut aig, 3);
        let ab = and(&mut aig, a, b);
        let ac = and(&mut aig, a, c);
        let root = or(&mut aig, ab, ac);
        output(&mut aig, root, 0);
        let mut ps = MapParams {
            max_inputs: 2,
            time_limit_secs: 0.0,
            ..Default::default()
        };
        let original = map(&aig, &lib, &ps).unwrap();
        ps.boolean_rules = 1;
        let factored = map(&aig, &lib, &ps).unwrap();
        assert!(factored.area < original.area - 0.5);
        assert_function(&factored, 3, |a| vec![a & 1 != 0 && a & 6 != 0]);
        ps.boolean_rules = 2;
        let associative = map(&aig, &lib, &ps).unwrap();
        assert_function(&associative, 3, |a| vec![a & 1 != 0 && a & 6 != 0]);
    }

    #[test]
    fn three_disjoint_output_supports_need_a_cumulative_boundary() {
        let lib = library(
            "GATE TRIPLE 0.4 Y=!A;\nPIN A INV 1 99 1 0 1 0\n\
             GATE TRIPLE 0.4 Z=!B;\nPIN B INV 1 99 2 0 2 0\n\
             GATE TRIPLE 0.4 W=!C;\nPIN C INV 1 99 3 0 3 0\n",
        );
        let mut aig = graph();
        for i in 1..=3 {
            let a = input(&mut aig, i);
            output(&mut aig, inv(a), i - 1);
        }
        let pair_only = map(
            &aig,
            &lib,
            &MapParams {
                max_inputs: 3,
                multioutput_cumulative: false,
                time_limit_secs: 0.0,
                ..params()
            },
        )
        .unwrap();
        assert_function(&pair_only, 3, |a| vec![a & 1 == 0, a & 2 == 0, a & 4 == 0]);
        assert!((pair_only.area - 0.6).abs() < 1e-6);
        for time_limit_secs in [0.0, 5.0] {
            let mapped = map(
                &aig,
                &lib,
                &MapParams {
                    max_inputs: 3,
                    time_limit_secs,
                    ..params()
                },
            )
            .unwrap();
            assert_function(&mapped, 3, |a| vec![a & 1 == 0, a & 2 == 0, a & 4 == 0]);
            assert!((mapped.area - 0.4).abs() < 1e-6, "area {}", mapped.area);
            assert_eq!(mapped.stats.selected_joint_multioutput, 1);
            assert!((mapped.delay - 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn disjoint_output_supports_share_one_physical_cell() {
        let lib = library(
            "GATE DUAL 1.5 Y=A*B;\nPIN A NONINV 1 99 1 0 1 0\nPIN B NONINV 1 99 1 0 1 0\n\
             GATE DUAL 1.5 Z=C*D;\nPIN C NONINV 1 99 2 0 2 0\nPIN D NONINV 1 99 2 0 2 0\n",
        );
        let mut aig = graph();
        let a = input(&mut aig, 1);
        let b = input(&mut aig, 2);
        let c = input(&mut aig, 3);
        let d = input(&mut aig, 4);
        let ab = and(&mut aig, a, b);
        let cd = and(&mut aig, c, d);
        output(&mut aig, ab, 0);
        output(&mut aig, cd, 1);
        for time_limit_secs in [0.0, 5.0] {
            let mapped = map(
                &aig,
                &lib,
                &MapParams {
                    max_inputs: 4,
                    time_limit_secs,
                    ..params()
                },
            )
            .unwrap();
            assert_function(&mapped, 4, |a| vec![a & 3 == 3, a & 12 == 12]);
            assert!((mapped.area - 1.5).abs() < 1e-6, "area {}", mapped.area);
            assert_eq!(mapped.stats.selected_joint_multioutput, 1);
            assert!((mapped.delay - 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn full_adder_is_one_instance_with_distinct_output_delays() {
        let lib = library(
            "GATE FA 2 S=A*!B*!C+!A*B*!C+!A*!B*C+A*B*C;\n\
            PIN A UNKNOWN 1 99 3 0 3 0\nPIN B UNKNOWN 1 99 3 0 3 0\nPIN C UNKNOWN 1 99 3 0 3 0\n\
            GATE FA 2 CO=A*B+A*C+B*C;\n\
            PIN A UNKNOWN 1 99 7 0 7 0\nPIN B UNKNOWN 1 99 7 0 7 0\nPIN C UNKNOWN 1 99 7 0 7 0\n",
        );
        let mut aig = graph();
        let a = input(&mut aig, 1);
        let b = input(&mut aig, 2);
        let c = input(&mut aig, 3);
        let ab = xor(&mut aig, a, b);
        let sum = xor(&mut aig, ab, c);
        let x = and(&mut aig, a, b);
        let y = and(&mut aig, a, c);
        let z = and(&mut aig, b, c);
        let xy = or(&mut aig, x, y);
        let carry = or(&mut aig, xy, z);
        output(&mut aig, sum, 0);
        output(&mut aig, carry, 1);
        let mapped = map(&aig, &lib, &params()).unwrap();
        assert_function(&mapped, 3, |a| {
            vec![a.count_ones() % 2 == 1, a.count_ones() >= 2]
        });
        assert!(
            (mapped.area - 2.0).abs() < 1e-6,
            "area {}, stats {:?}, FA candidates {}",
            mapped.area,
            mapped.stats,
            mapped.instances.iter().filter(|m| m.cell == "FA").count()
        );
        assert!((mapped.delay - 7.0).abs() < 1e-6, "delay {}", mapped.delay);
        assert_eq!(mapped.stats.selected_joint_multioutput, 1);
        let mut netlist = Vec::new();
        mapped.write_verilog(&mut netlist, "test").unwrap();
        let text = String::from_utf8(netlist).unwrap();
        assert_eq!(text.matches("FA __egm_g").count(), 1);
        assert!(text.contains(".S(") && text.contains(".CO("));
    }

    #[test]
    fn multioutput_unifies_different_supports_and_can_leave_pins_open() {
        let lib = library(
            "GATE PAIR 0.5 Y=A*B;\nPIN A NONINV 1 99 1 0 1 0\nPIN B NONINV 1 99 1 0 1 0\n\
            GATE PAIR 0.5 Z=A;\nPIN A NONINV 1 99 2 0 2 0\n",
        );
        let mut aig = graph();
        let a = input(&mut aig, 1);
        let b = input(&mut aig, 2);
        let ab = and(&mut aig, a, b);
        output(&mut aig, ab, 0);
        let mapped = map(&aig, &lib, &params()).unwrap();
        assert_function(&mapped, 2, |a| vec![a == 3]);
        assert!((mapped.area - 0.5).abs() < 1e-6);
        assert_eq!(mapped.stats.selected_multioutput, 1);
        assert_eq!(mapped.stats.selected_joint_multioutput, 0);
        let mut netlist = Vec::new();
        mapped.write_verilog(&mut netlist, "test").unwrap();
        let text = String::from_utf8(netlist).unwrap();
        assert!(text.contains(".A(") && text.contains(".B(") && text.contains(".Z()"));
    }

    #[test]
    fn sdc_removes_an_input_without_global_union() {
        let lib = library("");
        let mut aig = graph();
        let a = input(&mut aig, 1);
        let b = input(&mut aig, 2);
        let c = input(&mut aig, 3);
        let ab = and(&mut aig, a, b);
        let all = and(&mut aig, ab, c);
        let any_ab = or(&mut aig, a, b);
        let any = or(&mut aig, any_ab, c);
        let root = and(&mut aig, all, any);
        output(&mut aig, root, 0);
        // k=2 blocks the exact three-PI realization; correlated internal leaves enable BUF.
        let ps = MapParams {
            max_inputs: 2,
            dc_support: 3,
            ..params()
        };
        let precise = map(
            &aig,
            &lib,
            &MapParams {
                dc_support: 0,
                ..ps.clone()
            },
        )
        .unwrap();
        let mapped = map(&aig, &lib, &ps).unwrap();
        assert_function(&mapped, 3, |a| vec![a == 7]);
        assert!(mapped.stats.dc_with_unreachable > 0 && mapped.stats.dc_bindings > 0);
        assert!(
            mapped.area < precise.area - 1e-5,
            "SDC {} vs exact {}",
            mapped.area,
            precise.area
        );
        assert_eq!(mapped.stats.eclasses, precise.stats.eclasses);
        assert_eq!(mapped.stats.enodes, precise.stats.enodes);
    }

    #[test]
    fn dc_budget_exhaustion_is_conservative_and_zero_solve_budget_has_valid_cover() {
        let mut aig = graph();
        let a = input(&mut aig, 1);
        let b = input(&mut aig, 2);
        let ab = and(&mut aig, a, b);
        output(&mut aig, ab, 0);
        output(&mut aig, inv(ab), 1);
        let ps = MapParams {
            time_limit_secs: 0.0,
            dc_support: 1,
            cut_rounds: 0,
            cuts_per_class: 1,
            ..params()
        };
        let mapped = map(&aig, &library(""), &ps).unwrap();
        assert_function(&mapped, 2, |a| vec![a == 3, a != 3]);
        assert!(mapped.stats.used_fallback && mapped.stats.dc_skipped > 0);
    }

    #[test]
    fn physical_pin_bindings_can_share_a_net_but_cycles_are_rejected() {
        let instance = Instance {
            cell: "PAIR".into(),
            area: 1.0,
            inputs: vec![0],
            input_pins: vec!["A".into()],
            outputs: vec![OutputPin {sources:Vec::new(),
                name: "Y".into(),
                delays: vec![1.0],
                function: 2,
                targets: vec![1, 2],
            }],
        };
        let mut stats = MapStats::default();
        let chosen = extract(
            &[instance.clone()],
            &[true, false, false],
            &[1, 2],
            &params(),
            &mut stats,
        )
        .unwrap();
        let graph = MappedGraph {
            instances: vec![instance],
            chosen,
            inputs: vec![(0, 1, "a".into())],
            outputs: vec![(1, "y".into()), (2, "z".into())],
            constants: vec![None; 3],
            area: 1.0,
            delay: 1.0,
            stats,
        };
        let mut text = Vec::new();
        graph.write_verilog(&mut text, "test").unwrap();
        let text = String::from_utf8(text).unwrap();
        assert!(text.contains("assign y = __egm_s1;") && text.contains("assign z = __egm_s1;"));
        let cycle = Instance {
            cell: "BAD".into(),
            area: 0.0,
            inputs: vec![1],
            input_pins: vec!["A".into()],
            outputs: vec![OutputPin {sources:Vec::new(),
                name: "Y".into(),
                delays: vec![0.0],
                function: 2,
                targets: vec![1],
            }],
        };
        assert!(extract(
            &[cycle],
            &[true, false],
            &[1],
            &params(),
            &mut MapStats::default()
        )
        .is_err());
    }

    #[test]
    fn hard_delay_bound_applies_to_secondary_pin_and_never_returns_bad_fallback() {
        let instances = vec![Instance {
            cell: "PAIR".into(),
            area: 1.0,
            inputs: vec![0],
            input_pins: vec!["A".into()],
            outputs: vec![
                OutputPin {sources:Vec::new(),
                    name: "FAST".into(),
                    delays: vec![1.0],
                    function: 2,
                    targets: vec![1],
                },
                OutputPin {sources:Vec::new(),
                    name: "SLOW".into(),
                    delays: vec![8.0],
                    function: 2,
                    targets: vec![2],
                },
            ],
        }];
        let ps = MapParams {
            max_delay: Some(4.0),
            ..params()
        };
        assert!(extract(
            &instances,
            &[true, false, false],
            &[1, 2],
            &ps,
            &mut MapStats::default()
        )
        .is_err());
    }

    #[test]
    fn constants_and_duplicate_primary_outputs_remain_driven() {
        let mut aig = graph();
        let a = input(&mut aig, 1);
        let zero = (NodeIndex::new(0), false);
        output(&mut aig, a, 0);
        output(&mut aig, a, 1);
        output(&mut aig, zero, 2);
        output(&mut aig, inv(zero), 3);
        let mapped = map(&aig, &library(""), &params()).unwrap();
        assert_function(&mapped, 1, |a| vec![a != 0, a != 0, false, true]);
        assert_eq!(mapped.area, 0.0);
    }

    fn heuristic_instance(
        name: &str,
        area: f32,
        input: usize,
        outputs: &[(usize, f32)],
    ) -> Instance {
        Instance {
            cell: name.into(),
            area,
            inputs: vec![input],
            input_pins: vec!["A".into()],
            outputs: outputs
                .iter()
                .enumerate()
                .map(|(i, &(signal, delay))| OutputPin {sources:Vec::new(),
                    name: format!("Y{i}"),
                    delays: vec![delay],
                    function: 2,
                    targets: vec![signal],
                })
                .collect(),
        }
    }

    #[test]
    fn shared_deadline_recovery_keeps_joint_pin_area_accounting() {
        let instances = vec![Instance {
            cell: "AND_OR".into(),
            area: 2.0,
            inputs: vec![0, 1],
            input_pins: vec!["A".into(), "B".into()],
            outputs: vec![
                OutputPin {sources:Vec::new(),
                    name: "AND".into(),
                    delays: vec![1.0, 1.0],
                    function: 0b1000,
                    targets: vec![2],
                },
                OutputPin {sources:Vec::new(),
                    name: "OR".into(),
                    delays: vec![2.0, 3.0],
                    function: 0b1110,
                    targets: vec![3],
                },
            ],
        }];
        let mut stats = MapStats::default();
        let chosen = heuristic_cover(
            &instances,
            &[true, true, false, false],
            &[2, 3],
            &MapParams {
                flow_compare: true,
                objective: MapObjective::AreaDelay,
                ..params()
            },
            &mut stats,
        )
        .unwrap();
        assert!(stats.flow_compare_attempted);
        assert_eq!(chosen[2].unwrap().instance, chosen[3].unwrap().instance);
        assert_eq!(
            metrics(&instances, &chosen, &[true, true, false, false], &[2, 3]),
            (2.0, 3.0)
        );
    }

    #[test]
    fn seeded_slack_flow_rejects_cheap_reconvergent_cycle() {
        let instances = vec![
            heuristic_instance("ONE", 3.0, 0, &[(1, 1.0)]),
            heuristic_instance("TWO", 3.0, 0, &[(2, 1.0)]),
            heuristic_instance("CRITICAL", 1.0, 0, &[(3, 10.0)]),
            heuristic_instance("LOOP1", 0.1, 2, &[(1, 1.0)]),
            heuristic_instance("LOOP2", 0.1, 1, &[(2, 1.0)]),
        ];
        let seed = vec![
            None,
            Some(Binding {
                instance: 0,
                pin: 0,
            }),
            Some(Binding {
                instance: 1,
                pin: 0,
            }),
            Some(Binding {
                instance: 2,
                pin: 0,
            }),
        ];
        let free = [true, false, false, false];
        let roots = [1, 2, 3];
        let required = required_times(&instances, &seed, &free, &roots, 10.0);
        let (mut trial, _) = flow_choices_worklist_impl(
            &instances,
            &free,
            &[8.0; 4],
            0.0,
            true,
            true,
            MapObjective::AreaDelay,
            Instant::now() + std::time::Duration::from_secs(1),
            Some(&required),
            Some(&seed),
        );
        prune(&instances, &mut trial, &free, &roots).unwrap();
        let value = metrics(&instances, &trial, &free, &roots);
        assert!(value.0 < 7.0 && value.1 <= 10.0);
    }

    #[test]
    fn required_time_flow_can_slow_noncritical_root_without_changing_delay() {
        let instances = vec![
            heuristic_instance("CRITICAL", 1.0, 0, &[(1, 10.0)]),
            heuristic_instance("FAST", 3.0, 0, &[(2, 1.0)]),
            heuristic_instance("CHEAP", 1.0, 0, &[(2, 5.0)]),
        ];
        let free = [true, false, false];
        let roots = [1, 2];
        let mut chosen = vec![
            None,
            Some(Binding {
                instance: 0,
                pin: 0,
            }),
            Some(Binding {
                instance: 1,
                pin: 0,
            }),
        ];
        let required = required_times(&instances, &chosen, &free, &roots, 10.0);
        assert_eq!(required[1], 10.0);
        assert_eq!(required[2], 10.0);
        let (mut trial, _) = flow_choices_worklist_impl(
            &instances,
            &free,
            &[1.0; 3],
            0.0,
            true,
            true,
            MapObjective::AreaDelay,
            Instant::now() + std::time::Duration::from_secs(1),
            Some(&required),
            Some(&chosen),
        );
        fill_from_witness(&instances, &mut trial, &chosen, &free, &roots).unwrap();
        assert_eq!(metrics(&instances, &chosen, &free, &roots), (4.0, 10.0));
        assert_eq!(metrics(&instances, &trial, &free, &roots), (2.0, 10.0));
        chosen = trial;
        assert_eq!(chosen[2].unwrap().instance, 2);
    }

    #[test]
    fn regional_import_keeps_unused_ports_and_demanded_complements() {
        let mut aig = graph();
        let a = input(&mut aig, 1);
        let b = input(&mut aig, 2);
        let _unused = input(&mut aig, 3);
        let ab = and(&mut aig, a, b);
        output(&mut aig, ab, 0);
        output(&mut aig, inv(ab), 1);
        output(&mut aig, inv(a), 2);
        let (old, _) = import_aig(&aig).unwrap();
        let (new, _) = regions::import_demanded(&aig).unwrap();
        assert!(new.total_number_of_nodes() < old.total_number_of_nodes());
        let ps = MapParams {
            region_classes: 2,
            phase_recovery: true,
            strict_budgets: true,
            time_limit_secs: 0.,
            ..Default::default()
        };
        let mapped = map(&aig, &library(""), &ps).unwrap();
        assert_eq!(mapped.inputs.len(), 3);
        assert!(mapped.stats.regions > 1);
        assert!(mapped.stats.enodes <= ps.saturation.node_limit);
        assert!(mapped.stats.instances <= ps.max_instances);
        assert_function(&mapped, 3, |bits| {
            let a = bits & 1 != 0;
            let b = bits & 2 != 0;
            vec![a && b, !(a && b), !a]
        });
        let mut exact_floor = ps.clone();
        exact_floor.saturation.node_limit = new.total_number_of_nodes();
        let bounded = map(&aig, &library(""), &exact_floor).unwrap();
        assert!(bounded.stats.enodes <= exact_floor.saturation.node_limit);
        assert!(bounded.stats.saturation_budget_skips > 0);
        let mut small = ps.clone();
        small.saturation.node_limit = 1;
        assert!(map(&aig, &library(""), &small)
            .err()
            .unwrap()
            .contains("cannot import AIG within node allowance"));
    }

    #[test]
    fn strict_budget_rejects_imported_floor_and_zero_recovery_time() {
        let mut aig = graph();
        let a = input(&mut aig, 1);
        let b = input(&mut aig, 2);
        let ab = and(&mut aig, a, b);
        output(&mut aig, ab, 0);
        let mut ps = params();
        ps.strict_budgets = true;
        ps.saturation.node_limit = 1;
        assert!(map(&aig, &library(""), &ps)
            .err()
            .unwrap()
            .contains("cannot import AIG within node allowance"));
        ps.saturation.node_limit = 20000;
        ps.refine_time_secs = 0.0;
        assert!(map(&aig, &library(""), &ps)
            .err()
            .unwrap()
            .contains("recovery"));
    }

    #[test]
    fn dependency_witness_matches_legacy_round_order() {
        for seed in 0..128u64 {
            let mut state = seed + 1;
            let mut random = || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state >> 32) as usize
            };
            let mut instances = Vec::new();
            for _ in 0..36 {
                let area = (random() % 5) as f32;
                let a = random() % 8;
                let b = random() % 8;
                let target = 1 + random() % 7;
                let mut m = heuristic_instance("TEST", area, a, &[(target, 1.0)]);
                m.inputs = vec![a, b, a];
                instances.push(m);
            }
            let free = [true, false, false, false, false, false, false, false];
            let mut available = free.to_vec();
            let mut old = vec![None; free.len()];
            let mut order: Vec<_> = (0..instances.len()).collect();
            order.sort_by(|&a, &b| instances[a].area.total_cmp(&instances[b].area));
            loop {
                let mut changed = false;
                for &i in &order {
                    if !instances[i].inputs.iter().all(|&s| available[s]) {
                        continue;
                    }
                    for (pin, o) in instances[i].outputs.iter().enumerate() {
                        for &s in &o.targets {
                            if !available[s] {
                                available[s] = true;
                                old[s] = Some(Binding { instance: i, pin });
                                changed = true;
                            }
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
            let keys = |xs: Vec<Option<Binding>>| {
                xs.into_iter()
                    .map(|b| b.map(|x| (x.instance, x.pin)))
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                keys(finite_choices_queued(&instances, &free)),
                keys(old),
                "seed {seed}"
            );
        }
    }

    #[test]
    fn pool_diagnostic_reports_discarded_early_arrival_without_mutating_candidates() {
        let original = vec![
            heuristic_instance("LATE", 1.0, 0, &[(1, 10.0)]),
            heuristic_instance("WITNESS", 1.0, 1, &[(2, 20.0)]),
            heuristic_instance("FAST_LATE", 2.0, 1, &[(2, 1.0)]),
            heuristic_instance("EARLY", 2.0, 0, &[(2, 5.0)]),
        ];
        let free = [true, false, false];
        let witness = finite_choices_queued(&original, &free);
        let file = std::env::temp_dir().join(format!("egm-pool-diagnostic-{}.json", std::process::id()));
        write_pool_diagnostic(&original, &free, &[2], &witness,
            &BTreeSet::from([0,1,2]), 2, 3, file.to_str().unwrap()).unwrap();
        let data: serde_json::Value = serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
        std::fs::remove_file(file).unwrap();
        let row = data["signals"].as_array().unwrap().iter().find(|r| r["signal"] == 2).unwrap();
        assert_eq!(row["before"],3);
        assert_eq!(row["after"],2);
        assert_eq!(row["best_before"]["cell"],"EARLY");
        assert_eq!(row["best_before"]["area_witness_arrival_estimate"],5.0);
        assert_eq!(row["best_after"]["area_witness_arrival_estimate"],11.0);
        assert_eq!(row["best_before"]["kept"],false);
        assert_eq!(original.len(),4);
        let keys = |bs: Vec<Option<Binding>>| bs.into_iter().map(|b| b.map(|x| (x.instance,x.pin))).collect::<Vec<_>>();
        assert_eq!(keys(witness),keys(finite_choices_queued(&original,&free)));
    }

    #[test]
    fn canonical_phase_aliases_are_exact_and_preserve_complement_pairs() {
        let ns=vec![vec![AigLang::Pi(1)],vec![AigLang::Not(Id::from(0))],vec![AigLang::Not(Id::from(1))]];
        let a=phase_aliases(&ns,usize::from).unwrap();assert_eq!(a,vec![0,1,1,0,0,1]);
        let x=normalize_phase_cut(&Cut{leaves:vec![0,1],function:8},&a);
        assert_eq!(x,Cut{leaves:vec![],function:0});
        assert_eq!(normalize_phase_cut(&Cut{leaves:vec![1],function:2},&a),Cut{leaves:vec![0],function:1});
        assert!(phase_aliases(&[vec![AigLang::Not(Id::from(0))]],usize::from).is_err());
    }

    #[test]
    fn canonical_phase_map_preserves_shared_positive_negative_primary_outputs() {
        let mut aig=graph();let a=input(&mut aig,1);let b=input(&mut aig,2);
        let ab=and(&mut aig,a,b);output(&mut aig,ab,0);output(&mut aig,inv(ab),1);output(&mut aig,inv(a),2);
        for mode in [false,true] {
            let mut ps=params();ps.canonical_phases=true;ps.canonical_wire_only=mode;ps.strict_budgets=true;ps.cuts_per_class=16;
            let mapped=map(&aig,&library(""),&ps).unwrap();
            assert_function(&mapped,2,|x|vec![x==3,x!=3,x&1==0]);
            assert!(mapped.stats.max_cuts_per_class<=16);
        }
    }

    #[test]
    fn producer_rank_interleave_preserves_both_heads_and_deduplicates() {
        assert_eq!(interleave_ranks(&[0,1,2,3],&[3,2,0,1]),vec![0,3,1,2]);
        assert_eq!(interleave_ranks(&[],&[2,1]),vec![2,1]);
        assert_eq!(interleave_ranks(&[1,1],&[1]),vec![1]);
    }

    #[test]
    fn timing_cap_retains_early_boundary_over_fast_cell_on_late_input() {
        let original = vec![
            heuristic_instance("LATE", 1.0, 0, &[(1, 10.0)]),
            heuristic_instance("WITNESS", 1.0, 1, &[(2, 20.0)]),
            heuristic_instance("FAST_LATE", 2.0, 1, &[(2, 1.0)]),
            heuristic_instance("EARLY", 2.0, 0, &[(2, 5.0)]),
        ];
        let mut timed = original.clone();
        cap_instances(&mut timed, &[true, false, false], &[2], 3, true).unwrap();
        assert!(timed.iter().any(|m| m.cell == "EARLY"));
        assert!(!timed.iter().any(|m| m.cell == "FAST_LATE"));
        let mut static_rank = original;
        cap_instances(&mut static_rank, &[true, false, false], &[2], 3, false).unwrap();
        assert!(static_rank.iter().any(|m| m.cell == "FAST_LATE"));
        let witness = fallback(&timed, &[true, false, false], &[2]).unwrap();
        assert_eq!(
            metrics(&timed, &witness, &[true, false, false], &[2]),
            (2.0, 30.0)
        );
    }

    #[test]
    fn hard_instance_cap_preserves_witness_and_rejects_insufficient_budget() {
        let mut instances = vec![
            heuristic_instance("CHEAP1", 1.0, 0, &[(1, 5.0)]),
            heuristic_instance("CHEAP2", 1.0, 1, &[(2, 5.0)]),
            heuristic_instance("FAST1", 2.0, 0, &[(1, 1.0)]),
            heuristic_instance("FAST2", 2.0, 1, &[(2, 1.0)]),
            heuristic_instance("OTHER1", 3.0, 0, &[(1, 2.0)]),
            heuristic_instance("OTHER2", 3.0, 1, &[(2, 2.0)]),
        ];
        let original = instances.clone();
        assert_eq!(
            cap_instances(&mut instances, &[true, false, false], &[2], 4, false).unwrap(),
            2
        );
        assert_eq!(instances.len(), 4);
        let witness = fallback(&instances, &[true, false, false], &[2]).unwrap();
        assert_eq!(
            metrics(&instances, &witness, &[true, false, false], &[2]),
            (2.0, 10.0)
        );
        let mut too_small = original;
        assert!(cap_instances(&mut too_small, &[true, false, false], &[2], 1, false).is_err());
        assert_eq!(too_small.len(), 6);
    }

    #[test]
    fn library_timing_bounded_reassociation_preserves_function() {
        let mut aig = graph();
        let a = input(&mut aig, 1);
        let b = input(&mut aig, 2);
        let c = input(&mut aig, 3);
        let ab = and(&mut aig, a, b);
        let abc = and(&mut aig, ab, c);
        output(&mut aig, abc, 0);
        let mapped = map(
            &aig,
            &library(""),
            &MapParams {
                library_timing: true,
                objective: MapObjective::DelayArea,
                max_instances: 200,
                saturation: SatParams {
                    node_limit: 64,
                    ..Default::default()
                },
                ..params()
            },
        )
        .unwrap();
        assert_function(&mapped, 3, |a| vec![a == 7]);
        assert!(mapped.stats.instances <= 200);
        assert!(mapped.stats.enodes <= 64);
    }

    #[test]
    fn mixed_cut_quota_keeps_legacy_prefix_without_increasing_cap() {
        let optional: Vec<_> = (0..4)
            .map(|i| Cut {
                leaves: vec![i],
                function: 2,
            })
            .collect();
        let depth = [4, 3, 2, 1];
        assert_eq!(
            mixed_optional_cuts(&optional, &depth, 3, 0),
            optional[..3].iter().cloned().collect()
        );
        assert_eq!(
            mixed_optional_cuts(&optional, &depth, 3, 1),
            [
                optional[0].clone(),
                optional[1].clone(),
                optional[3].clone()
            ]
            .into_iter()
            .collect()
        );
        assert_eq!(
            mixed_optional_cuts(&optional, &depth, 3, 99),
            optional[1..].iter().cloned().collect()
        );
        assert!(mixed_optional_cuts(&optional, &depth, 0, 99).is_empty());
        assert_eq!(mixed_optional_cuts(&optional, &[0, 0, 1, 2], 3, 2).len(), 3);
    }

    #[test]
    fn depth_ranked_tiny_cut_budget_retains_finite_equivalent_cover() {
        let mut aig = graph();
        let a = input(&mut aig, 1);
        let b = input(&mut aig, 2);
        let c = input(&mut aig, 3);
        let ab = and(&mut aig, a, b);
        let abc = and(&mut aig, ab, c);
        output(&mut aig, abc, 0);
        let mapped = map(
            &aig,
            &library(""),
            &MapParams {
                cut_depth_priority: true,
                cuts_per_class: 1,
                cut_rounds: 2,
                time_limit_secs: 0.0,
                ..params()
            },
        )
        .unwrap();
        assert_function(&mapped, 3, |a| vec![a == 7]);
        assert!(mapped.area.is_finite() && mapped.delay.is_finite());
        assert!(mapped.stats.cut_limit_hit);
    }

    #[test]
    fn critical_paths_include_ties_and_ignore_nondependent_pin_arcs() {
        let mut joint = heuristic_instance("JOINT", 2.0, 0, &[(2, 5.0), (3, 5.0)]);
        joint.inputs = vec![0, 1];
        joint.input_pins = vec!["A".into(), "B".into()];
        // Both functions depend only on A; a large B delay is irrelevant.
        for o in &mut joint.outputs {
            o.function = 0b1010;
            o.delays = vec![5.0, 99.0];
        }
        let instances = vec![joint];
        let chosen = vec![
            None,
            None,
            Some(Binding {
                instance: 0,
                pin: 0,
            }),
            Some(Binding {
                instance: 0,
                pin: 1,
            }),
        ];
        let (arrival, critical) =
            critical_arrivals(&instances, &chosen, &[true, true, false, false], &[2, 3]);
        assert_eq!(arrival[2], Some(5.0));
        assert_eq!(arrival[3], Some(5.0));
        assert_eq!(critical, vec![true, false, true, true]);
        let faster = heuristic_instance("FAST", 3.0, 0, &[(2, 1.0)]);
        assert_eq!(critical_gain(&faster, &chosen, &arrival, &critical), -4.0);
        let disconnected = heuristic_instance("OTHER", 0.1, 0, &[(1, 0.1)]);
        assert_eq!(
            critical_gain(&disconnected, &chosen, &arrival, &critical),
            0.0
        );
    }

    #[test]
    fn flow_comparison_preserves_incumbent_and_zero_budget_skips_second_stage() {
        let instances = vec![
            heuristic_instance("SLOW", 1.0, 0, &[(1, 5.0)]),
            heuristic_instance("FAST", 3.0, 0, &[(1, 1.0)]),
        ];
        for objective in [MapObjective::AreaDelay, MapObjective::DelayArea] {
            let ps = MapParams {
                objective,
                time_limit_secs: 0.0,
                ..params()
            };
            let original = heuristic_cover(
                &instances,
                &[true, false],
                &[1],
                &ps,
                &mut MapStats::default(),
            )
            .unwrap();
            let mut stats = MapStats::default();
            let combined = heuristic_cover(
                &instances,
                &[true, false],
                &[1],
                &MapParams {
                    flow_compare: true,
                    ..ps.clone()
                },
                &mut stats,
            )
            .unwrap();
            assert!(stats.flow_compare_attempted);
            assert_eq!(
                metrics(&instances, &combined, &[true, false], &[1]),
                metrics(&instances, &original, &[true, false], &[1])
            );
            let mut stats = MapStats::default();
            heuristic_cover(
                &instances,
                &[true, false],
                &[1],
                &MapParams {
                    flow_compare: true,
                    refine_time_secs: 0.0,
                    ..ps
                },
                &mut stats,
            )
            .unwrap();
            assert!(!stats.flow_compare_attempted);
        }
    }

    #[test]
    fn expired_shared_deadline_does_not_restart_recovery() {
        let instances = vec![
            heuristic_instance("SLOW", 1.0, 0, &[(1, 5.0)]),
            heuristic_instance("FAST", 3.0, 0, &[(1, 1.0)]),
        ];
        let mut stats = MapStats::default();
        heuristic_cover_until(
            &instances,
            &[true, false],
            &[1],
            &MapParams {
                objective: MapObjective::DelayArea,
                flow_worklist: true,
                ..params()
            },
            &mut stats,
            Instant::now(),
        )
        .unwrap();
        assert_eq!(stats.refinement_trials, 0);
    }

    #[test]
    fn worklist_reaches_deep_reverse_ordered_cones_and_keeps_timing_objective() {
        let mut instances = Vec::new();
        for s in (1..=80).rev() {
            instances.push(heuristic_instance("FAST", 2.0, s - 1, &[(s, 1.0)]));
            instances.push(heuristic_instance("SLOW", 1.0, s - 1, &[(s, 3.0)]));
        }
        let mut free = vec![false; 81];
        free[0] = true;
        let refs = vec![1.0; 81];
        let deadline = Instant::now() + std::time::Duration::from_secs(2);
        let (sweeps, _) = flow_choices(
            &instances,
            &free,
            &refs,
            0.0,
            false,
            false,
            MapObjective::DelayArea,
            deadline,
        );
        assert!(sweeps[80].is_none());
        for (objective, expected) in [
            (MapObjective::AreaDelay, (80.0, 240.0)),
            (MapObjective::DelayArea, (160.0, 80.0)),
        ] {
            let (mut chosen, _) = flow_choices_worklist(
                &instances, &free, &refs, 0.0, false, false, objective, deadline,
            );
            assert!(chosen[80].is_some());
            prune(&instances, &mut chosen, &free, &[80]).unwrap();
            assert_eq!(metrics(&instances, &chosen, &free, &[80]), expected);
        }
    }

    #[test]
    fn lexicographic_objectives_choose_primary_then_secondary_with_and_without_cbc() {
        let instances = vec![
            heuristic_instance("SLOW_CHEAP", 1.0, 0, &[(1, 8.0)]),
            heuristic_instance("FAST_EXPENSIVE", 4.0, 0, &[(1, 2.0)]),
            heuristic_instance("FAST_CHEAPER", 3.0, 0, &[(1, 2.0)]),
        ];
        for time_limit_secs in [0.0, 5.0] {
            for (objective, expected, max_delay) in [
                (MapObjective::AreaDelay, (1.0, 8.0), None),
                (MapObjective::DelayArea, (3.0, 2.0), None),
                (MapObjective::AreaDelay, (3.0, 2.0), Some(4.0)),
            ] {
                let ps = MapParams {
                    objective,
                    time_limit_secs,
                    max_delay,
                    ..params()
                };
                let mut stats = MapStats::default();
                let chosen = extract(&instances, &[true, false], &[1], &ps, &mut stats).unwrap();
                assert_eq!(metrics(&instances, &chosen, &[true, false], &[1]), expected);
                if time_limit_secs > 0.0 {
                    assert!(stats.proven_optimal, "{:?}", stats);
                }
            }
        }
        let delay_first = MapParams {
            objective: MapObjective::DelayArea,
            ..params()
        };
        assert!(better_cover((1e9, 1.0), (1.0, 1.001), &delay_first));
        assert!(!better_cover((0.001, 1.001), (1.0, 1.0), &delay_first));
    }

    #[test]
    fn equal_area_flow_prefers_faster_pin_independent_of_candidate_order() {
        for reversed in [false, true] {
            let mut instances = vec![
                heuristic_instance("SLOW", 1.0, 0, &[(1, 8.0)]),
                heuristic_instance("FAST", 1.0, 0, &[(1, 2.0)]),
            ];
            if reversed {
                instances.reverse();
            }
            let (chosen, _) = flow_choices(
                &instances,
                &[true, false],
                &[1.0, 1.0],
                0.0,
                false,
                true,
                MapObjective::Weighted,
                Instant::now() + std::time::Duration::from_secs(1),
            );
            assert_eq!(instances[chosen[1].unwrap().instance].cell, "FAST");
        }
    }

    #[test]
    fn physical_refs_count_shared_cell_inputs_once_and_each_physical_consumer() {
        let instances = vec![
            heuristic_instance("PAIR", 1.0, 0, &[(1, 1.0), (2, 1.0)]),
            heuristic_instance("A", 1.0, 1, &[(3, 1.0)]),
            heuristic_instance("B", 1.0, 1, &[(4, 1.0)]),
        ];
        let chosen = vec![
            None,
            Some(Binding {
                instance: 0,
                pin: 0,
            }),
            Some(Binding {
                instance: 0,
                pin: 1,
            }),
            Some(Binding {
                instance: 1,
                pin: 0,
            }),
            Some(Binding {
                instance: 2,
                pin: 0,
            }),
        ];
        let refs = physical_refs(&instances, &chosen, &[2, 3, 4, 4]);
        assert_eq!(refs, vec![1.0, 2.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn cone_reclaim_counts_shared_multioutput_cell_only_after_both_outputs_die() {
        let instances = vec![
            heuristic_instance("PAIR", 4.0, 0, &[(1, 1.0), (2, 1.0)]),
            heuristic_instance("A", 1.0, 1, &[(3, 1.0)]),
            heuristic_instance("B", 1.0, 2, &[(4, 1.0)]),
        ];
        let chosen = vec![
            None,
            Some(Binding {
                instance: 0,
                pin: 0,
            }),
            Some(Binding {
                instance: 0,
                pin: 1,
            }),
            Some(Binding {
                instance: 1,
                pin: 0,
            }),
            Some(Binding {
                instance: 2,
                pin: 0,
            }),
        ];
        let free = [true, false, false, false, false];
        let refs = [1, 1, 1, 1, 1];
        let users = [2, 1, 1];
        assert_eq!(
            reclaimable_area(&[3], &instances, &chosen, &free, &refs, &users),
            1.0
        );
        assert_eq!(
            reclaimable_area(&[3, 4], &instances, &chosen, &free, &refs, &users),
            6.0
        );
        assert_eq!(
            reclaimable_area(&[3, 3, 4], &instances, &chosen, &free, &refs, &users),
            6.0
        );
    }

    #[test]
    fn flow_warm_start_finds_joint_sharing_that_independent_tree_cost_misses() {
        let instances = vec![
            heuristic_instance("SHARED", 4.0, 0, &[(1, 1.0)]),
            heuristic_instance("DIRECT_A", 4.0, 0, &[(2, 1.0)]),
            heuristic_instance("REUSE_A", 1.0, 1, &[(2, 1.0)]),
            heuristic_instance("DIRECT_B", 4.0, 0, &[(3, 1.0)]),
            heuristic_instance("REUSE_B", 1.0, 1, &[(3, 1.0)]),
        ];
        let free = [true, false, false, false];
        let roots = [2, 3];
        let initial = fallback(&instances, &free, &roots).unwrap();
        assert_eq!(metrics(&instances, &initial, &free, &roots).0, 8.0);
        let mut stats = MapStats::default();
        let ps = MapParams {
            time_limit_secs: 0.0,
            structural_refs: true,
            ..params()
        };
        let recovered = extract(&instances, &free, &roots, &ps, &mut stats).unwrap();
        assert_eq!(metrics(&instances, &recovered, &free, &roots).0, 6.0);
        assert_eq!(stats.initial_area, 8.0);
        assert_eq!(stats.heuristic_area, 6.0);
    }

    #[test]
    fn exact_recovery_changes_multiple_outputs_as_one_physical_instance() {
        let instances = vec![
            heuristic_instance("A", 3.0, 0, &[(1, 1.0)]),
            heuristic_instance("B", 3.0, 0, &[(2, 1.0)]),
            heuristic_instance("BOTH", 4.0, 0, &[(1, 1.0), (2, 1.0)]),
        ];
        let mut stats = MapStats::default();
        let ps = MapParams {
            time_limit_secs: 0.0,
            ..params()
        };
        let recovered =
            extract(&instances, &[true, false, false], &[1, 2], &ps, &mut stats).unwrap();
        assert_eq!(stats.initial_area, 6.0);
        assert_eq!(stats.heuristic_area, 4.0);
        assert!(stats.refinement_accepted > 0);
        assert_eq!(
            recovered[1].unwrap().instance,
            recovered[2].unwrap().instance
        );
    }

    #[test]
    fn recovery_can_use_only_the_fast_pin_under_a_hard_delay_limit() {
        let instances = vec![
            heuristic_instance("A", 3.0, 0, &[(1, 1.0)]),
            heuristic_instance("B", 3.0, 0, &[(2, 1.0)]),
            heuristic_instance("PAIR", 0.5, 0, &[(1, 1.0), (2, 8.0)]),
        ];
        let mut stats = MapStats::default();
        let ps = MapParams {
            max_delay: Some(4.0),
            time_limit_secs: 0.0,
            ..params()
        };
        // The old first-feasible cover picks both pins and violates the bound. Recovery must find
        // a mixed cover rather than return that infeasible seed or insist on using both pins.
        let recovered =
            extract(&instances, &[true, false, false], &[1, 2], &ps, &mut stats).unwrap();
        let (area, delay) = metrics(&instances, &recovered, &[true, false, false], &[1, 2]);
        assert_eq!(area, 3.5);
        assert_eq!(delay, 1.0);
    }

    #[test]
    fn disabled_recovery_preserves_the_original_finite_cover() {
        let instances = vec![heuristic_instance("BUF", 1.0, 0, &[(1, 1.0)])];
        let ps = MapParams {
            refine_time_secs: 0.0,
            time_limit_secs: 0.0,
            ..params()
        };
        let mut stats = MapStats::default();
        let recovered = extract(&instances, &[true, false], &[1], &ps, &mut stats).unwrap();
        assert_eq!(metrics(&instances, &recovered, &[true, false], &[1]).0, 1.0);
        assert_eq!(stats.refinement_trials, 0);
    }

    #[test]
    fn three_output_cell_and_pin_symmetry_preserve_all_outputs() {
        let lib = library(
            "GATE THREE 0.5 Y=A*B;\nPIN A NONINV 1 99 1 0 1 0\nPIN B NONINV 1 99 1 0 1 0\n\
            GATE THREE 0.5 Z=A+B;\nPIN A NONINV 1 99 2 0 2 0\nPIN B NONINV 1 99 2 0 2 0\n\
            GATE THREE 0.5 W=A*!B+!A*B;\nPIN A UNKNOWN 1 99 3 0 3 0\nPIN B UNKNOWN 1 99 3 0 3 0\n",
        );
        let mut aig = graph();
        let a = input(&mut aig, 1);
        let b = input(&mut aig, 2);
        let x = and(&mut aig, a, b);
        let y = or(&mut aig, a, b);
        let z = xor(&mut aig, a, b);
        output(&mut aig, x, 0);
        output(&mut aig, y, 1);
        output(&mut aig, z, 2);
        let mapped = map(&aig, &lib, &params()).unwrap();
        assert_function(&mapped, 2, |a| vec![a == 3, a != 0, a == 1 || a == 2]);
        assert!((mapped.area - 0.5).abs() < 1e-6);
        let used: HashSet<_> = mapped
            .chosen
            .iter()
            .flatten()
            .map(|b| (b.instance, b.pin))
            .collect();
        assert_eq!(used.len(), 3);
    }

    #[test]
    #[ignore = "requires Yosys and ABC executables"]
    fn emitted_multioutput_netlist_passes_cec_and_corrupted_netlist_fails() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let library = TechLib::from_genlib(root.join("egraph_map.genlib"), 6).unwrap();
        let golden = root.join("full_adder.aag");
        let aig = Aig::from_aiger(golden.to_str().unwrap());
        let mapped = map(&aig, &library, &params()).unwrap();
        let path = std::env::temp_dir().join(format!("egm_verify_test_{}.v", std::process::id()));
        mapped
            .write_verilog(std::fs::File::create(&path).unwrap(), "mapped")
            .unwrap();
        assert!(verify_with_models(&library, &golden, &path).unwrap());
        let text = std::fs::read_to_string(&path).unwrap();
        let corrupted = text
            .lines()
            .map(|line| {
                if line.contains("assign sum =") {
                    "  assign sum = 1'b0;"
                } else {
                    line
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&path, corrupted).unwrap();
        assert!(!verify_with_models(&library, &golden, &path).unwrap());
        let unnamed = path.with_extension("aag");
        let source = std::fs::read_to_string(&golden).unwrap();
        let source = source
            .lines()
            .filter(|line| {
                line.starts_with("aag ") || line.as_bytes().first().is_some_and(u8::is_ascii_digit)
            })
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&unnamed, format!("{source}\n")).unwrap();
        let anonymous = Aig::from_aiger(unnamed.to_str().unwrap());
        let anonymous = map(&anonymous, &library, &params()).unwrap();
        anonymous
            .write_verilog(std::fs::File::create(&path).unwrap(), "mapped")
            .unwrap();
        assert!(verify_with_models(&library, &unnamed, &path).unwrap());
        std::fs::remove_file(unnamed).unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    #[ignore = "requires Yosys and ABC executables"]
    fn named_port_cec_handles_large_scripts_and_rejects_wrong_indexed_output() {
        let n=1200usize;
        let directory=std::env::temp_dir().join(format!("egm_largeports_test_{}",std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let golden=directory.join("golden.aag");let mapped=directory.join("mapped.v");
        let mut aag=format!("aag {n} {n} 0 {n} 0\n");
        for v in 1..=n {aag.push_str(&format!("{}\n",2*v));}
        for v in 1..=n {aag.push_str(&format!("{}\n",2*v+1));}
        for i in 0..n {aag.push_str(&format!("i{i} a[{i}]\n"));}
        for i in 0..n {aag.push_str(&format!("o{i} y[{i}]\n"));}
        std::fs::write(&golden,aag).unwrap();
        let ports=(0..n).flat_map(|i|[format!("\\a[{i}] "),format!("\\y[{i}] ")]).collect::<Vec<_>>();
        let mut text=format!("module mapped({});\n",ports.join(","));
        for i in 0..n {text.push_str(&format!("input \\a[{i}] ; output \\y[{i}] ; assign \\y[{i}] = ~\\a[{i}] ;\n"));}
        text.push_str("endmodule\n");std::fs::write(&mapped,&text).unwrap();
        assert!(verify_with_verilog_models("",&golden,&mapped).unwrap());
        let broken=text.replace("assign \\y[0] = ~\\a[0] ;","assign \\y[0] = ~\\a[1] ;");
        assert_ne!(text,broken);std::fs::write(&mapped,broken).unwrap();
        assert!(!verify_with_verilog_models("",&golden,&mapped).unwrap());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn extraction_uses_sharing_exposed_by_an_alternative_egraph_realization() {
        let mut aig = graph();
        let a = input(&mut aig, 1);
        let b = input(&mut aig, 2);
        let c = input(&mut aig, 3);
        let ab = and(&mut aig, a, b);
        let ac = and(&mut aig, a, c);
        let root = and(&mut aig, ab, ac);
        let bc = and(&mut aig, b, c);
        output(&mut aig, root, 0);
        output(&mut aig, bc, 1);
        let lib = library("");
        let mut no_factor = params();
        no_factor.saturation.use_factoring = false;
        let original = map(&aig, &lib, &no_factor).unwrap();
        let factored = map(&aig, &lib, &params()).unwrap();
        assert_function(&factored, 3, |a| vec![a == 7, a & 6 == 6]);
        assert!((factored.area - 2.0).abs() < 1e-6);
        assert!(factored.area < original.area - 1e-6);
    }

    #[test]
    fn simultaneous_sdc_and_multioutput_matches_preserve_all_primary_input_assignments() {
        let lib = library(
            "GATE PAIR 0.9 Y=A*B;\nPIN A NONINV 1 99 1 0 1 0\nPIN B NONINV 1 99 1 0 1 0\n\
            GATE PAIR 0.9 Z=!(A*B);\nPIN A INV 1 99 2 0 2 0\nPIN B INV 1 99 2 0 2 0\n",
        );
        let mut seed = 12345u64;
        for case in 0..3 {
            let mut aig = graph();
            let mut signals = Vec::new();
            for i in 0..4 {
                let s = input(&mut aig, i + 1);
                let mut tt = 0u16;
                for a in 0..16 {
                    tt |= (((a >> i) & 1) as u16) << a;
                }
                signals.push((s, tt));
            }
            for _ in 0..8 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let (a, ta) = signals[(seed >> 16) as usize % signals.len()];
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let (mut b, mut tb) = signals[(seed >> 16) as usize % signals.len()];
                if seed & 1 != 0 {
                    b = inv(b);
                    tb = !tb;
                }
                signals.push((and(&mut aig, a, b), ta & tb));
            }
            let expected: Vec<_> = signals.iter().rev().take(3).map(|(_, tt)| *tt).collect();
            for (pos, &(s, _)) in signals.iter().rev().take(3).enumerate() {
                output(&mut aig, s, pos as u32);
            }
            let ps = MapParams {
                max_inputs: 3,
                dc_support: 4,
                time_limit_secs: if case == 0 { 3.0 } else { 0.0 },
                ..params()
            };
            let precise = map(
                &aig,
                &lib,
                &MapParams {
                    dc_support: 0,
                    ..ps.clone()
                },
            )
            .unwrap();
            assert_function(&precise, 4, |a| {
                expected.iter().map(|tt| (tt >> a) & 1 != 0).collect()
            });
            let (rebuilt, _) = crate::egraph_sat::restructure(
                &aig,
                &SatParams {
                    require_improvement: false,
                    ..Default::default()
                },
            )
            .unwrap();
            let rebuilt_cover = map(
                &rebuilt,
                &lib,
                &MapParams {
                    dc_support: 0,
                    time_limit_secs: 0.0,
                    ..ps.clone()
                },
            )
            .unwrap();
            assert_function(&rebuilt_cover, 4, |a| {
                expected.iter().map(|tt| (tt >> a) & 1 != 0).collect()
            });
            let mapped = map(&aig, &lib, &ps).unwrap();
            assert_function(&mapped, 4, |a| {
                expected.iter().map(|tt| (tt >> a) & 1 != 0).collect()
            });
        }
    }
}

#[cfg(test)]
mod egraph_unique_input_cost_tests {
    use super::*;
    #[test]
    fn sorted_input_cost_matches_tree_bitwise_with_aliases_and_infinity() {
        let mut random = 907u64;
        for n in 0..20 {
            for _ in 0..400 {
                let mut next = || { random = random.wrapping_mul(6364136223846793005).wrapping_add(1); random };
                let costs: Vec<f64> = (0..32).map(|_| { let x = next(); if x%13==0 { f64::INFINITY } else { (x%10000) as f64 / 37.0 } }).collect();
                let inputs: Vec<_> = (0..n).map(|_| (next()%32) as usize).collect();
                let old: f64 = inputs.iter().copied().collect::<BTreeSet<_>>().into_iter().map(|s| costs[s]).sum();
                assert_eq!(unique_input_cost(&inputs,&costs).to_bits(),old.to_bits());
            }
        }
    }
}

#[cfg(test)]
mod egraph_flow_area_overflow_tests {
    use super::*;
    #[test]
    fn deep_reconvergent_delay_flow_does_not_confuse_area_overflow_with_unreachable() {
        // Preserve the full overflow fixture in unoptimized test builds. This
        // changes the test thread stack only, never a mapper budget or CLI stack.
        std::thread::Builder::new().stack_size(16 * 1024 * 1024).spawn(|| {
        let levels=2200usize;let mut instances=Vec::new();
        for signal in 2..levels+2 {
            for (area,delay) in [(1.0f32,2.0f32),(2.0,1.0)] {
                instances.push(Instance{cell:"AND2".into(),area,inputs:vec![signal-1,signal-2],input_pins:vec!["A".into(),"B".into()],outputs:vec![OutputPin{sources:Vec::new(),name:"Y".into(),delays:vec![delay,delay],function:8,targets:vec![signal]}]});
            }
        }
        let mut free=vec![false;levels+2];free[0]=true;free[1]=true;
        let (chosen,costs)=flow_choices_worklist(&instances,&free,&vec![1.;levels+2],0.,false,false,MapObjective::DelayArea,Instant::now()+std::time::Duration::from_secs(4));
        assert!(chosen[2..].iter().all(Option::is_some),"finite gates lost after tree-area f64 overflow");
        assert!(costs[2..].iter().all(|x|x.is_finite()));
        let value=metrics(&instances,&chosen,&free,&[levels+1]);
        assert_eq!(value.0,2.0*levels as f64);assert_eq!(value.1,levels as f64);
        }).unwrap().join().unwrap();
    }
}

#[cfg(test)]
mod egraph_flow_proxy_saturation_tests {
    use super::*;
    #[test]
    fn unknown_input_is_not_turned_into_a_finite_label() {
        assert_eq!(flow_input_cost(&[0,1],&[f64::MAX,f64::MAX],true),f64::MAX);
        assert!(flow_input_cost(&[0,1],&[f64::MAX,f64::INFINITY],true).is_infinite());
        assert!(flow_input_cost(&[0,1],&[f64::MAX,f64::MAX],false).is_infinite());
        assert_eq!(flow_input_cost(&[0,0],&[3.0],true),3.0);
    }
}

#[cfg(test)] mod bounded_proposal_tests {use super::*;
#[test] fn bounded_topk_equals_full_sort_with_timing_and_stable_ties(){let values:Vec<_>=(0..10000).map(|i|(-(((i*13)%17)as f64),((i*7)%31)as f64,i)).collect();let mut all=values.clone();all.sort_by(|a,b|RecoveryProposal(a.0,a.1,a.2).cmp(&RecoveryProposal(b.0,b.1,b.2)));for k in [0,1,16,2000,10000]{assert_eq!(best_recovery_proposals(values.iter().copied(),k),all[..k]);}}
}

#[cfg(test)]mod critical_proposal_order_tests{use super::*;#[test]fn timing_gain_precedes_area_cost(){assert_eq!(best_recovery_proposals([(0.,-100.,1),(-1.,100.,2),(0.,-100.,0)].into_iter(),2),vec![(-1.,100.,2),(0.,-100.,0)]);}}

#[cfg(test)]mod delay_frontier_tests{use super::*;
fn cell(input:Vec<usize>,target:usize,delay:f32,area:f32)->Instance{let k=input.len();Instance{cell:"fixture".into(),area,inputs:input,input_pins:(0..k).map(|i|format!("p{i}")).collect(),outputs:vec![OutputPin{sources:Vec::new(),name:"Y".into(),delays:vec![delay;k],function:1u64<<((1usize<<k)-1),targets:vec![target]}]}}
#[test]fn readiness_handles_cycles_duplicate_inputs_and_expired_deadline(){let m=vec![cell(vec![0],1,10.,1.),cell(vec![0],1,1.,2.),cell(vec![1],2,1.,1.),cell(vec![2],1,0.,0.1),cell(vec![1,1],3,2.,1.),cell(vec![4],4,0.,1.)];let free=vec![true,false,false,false,false];let(mut c,_)=delay_frontier_seed(&m,&free,Instant::now()+std::time::Duration::from_secs(1)).unwrap();assert_eq!(c[1].unwrap().instance,1);assert!(c[4].is_none());prune(&m,&mut c,&free,&[2,3]).unwrap();assert_eq!(metrics(&m,&c,&free,&[2,3]).1,3.);assert!(delay_frontier_seed(&m,&free,Instant::now()).is_none());}
#[test]fn all_arc_small_DAG_agrees_with_exhaustive_cover_delays(){for seed in 0..16{let mut m=Vec::new();for target in 2..8{let input=vec![target-1,(target+seed)%target];m.push(cell(input.clone(),target,1.+((target+seed)%7)as f32,1.));m.push(cell(input,target,1.+((3*target+seed)%5)as f32,2.));}let free=vec![true,true,false,false,false,false,false,false];let(mut c,_)=delay_frontier_seed(&m,&free,Instant::now()+std::time::Duration::from_secs(1)).unwrap();prune(&m,&mut c,&free,&[7]).unwrap();let got=metrics(&m,&c,&free,&[7]).1;let mut minimum=f64::INFINITY;for mask in 0..64{let mut trial=vec![None;8];for t in 2..8{trial[t]=Some(Binding{instance:2*(t-2)+((mask>>(t-2))&1),pin:0});}prune(&m,&mut trial,&free,&[7]).unwrap();minimum=minimum.min(metrics(&m,&trial,&free,&[7]).1);}assert_eq!(got,minimum);}}
}

#[cfg(test)]mod required_area_dag_tests{use super::*;
fn gate(inputs:Vec<usize>,target:usize,area:f32,delay:f32)->Instance{let n=inputs.len();Instance{cell:"fixture".into(),area,inputs,input_pins:(0..n).map(|i|format!("p{i}")).collect(),outputs:vec![OutputPin{sources:Vec::new(),name:"Y".into(),delays:vec![delay;n],function:1u64<<((1usize<<n)-1),targets:vec![target]}]}}
#[test]fn timed_sweep_reclaims_area_without_slowing_critical_output(){let m=vec![gate(vec![0],1,3.,1.),gate(vec![0],1,1.,2.),gate(vec![0],2,2.,4.),gate(vec![1,2],3,1.,1.)];let free=vec![true,false,false,false];let full=vec![None,Some(Binding{instance:0,pin:0}),Some(Binding{instance:2,pin:0}),Some(Binding{instance:3,pin:0})];let mut old=full.clone();prune(&m,&mut old,&free,&[3]).unwrap();let mut new=required_area_dag(&m,&free,&[3],&old,&full,Instant::now()+std::time::Duration::from_secs(1)).unwrap();prune(&m,&mut new,&free,&[3]).unwrap();assert_eq!(metrics(&m,&old,&free,&[3]),(6.,5.));assert_eq!(metrics(&m,&new,&free,&[3]),(4.,5.));assert!(required_area_dag(&m,&free,&[3],&old,&full,Instant::now()).is_none());}
#[test]fn later_dependency_is_not_used_to_create_a_cycle(){let m=vec![gate(vec![0],1,2.,1.),gate(vec![1],2,2.,1.),gate(vec![2],1,0.1,0.)];let free=vec![true,false,false];let full=vec![None,Some(Binding{instance:0,pin:0}),Some(Binding{instance:1,pin:0})];let mut new=required_area_dag(&m,&free,&[2],&full,&full,Instant::now()+std::time::Duration::from_secs(1)).unwrap();prune(&m,&mut new,&free,&[2]).unwrap();assert_eq!(metrics(&m,&new,&free,&[2]),(4.,2.));}
#[test]fn typed_recovery_configuration_enables_and_scopes_metric_workspace(){let m=vec![gate(vec![0],1,3.,1.),gate(vec![0],1,1.,2.),gate(vec![0],2,2.,4.),gate(vec![1,2],3,1.,1.)];let free=vec![true,false,false,false];let mut ps=MapParams::default();ps.objective=MapObjective::DelayArea;ps.strict_budgets=true;ps.flow_worklist=true;ps.structural_refs=true;ps.slack_flow=true;ps.fast_delay_seed=true;ps.required_area_dag=true;ps.defer_area_dag=true;ps.preserve_legacy_seeds=true;ps.metric_epochs=true;let before=METRIC_WORKSPACE.with(|w|w.borrow().epoch);let c=heuristic_cover(&m,&free,&[3],&ps,&mut MapStats::default()).unwrap();let after=METRIC_WORKSPACE.with(|w|w.borrow().epoch);assert!(after>before,"typed metric_epochs must enable the actual recovery workspace");assert!(!METRIC_EPOCHS.with(|v|v.get()));assert_eq!(metrics(&m,&c,&free,&[3]),(4.,5.));ps.metric_epochs=false;heuristic_cover(&m,&free,&[3],&ps,&mut MapStats::default()).unwrap();assert_eq!(METRIC_WORKSPACE.with(|w|w.borrow().epoch),after);}

}

#[cfg(test)]mod metric_epoch_tests {use super::*;
pub(super) fn fixture(n:usize)->(Vec<Instance>,Vec<Option<Binding>>,Vec<bool>){let mut m=vec![];let mut c=vec![None;n];let mut f=vec![false;n];f[0]=true;for s in 1..n{let inputs=vec![s-1,s/2];m.push(Instance{cell:"fixture".into(),area:(s%5+1)as f32/10.,inputs,input_pins:vec!["A".into(),"B".into()],outputs:vec![OutputPin{sources:Vec::new(),name:"Y".into(),delays:vec![(s%7+1)as f32,2.],function:8,targets:vec![s]}]});c[s]=Some(Binding{instance:s-1,pin:0});}(m,c,f)}
#[test]fn cached_metrics_agree_across_graph_sizes_roots_free_signals_and_epoch_wrap(){for n in [33,8,97,12,129,4]{let(m,c,f)=fixture(n);for root in (1..n).rev(){let mut cover=c.clone();prune(&m,&mut cover,&f,&[root]).unwrap();let expected=metrics(&m,&cover,&f,&[root]);{let _guard=MetricEpochGuard::enter(true);assert_eq!(metrics(&m,&cover,&f,&[root]),expected);assert_eq!(metrics(&m,&cover,&f,&[0]),(expected.0,0.));}assert_eq!(metrics(&m,&cover,&f,&[root]),expected);}}METRIC_WORKSPACE.with(|w|w.borrow_mut().epoch=u32::MAX);let(m,c,f)=fixture(7);let expected=metrics(&m,&c,&f,&[6]);let _guard=MetricEpochGuard::enter(true);assert_eq!(metrics(&m,&c,&f,&[6]),expected);}
#[test]fn epoch_guard_restores_state_on_early_error(){let result:Result<(),()>= (||{let _guard=MetricEpochGuard::enter(true);assert!(METRIC_EPOCHS.with(|v|v.get()));Err(())})();assert!(result.is_err());assert!(!METRIC_EPOCHS.with(|v|v.get()));}
#[test]fn shared_physical_instance_is_charged_once_and_pin_arcs_remain_independent(){let m=vec![Instance{cell:"joint".into(),area:2.,inputs:vec![0,1],input_pins:vec!["A".into(),"B".into()],outputs:vec![OutputPin{sources:Vec::new(),name:"X".into(),delays:vec![3.,0.],function:10,targets:vec![2]},OutputPin{sources:Vec::new(),name:"Y".into(),delays:vec![0.,5.],function:12,targets:vec![3]}]}];let c=vec![None,None,Some(Binding{instance:0,pin:0}),Some(Binding{instance:0,pin:1})];let f=vec![true,true,false,false];let expected=metrics(&m,&c,&f,&[2,3]);let _guard=MetricEpochGuard::enter(true);assert_eq!(metrics(&m,&c,&f,&[2,3]),expected);assert_eq!(expected,(2.,5.));}
}

#[cfg(test)]mod metric_epoch_profile {use super::*;
#[test]#[ignore]fn sparse_cover_exact_metric_cost(){let(m,mut c,f)=super::metric_epoch_tests::fixture(20000);prune(&m,&mut c,&f,&[32]).unwrap();let expected=metrics(&m,&c,&f,&[32]);for repetition in 0..3{let mut times=[0.;2];for enabled in if repetition%2==0{[false,true]}else{[true,false]}{let _guard=MetricEpochGuard::enter(enabled);let t=Instant::now();for _ in 0..3000{assert_eq!(std::hint::black_box(metrics(&m,&c,&f,&[32])),expected);}times[usize::from(enabled)]=t.elapsed().as_secs_f64();}eprintln!("[PROFILE-metric-epoch] repetition={repetition} original={} epoch={}",times[0],times[1]);}}
}

#[cfg(test)]mod diagnostic_index_tests {use super::*;
#[test]fn producer_index_matches_naive_signal_scan_with_aliases_duplicate_targets_and_pins(){for seed in 0..64{let mut m=Vec::new();for i in 0..19{m.push(Instance{cell:"fixture".into(),area:1.,inputs:vec![0],input_pins:vec!["A".into()],outputs:(0..2).map(|p|OutputPin{sources:Vec::new(),name:format!("Y{p}"),delays:vec![1.],function:2,targets:vec![(i+seed+p)%13,(i+seed+p)%13,(3*i+p)%13,99]}).collect()});}let index=diagnostic_producer_index(&m,13);for target in 0..13{let expected:Vec<_>=m.iter().enumerate().flat_map(|(i,g)|g.outputs.iter().enumerate().filter(move |(_,o)|o.targets.contains(&target)).map(move |(p,_)|(i,p))).collect();assert_eq!(index[target],expected);}}}
}

#[cfg(test)]mod diagnostic_names_golden_tests{use super::*;
#[test]fn memoized_cell_names_agree_with_saved_original_full_sqrt_cut_report(){let root=std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");let lib=TechLib::from_genlib(root.join("diagnostic_names.genlib"),6).unwrap();let rows:serde_json::Value=serde_json::from_slice(&std::fs::read(root.join("diagnostic_names_golden.json")).unwrap()).unwrap();let mut cache=std::collections::BTreeMap::new();for row in rows.as_array().unwrap(){let k=row["k"].as_u64().unwrap()as usize;let function=row["function"].as_u64().unwrap();let expected:BTreeSet<String>=row["names"].as_array().unwrap().iter().map(|v|v.as_str().unwrap().to_owned()).collect();let actual=cache.entry((k,function)).or_insert_with(||diagnostic_cell_names(&lib,k,function));assert_eq!(*actual,expected);}}}

#[cfg(test)] mod joint_cone_tests {
 use super::*;
 fn inv(input:usize,target:usize,delay:f32,area:f32)->Instance {Instance{cell:"INV".into(),area,inputs:vec![input],input_pins:vec!["A".into()],outputs:vec![OutputPin{sources:Vec::new(),name:"Y".into(),delays:vec![delay],function:1,targets:vec![target]}]}}
 #[test] fn joint_input_cone_breaks_witness_cycle_and_preserves_old_shared_cover(){
  // s0=x; s1=x is the PO, s2=!x is initially unused. Every output function is exact.
  let mut slow=inv(0,1,10.,1.);slow.cell="BUF".into();slow.outputs[0].function=2;
  let m=vec![slow,inv(1,2,0.1,0.1),inv(0,2,1.,0.5),inv(2,1,1.,0.5)];
  let free=vec![true,false,false];let roots=[1];let chosen=vec![None,Some(Binding{instance:0,pin:0}),None];
  let witness=vec![None,Some(Binding{instance:0,pin:0}),Some(Binding{instance:1,pin:0})];
  let old=chosen.clone();let mut trial=chosen;trial[1]=Some(Binding{instance:3,pin:0});
  let (alternate,_)=delay_frontier_seed(&m,&free,Instant::now()+std::time::Duration::from_secs(1)).unwrap();
  joint_input_cone(&m,&mut trial,&witness,&free,&roots,Some(&alternate),&old,&mut MapStats::default(),Instant::now()+std::time::Duration::from_secs(1)).unwrap();
  assert_eq!(metrics(&m,&trial,&free,&roots),(1.,2.));
  assert_eq!(metrics(&m,&old,&free,&roots),(1.,10.));
 }

 #[test] fn joint_disabled_or_expired_does_not_admit_cyclic_trial(){
  let mut slow=inv(0,1,10.,1.);slow.outputs[0].function=2;
  let m=vec![slow,inv(1,2,0.1,0.1),inv(0,2,1.,0.5),inv(2,1,1.,0.5)];let free=vec![true,false,false];
  let old=vec![None,Some(Binding{instance:0,pin:0}),None];let w=vec![None,old[1],Some(Binding{instance:1,pin:0})];let alternate=vec![None,old[1],Some(Binding{instance:2,pin:0})];
  for expired in [false,true]{let mut t=old.clone();t[1]=Some(Binding{instance:3,pin:0});let mut stats=MapStats::default();assert!(joint_input_cone(&m,&mut t,&w,&free,&[1],if expired{Some(&alternate)}else{None},&old,&mut stats,if expired{Instant::now()}else{Instant::now()+std::time::Duration::from_secs(1)}).is_err());assert_eq!(stats.joint_attempts,0);}
  assert!(!MapParams::default().joint_cones);
 }
 #[test] fn valid_legacy_repair_is_unchanged_even_if_alternate_is_faster(){
  let mut b=inv(0,1,10.,1.);b.outputs[0].function=2;
  let m=vec![b,inv(0,2,5.,0.1),inv(0,2,1.,0.5),inv(2,1,1.,0.5)];let free=vec![true,false,false];let old=vec![None,Some(Binding{instance:0,pin:0}),None];let w=vec![None,old[1],Some(Binding{instance:1,pin:0})];let a=vec![None,old[1],Some(Binding{instance:2,pin:0})];let mut t=old.clone();t[1]=Some(Binding{instance:3,pin:0});let mut stats=MapStats::default();joint_input_cone(&m,&mut t,&w,&free,&[1],Some(&a),&old,&mut stats,Instant::now()+std::time::Duration::from_secs(1)).unwrap();assert_eq!(metrics(&m,&t,&free,&[1]),(f64::from(0.1f32)+0.5,6.));assert_eq!(stats.joint_attempts,0);
 }
 #[test] fn joint_repair_preserves_existing_shared_output_and_charges_once(){
  let mut slow=inv(0,1,10.,1.);slow.outputs[0].function=2;
  let mut shared=inv(0,2,1.,0.5);shared.outputs[0].targets.push(3);
  let m=vec![slow,inv(1,2,0.1,0.1),shared,inv(2,1,1.,0.5)];let free=vec![true,false,false,false];let old=vec![None,Some(Binding{instance:0,pin:0}),None,Some(Binding{instance:2,pin:0})];let w=vec![None,old[1],Some(Binding{instance:1,pin:0}),old[3]];let a=vec![None,old[1],old[3],old[3]];let mut t=old.clone();t[1]=Some(Binding{instance:3,pin:0});joint_input_cone(&m,&mut t,&w,&free,&[1,3],Some(&a),&old,&mut MapStats::default(),Instant::now()+std::time::Duration::from_secs(1)).unwrap();assert_eq!(t[3].map(|b|(b.instance,b.pin)),old[3].map(|b|(b.instance,b.pin)));assert_eq!(metrics(&m,&t,&free,&[1,3]),(1.,2.));
  for x in [false,true]{fn value(s:usize,x:bool,m:&[Instance],c:&[Option<Binding>])->bool{if s==0{return x}let b=c[s].unwrap();let i=&m[b.instance];((i.outputs[b.pin].function>>usize::from(value(i.inputs[0],x,m,c)))&1)!=0}assert_eq!(value(1,x,&m,&t),x);assert_eq!(value(3,x,&m,&t),!x);}
 }
}
// Original valid trials remain byte-for-byte unchanged. Only failed repair can
// graft missing input phases from the existing all-signal fast witness. Every
// formerly selected binding, including forced target/phase pairs, is preserved.
// No candidate, logical equality, separate search or extra time window is created.
fn joint_input_cone(instances:&[Instance],trial:&mut[Option<Binding>],witness:&[Option<Binding>],free:&[bool],roots:&[usize],alternate:Option<&[Option<Binding>]>,incumbent:&[Option<Binding>],stats:&mut MapStats,deadline:Instant)->Result<(),String>{
 match fill_from_witness(instances,trial,witness,free,roots) {
  Ok(())=>Ok(()),
  Err(original)=>{
   let Some(alternate)=alternate else{return Err(original)};
   if Instant::now()>=deadline{return Err(original)}
   stats.joint_attempts+=1;
   // Failed prune does not clear bindings; remove only the old witness's newly
   // filled cone. All callers force bindings only for previously selected signals.
   for s in 0..trial.len(){if incumbent[s].is_none(){trial[s]=None;}}
   fill_from_witness(instances,trial,alternate,free,roots)?;
   if Instant::now()>=deadline{return Err("joint repair deadline".into())}
   stats.joint_valid+=1;Ok(())
  }
 }
}

#[cfg(test)] mod joint_pool_tests {
 use super::*;
 fn edge(input:usize,target:usize,delay:f32,area:f32,function:u64)->Instance{Instance{cell:format!("G{input}_{target}_{delay}"),area,inputs:vec![input],input_pins:vec!["A".into()],outputs:vec![OutputPin{sources:Vec::new(),name:"Y".into(),delays:vec![delay],function,targets:vec![target]}]}}
 #[test] fn cap_preserves_joint_fast_input_phase_cone(){
  let m=vec![edge(0,1,10.,0.1,2),edge(0,2,100.,0.1,1),edge(0,2,1.,1.,1),edge(2,1,1.,0.1,1),edge(0,1,5.,1.,2),edge(0,1,4.,2.,2),edge(0,1,3.,3.,2),edge(0,3,100.,0.05,2),edge(3,1,0.1,0.2,2)];
  let free=vec![true,false,false,false];let mut capped=m.clone();let prior=std::env::var_os("EGM_CAP_DIVERSITY");std::env::set_var("EGM_CAP_DIVERSITY","1");let r=cap_instances_joint(&mut capped,&free,&[1],5,true,true,true);if let Some(v)=prior{std::env::set_var("EGM_CAP_DIVERSITY",v)}else{std::env::remove_var("EGM_CAP_DIVERSITY")}r.unwrap();
  assert!(capped.len()<=5);let (mut c,_)=delay_frontier_seed(&capped,&free,Instant::now()+std::time::Duration::from_secs(1)).unwrap();prune(&capped,&mut c,&free,&[1]).unwrap();assert_eq!(metrics(&capped,&c,&free,&[1]).1,2.);
 }
}


// This certificate only concerns the immutable supplied physical pool and the
// unloaded dependent-pin model. It does not prove area/lex optimality or optimal
// mapping over omitted cuts, library matches, rules or source representations.
fn pool_delay_certificate(instances:&[Instance],free:&[bool],roots:&[usize],full:&[Option<Binding>])->Result<serde_json::Value,String>{
 let all:Vec<_>=full.iter().enumerate().filter_map(|(s,b)|b.map(|_|s)).collect();let mut checked=full.to_vec();prune(instances,&mut checked,free,&all)?;
 let (arrival,_)=critical_arrivals(instances,&checked,free,&all);let mut violations=0usize;let mut arcs=0usize;let mut max_violation=0f64;
 for m in instances {for o in &m.outputs {
  let rhs=m.inputs.iter().enumerate().filter(|(p,_)|o.has_arc(*p)).map(|(p,&input)|arrival[input].unwrap_or(f64::INFINITY)+f64::from(o.delays[p])).fold(0f64,f64::max);
  for &target in &o.targets {arcs+=1;let lhs=arrival[target].unwrap_or(f64::INFINITY);if rhs.is_finite() && lhs>rhs+1e-6{violations+=1;max_violation=max_violation.max(lhs-rhs);}}
 }}
 let bound=roots.iter().map(|&s|arrival[s].unwrap_or(f64::INFINITY)).fold(0f64,f64::max);let mut cover=full.to_vec();prune(instances,&mut cover,free,roots)?;let actual=metrics(instances,&cover,free,roots);
 Ok(serde_json::json!({"instances":instances.len(),"producer_target_inequalities":arcs,"violations":violations,"max_violation":if max_violation.is_finite(){Some(max_violation)}else{None},"root_label_bound":bound,"attaining_cover_delay":actual.1,"attaining_cover_area":actual.0,"pool_delay_certificate_pass":violations==0 && (actual.1-bound).abs()<1e-6,"scope":"immutable supplied candidate pool; dependent-pin unloaded delay only; no area/lex/global-library optimality claim"}))
}
fn write_frontier_audit(path:Option<&str>,enabled:bool,area:usize,union:usize,secs:f64,before:serde_json::Value,after:serde_json::Value)->Result<(),String>{
 if let Some(p)=path {let p=std::path::Path::new(p).with_extension("frontier.json");std::fs::write(p,serde_json::json!({"timing_witness_pool":enabled,"area_witness_instances":area,"union_witness_instances":union,"frontier_generation_secs":secs,"generation_inside_native_mapping_guard":true,"recovery_window_unchanged":true,"before":before,"after":after}).to_string()).map_err(|e|e.to_string())?;}Ok(())
}

#[cfg(test)]mod pool_certificate_tests{
 use super::*;
 fn edge(input:usize,target:usize,delay:f32)->Instance{Instance{cell:"BUF".into(),area:1.,inputs:vec![input],input_pins:vec!["A".into()],outputs:vec![OutputPin{sources:Vec::new(),name:"Y".into(),delays:vec![delay],function:2,targets:vec![target]}]}}
 #[test]fn all_producer_certificate_matches_exhaustive_small_dags(){
  for seed in 0..32 {let mut m=Vec::new();for s in 1..5{for p in 0..3{m.push(edge((seed*7+s+p)%s,s,1.+((seed*3+s+p*5)%9)as f32));}}
   let free=vec![true,false,false,false,false];let (fast,_)=delay_frontier_seed(&m,&free,Instant::now()+std::time::Duration::from_secs(1)).unwrap();let cert=pool_delay_certificate(&m,&free,&[4],&fast).unwrap();assert_eq!(cert["pool_delay_certificate_pass"],true);
   let mut minimum=f64::INFINITY;for choice in 0..81 {let mut c=vec![None;5];let mut q=choice;for s in 1..5{c[s]=Some(Binding{instance:(s-1)*3+q%3,pin:0});q/=3;}prune(&m,&mut c,&free,&[4]).unwrap();minimum=minimum.min(metrics(&m,&c,&free,&[4]).1);}assert_eq!(cert["root_label_bound"].as_f64().unwrap(),minimum);
  }
 }
 #[test]fn conservative_nonarc_frontier_is_not_falsely_certified(){
  let slow=edge(0,1,100.);let mut nonarc=edge(0,2,1.);nonarc.inputs.push(1);nonarc.input_pins.push("IGNORED".into());nonarc.outputs[0].delays.push(100.);nonarc.outputs[0].function=10;
  let m=vec![slow,nonarc,edge(0,2,50.)];let free=vec![true,false,false];let (fast,conservative)=delay_frontier_seed(&m,&free,Instant::now()+std::time::Duration::from_secs(1)).unwrap();assert_eq!(conservative,1);let c=pool_delay_certificate(&m,&free,&[2],&fast).unwrap();assert_eq!(c["pool_delay_certificate_pass"],false);assert!(c["violations"].as_u64().unwrap()>0);
 }
 #[test]fn certificate_rejects_a_slower_feasible_cover(){let m=vec![edge(0,1,10.),edge(0,1,1.)];let c=vec![None,Some(Binding{instance:0,pin:0})];let x=pool_delay_certificate(&m,&[true,false],&[1],&c).unwrap();assert_eq!(x["pool_delay_certificate_pass"],false);assert_eq!(x["violations"],1);}
}

#[cfg(test)]mod online_area_tests {
 use super::*;
 #[test]fn scored_cover_can_accumulate_independent_area_savings_without_replay(){
  let make=|target,area|Instance{cell:format!("BUF{target}_{area}"),area,inputs:vec![0],input_pins:vec!["A".into()],outputs:vec![OutputPin{sources:Vec::new(),name:"Y".into(),delays:vec![1.],function:2,targets:vec![target]}]};
  let m=vec![make(1,8.),make(2,8.),make(1,1.),make(2,1.)];let free=vec![true,false,false];let roots=[1,2];let seed=vec![None,Some(Binding{instance:0,pin:0}),Some(Binding{instance:1,pin:0})];let ps=MapParams{objective:MapObjective::DelayArea,refine_rounds:1,refine_trials:0,online_area:true,..Default::default()};let mut stats=MapStats::default();let (c,v)=slack_area_pass(&m,&free,&roots,&seed,seed.clone(),(16.,1.),&ps,&mut stats,Instant::now()+std::time::Duration::from_secs(1),true);assert_eq!(v,(2.,1.));assert_eq!(metrics(&m,&c,&free,&roots),v);assert_eq!(stats.slack_local_trials,0);
  let off=MapParams{online_area:false,..ps.clone()};let mut old_stats=MapStats::default();let (_,vold)=slack_area_pass(&m,&free,&roots,&seed,seed.clone(),(16.,1.),&off,&mut old_stats,Instant::now()+std::time::Duration::from_secs(1),true);assert_eq!(vold,(9.,1.));assert!(stats.slack_local_scored<=old_stats.slack_local_scored,"online reuse may not add scoring calls in this fixture");
 }
 #[test]fn online_area_rejects_late_or_slower_joint_cover_and_charges_once(){
  let one=|s|Instance{cell:format!("BUF{s}"),area:8.,inputs:vec![0],input_pins:vec!["A".into()],outputs:vec![OutputPin{sources:Vec::new(),name:"Y".into(),delays:vec![1.],function:2,targets:vec![s]}]};let mut joint=one(1);joint.area=1.;let mut pin=joint.outputs[0].clone();pin.name="Z".into();pin.targets=vec![2];joint.outputs.push(pin);
  let mut m=vec![one(1),one(2),joint];let f=vec![true,false,false];let c=vec![None,Some(Binding{instance:0,pin:0}),Some(Binding{instance:1,pin:0})];let ps=MapParams{objective:MapObjective::DelayArea,refine_rounds:1,refine_trials:0,online_area:true,..Default::default()};let mut n=MapStats::default();let (out,v)=slack_area_pass(&m,&f,&[1,2],&c,c.clone(),(16.,1.),&ps,&mut n,Instant::now()+std::time::Duration::from_secs(1),true);assert_eq!(v,(1.,1.));assert_eq!(out[1].unwrap().instance,out[2].unwrap().instance);
  for pin in &mut m[2].outputs{pin.delays[0]=2.;}let (_,v)=slack_area_pass(&m,&f,&[1,2],&c,c.clone(),(16.,1.),&ps,&mut MapStats::default(),Instant::now()+std::time::Duration::from_secs(1),true);assert_eq!(v,(16.,1.));
  let mut n=MapStats::default();let (_,v)=slack_area_pass(&m,&f,&[1,2],&c,c.clone(),(16.,1.),&ps,&mut n,Instant::now(),true);assert_eq!(v,(16.,1.));assert_eq!(n.online_area_updates,0);
 }
}

#[cfg(test)] thread_local! { static TEST_LOCAL_START:std::cell::Cell<Option<(f64,f64)>>=const {std::cell::Cell::new(None)}; }

#[cfg(test)]mod active_joint_tests{use super::*;#[test]
    fn fast_deferred_cover_is_the_local_working_reference() {
        let one=|area,delay|Instance{cell:format!("cell_{area}_{delay}"),area,inputs:vec![0],input_pins:vec!["A".into()],outputs:vec![OutputPin{sources:Vec::new(),name:"Y".into(),delays:vec![delay],function:2,targets:vec![1]}]};
        let cells=vec![one(1.,10.),one(2.,1.)];let free=vec![true,false];let roots=vec![1];
        let ps=MapParams{activate_deferred:true,objective:MapObjective::DelayArea,strict_budgets:true,fast_delay_seed:true,required_area_dag:true,defer_area_dag:true,preserve_legacy_seeds:true,metric_epochs:true,flow_worklist:true,seed_window_fraction:0.,refine_rounds:1,refine_trials:0,recover_live_refs:false,recover_cones:false,..Default::default()};
        let mut stats=MapStats::default();let cover=heuristic_cover(&cells,&free,&roots,&ps,&mut stats).unwrap();
        assert_eq!(metrics(&cells,&cover,&free,&roots),(2.,1.));
        assert_eq!(TEST_LOCAL_START.with(|v|v.get()),Some((2.,1.)),"later recovery must use the faster validated deferred cover, not slow legacy10");
        let off=MapParams{activate_deferred:false,..ps.clone()};let old=heuristic_cover(&cells,&free,&roots,&off,&mut MapStats::default()).unwrap();assert_eq!(metrics(&cells,&old,&free,&roots),(2.,1.));assert_eq!(TEST_LOCAL_START.with(|v|v.get()),Some((1.,10.)));
        let same_delay=vec![one(1.,1.),one(2.,1.)];let shared=heuristic_cover(&same_delay,&free,&roots,&ps,&mut MapStats::default()).unwrap();assert_eq!(metrics(&same_delay,&shared,&free,&roots),(1.,1.));assert_eq!(TEST_LOCAL_START.with(|v|v.get()),Some((1.,1.)),"equal delay must not replace cheaper legacy trajectory");

    }
}

#[cfg(test)]mod online_batch_tests {
 use super::*;
 #[test]fn score_batch_avoids_first_joint_tie_locking_both_outputs(){
  let make=|target,area|Instance{cell:format!("B{target}_{area}"),area,inputs:vec![0],input_pins:vec!["A".into()],outputs:vec![OutputPin{sources:Vec::new(),name:"Y".into(),delays:vec![1.],function:2,targets:vec![target]}]};
  let mut joint=make(1,9.);let mut y=joint.outputs[0].clone();y.name="Z".into();y.targets=vec![2];joint.outputs.push(y);
  let mut m=vec![make(1,8.),make(2,8.),joint,make(1,1.),make(2,1.)];while m.len()<2048{m.push(make(3,1.));}
  let free=vec![true,false,false,false];let seed=vec![None,Some(Binding{instance:0,pin:0}),Some(Binding{instance:1,pin:0}),None];let ps=MapParams{objective:MapObjective::DelayArea,online_area:true,refine_rounds:1,refine_trials:0,..Default::default()};let mut stats=MapStats::default();let (_,v)=slack_area_pass(&m,&free,&[1,2],&seed,seed.clone(),(16.,1.),&ps,&mut stats,Instant::now()+std::time::Duration::from_secs(1),true);assert_eq!(v,(2.,1.),"equal whole-cover score must not greedily lock an expensive shared producer");assert_eq!(stats.slack_local_trials,0);
 }
}

// The batch contains only a full-cover score from the current immutable pool.
// Flushing reuses that validation; no additional metric/repair/trial is executed.
type OnlineAreaBatch=Option<(Vec<Option<Binding>>,(f64,f64),f32,usize)>;
fn commit_online_area(batch:&mut OnlineAreaBatch,chosen:&mut Vec<Option<Binding>>,best:&mut(f64,f64),ps:&MapParams,stats:&mut MapStats)->bool{
 if let Some((cover,value,_,_))=batch.take(){if value.0<best.0-1e-6 && value.1<=if ps.fixed_area_bound && ps.objective==MapObjective::AreaDelay {ps.max_delay.unwrap_or(best.1)+1e-6}else{best.1+1e-6} && better_cover(value,*best,ps){*chosen=cover;*best=value;stats.online_area_updates+=1;stats.refinement_accepted+=1;return true;}}false
}


fn shared_area_params(ps:&MapParams,activated:bool)->MapParams{let mut result=ps.clone();if !activated {result.introduce_phase_inverters=false;}if ps.preserve_shared_trajectory && !activated {result.online_area=false;}result}
#[cfg(test)]mod shared_incumbent_guard_tests {
 use super::*;
 #[test]fn introduced_phase_preserves_unactivated_legacy_trajectory(){
  let p=MapParams{introduce_phase_inverters:true,..Default::default()};
  assert!(!shared_area_params(&p,false).introduce_phase_inverters);
  assert!(shared_area_params(&p,true).introduce_phase_inverters);
 }

 fn buf(input:usize,target:usize,area:f32,delay:f32)->Instance{Instance{cell:format!("B{input}_{target}_{area}"),area,inputs:vec![input],input_pins:vec!["A".into()],outputs:vec![OutputPin{sources:Vec::new(),name:"Y".into(),delays:vec![delay],function:2,targets:vec![target]}]}}
 #[test]fn equal_delay_online_commit_loses_a_complete_shared_incumbent(){
  let mut joint=buf(1,2,3.6,0.);let mut second=joint.outputs[0].clone();second.name="Z".into();second.targets=vec![3];joint.outputs.push(second);
  let m=vec![buf(0,1,8.,1.),buf(0,2,8.,1.),buf(0,3,8.,1.),joint,buf(0,3,2.4,0.)];
  let free=[true,false,false,false];let roots=[2,3];let full=vec![None,Some(Binding{instance:0,pin:0}),Some(Binding{instance:1,pin:0}),Some(Binding{instance:2,pin:0})];let mut chosen=full.clone();prune(&m,&mut chosen,&free,&roots).unwrap();
  let ps=MapParams{objective:MapObjective::DelayArea,preserve_shared_trajectory:true,online_area:true,refine_rounds:1,refine_trials:20,..Default::default()};
  let effective=shared_area_params(&ps,false);let (c,v)=slack_area_pass(&m,&free,&roots,&full,chosen,(16.,1.),&effective,&mut MapStats::default(),Instant::now()+std::time::Duration::from_secs(1),true);
  assert!((v.0-10.4).abs()<1e-5,"complete old shared incumbent lost: {v:?}");assert_eq!(v.1,1.);assert_eq!(metrics(&m,&c,&free,&roots),v);
 }
}

#[cfg(test)]mod pool_shared_guard_tests{use super::*;
 fn buf(s:usize,area:f32,delay:f32)->Instance{Instance{cell:format!("B{s}_{area}"),area,inputs:vec![0],input_pins:vec!["A".into()],outputs:vec![OutputPin{sources:Vec::new(),name:"Y".into(),delays:vec![delay],function:2,targets:vec![s]}]}}
 #[test]fn unused_fast_phase_cannot_displace_old_joint_area_cover(){
  let mut joint=buf(1,1.5,1.);let mut out=joint.outputs[0].clone();out.name="Z".into();out.targets=vec![3];joint.outputs.push(out);
  let mut m=vec![buf(1,1.,1.),buf(2,1.,10.),buf(3,1.,1.),joint,buf(2,2.,1.)];let free=[true,false,false,false];cap_instances_guarded(&mut m,&free,&[1,3],4,true,true,true,true).unwrap();
  assert!(m.iter().any(|m|m.outputs.len()==2),"irrelevant fast phase evicted the complete old shared-area cover at identical root delay");assert_eq!(m.len(),4);
 }
}

fn fill_optional_pool(instances:&[Instance],free:&[bool],budget:usize,timing_cap:bool,witness:&[Option<Binding>],mut keep:BTreeSet<usize>)->BTreeSet<usize>{
    let mut producers = vec![Vec::new(); free.len()];
    for (i, m) in instances.iter().enumerate() {
        for o in &m.outputs {
            for &s in &o.targets {
                producers[s].push(i);
            }
        }
    }
    let witness_roots: Vec<_> = witness
        .iter()
        .enumerate()
        .filter_map(|(s, b)| b.map(|_| s))
        .collect();
    let mixed_cap = std::env::var("EGM_CAP_DIVERSITY").as_deref()==Ok("1");
    let arrival = if timing_cap || mixed_cap {
        critical_arrivals(instances, &witness, free, &witness_roots).0
    } else {
        vec![None; free.len()]
    };
    let key = |i: usize, target: usize, timed: bool| {
        let m = &instances[i];
        let time = if timed {
            m.outputs
                .iter()
                .filter(|o| o.targets.contains(&target))
                .map(|o| {
                    m.inputs
                        .iter()
                        .enumerate()
                        .filter(|(p, _)| o.has_arc(*p))
                        .map(|(p, &s)| arrival[s].unwrap_or(f64::INFINITY) + f64::from(o.delays[p]))
                        .fold(0.0, f64::max)
                })
                .fold(f64::INFINITY, f64::min)
        } else {
            m.outputs
                .iter()
                .flat_map(|o| o.delays.iter())
                .copied()
                .fold(0.0, f32::max) as f64
        };
        (time, m.area, i)
    };
    for (target, bucket) in producers.iter_mut().enumerate() {
        bucket.sort_by(|&a, &b| {
            let x = key(a, target, timing_cap);
            let y = key(b, target, timing_cap);
            x.0.total_cmp(&y.0)
                .then(x.1.total_cmp(&y.1))
                .then(x.2.cmp(&y.2))
        });
        bucket.dedup();
        if mixed_cap {
            let mut intrinsic=bucket.clone(); let mut timed=bucket.clone();
            let cmp = |a:usize,b:usize,t:bool| {
                let x=key(a,target,t);let y=key(b,target,t);
                x.0.total_cmp(&y.0).then(x.1.total_cmp(&y.1)).then(x.2.cmp(&y.2))
            };
            intrinsic.sort_by(|&a,&b|cmp(a,b,false));
            timed.sort_by(|&a,&b|cmp(a,b,true));
            *bucket=interleave_ranks(&intrinsic,&timed);
        }
    }
    let mut positions = vec![0; free.len()];
    loop {
        let before = keep.len();
        for (s, bucket) in producers.iter().enumerate() {
            while positions[s] < bucket.len() && keep.contains(&bucket[positions[s]]) {
                positions[s] += 1;
            }
            if keep.len() >= budget {
                break;
            }
            if positions[s] < bucket.len() {
                keep.insert(bucket[positions[s]]);
                positions[s] += 1;
            }
        }
        if keep.len() >= budget || keep.len() == before {
            break;
        }
    }
    keep
}

#[cfg(test)]mod legacy_guard_policy_tests{use super::*;
 #[test]fn guard_preserves_disabled_behavior_and_faster_branch(){let p=MapParams{online_area:true,..Default::default()};assert!(shared_area_params(&p,false).online_area);let on=MapParams{preserve_shared_trajectory:true,..p};assert!(shared_area_params(&on,true).online_area);assert!(!shared_area_params(&on,false).online_area);assert_eq!(on.refine_trials,shared_area_params(&on,false).refine_trials);assert_eq!(on.refine_time_secs,shared_area_params(&on,false).refine_time_secs);}
}

#[derive(Default)]struct FusedScoreWorkspace { stamps:Vec<u32>, visiting:Vec<bool>, values:Vec<f64>, live:Vec<usize>, epoch:u32 }
thread_local!{static FUSED_SCORE_WORKSPACE:std::cell::RefCell<FusedScoreWorkspace>=std::cell::RefCell::new(FusedScoreWorkspace::default());}
fn validate_score_trial(instances:&[Instance],chosen:&mut[Option<Binding>],witness:&[Option<Binding>],free:&[bool],roots:&[usize],fused:bool)->Result<(f64,f64),String>{
 if !fused {fill_from_witness(instances,chosen,witness,free,roots)?;return Ok(metrics(instances,chosen,free,roots));}
 fn visit(s:usize,m:&[Instance],c:&mut[Option<Binding>],w:&[Option<Binding>],free:&[bool],cache:&mut FusedScoreWorkspace)->Result<f64,String>{
  if free[s]{return Ok(0.)}
  if cache.stamps[s]==cache.epoch{return if cache.visiting[s]{Err(format!("cycle through signal {s}"))}else{Ok(cache.values[s])}}
  cache.stamps[s]=cache.epoch;cache.visiting[s]=true;
  let b=c[s].or(w[s]).ok_or_else(||format!("unrealizable dependency {s}"))?;
  let instance=m.get(b.instance).ok_or_else(||format!("invalid instance for signal {s}"))?;
  let output=instance.outputs.get(b.pin).ok_or_else(||format!("invalid pin for signal {s}"))?;
  if !output.targets.contains(&s){return Err(format!("invalid binding for signal {s}"))}
  c[s]=Some(b);let mut value=0f64;
  for (p,&input) in instance.inputs.iter().enumerate(){let child=visit(input,m,c,w,free,cache)?;if output.has_arc(p){value=value.max(child+f64::from(output.delays[p]));}}
  cache.visiting[s]=false;cache.values[s]=value;cache.live.push(b.instance);Ok(value)
 }
 FUSED_SCORE_WORKSPACE.with(|workspace|{
  let mut cache=workspace.borrow_mut();cache.stamps.resize(free.len(),0);cache.visiting.resize(free.len(),false);cache.values.resize(free.len(),0.);cache.live.clear();cache.epoch=cache.epoch.wrapping_add(1);if cache.epoch==0{cache.stamps.fill(0);cache.epoch=1;}
  let mut delay=0f64;for &s in roots {delay=delay.max(visit(s,instances,chosen,witness,free,&mut cache)?);}
  for (s,b) in chosen.iter_mut().enumerate(){if cache.stamps[s]!=cache.epoch || free[s]{*b=None;}}
  cache.live.sort_unstable();cache.live.dedup();let area=cache.live.iter().map(|&i|f64::from(instances[i].area)).sum();Ok((area,delay))
 })
}

#[cfg(test)]mod fused_score_tests{use super::*;
 fn cell(inputs:Vec<usize>,s:usize,area:f32,delays:Vec<f32>,function:u64)->Instance{Instance{cell:format!("C{s}"),area,input_pins:(0..inputs.len()).map(|p|format!("I{p}")).collect(),inputs,outputs:vec![OutputPin{sources:Vec::new(),name:"Y".into(),delays,function,targets:vec![s]}]}}
 #[test]fn fused_validation_matches_independent_complete_prune_and_metrics(){
  for seed in 0..64 {let mut m=Vec::new();for s in 1..9 {m.push(cell(vec![(s+seed)%s,(s*3+seed*7)%s],s,(s+seed%5)as f32/10.,vec![1.+(seed%3)as f32,0.5+(s%3)as f32],if seed%2==0{8}else{10}));}
   let f:Vec<_>=(0..9).map(|i|i==0).collect();let w:Vec<_>=(0..9).map(|s|(s>0).then(||Binding{instance:s-1,pin:0})).collect();
   for root in [1,4,8] {for partial in 0..16 {let mut a=w.clone();for s in 1..9{if partial&(1<<(s%4))!=0{a[s]=None;}}let mut b=a.clone();fill_from_witness(&m,&mut a,&w,&f,&[root]).unwrap();let expected=metrics(&m,&a,&f,&[root]);assert_eq!(validate_score_trial(&m,&mut b,&w,&f,&[root],true).unwrap(),expected);assert_eq!(a.iter().map(|v|v.map(|b|(b.instance,b.pin))).collect::<Vec<_>>(),b.iter().map(|v|v.map(|b|(b.instance,b.pin))).collect::<Vec<_>>());}}
  }
 }
 #[test]fn fused_joint_pin_area_nonarc_connectivity_and_epoch_wrap(){
  let mut joint=cell(vec![0],1,3.,vec![1.],2);let mut second=joint.outputs[0].clone();second.name="Z".into();second.targets=vec![2];second.delays=vec![2.];joint.outputs.push(second);
  let m=vec![joint,cell(vec![1,2],3,1.,vec![5.,99.],10)];let f=vec![true,false,false,false];let w=vec![None,Some(Binding{instance:0,pin:0}),Some(Binding{instance:0,pin:1}),Some(Binding{instance:1,pin:0})];let mut c=w.clone();assert_eq!(validate_score_trial(&m,&mut c,&w,&f,&[3],true).unwrap(),(4.,6.));
  FUSED_SCORE_WORKSPACE.with(|v|v.borrow_mut().epoch=u32::MAX);let mut c=w.clone();assert_eq!(validate_score_trial(&m,&mut c,&w,&f,&[1],true).unwrap(),(3.,1.));assert!(c[2].is_none()&&c[3].is_none());
  let cycle=vec![cell(vec![1],1,1.,vec![0.],2)];let mut c=vec![None,Some(Binding{instance:0,pin:0})];assert!(validate_score_trial(&cycle,&mut c,&[None,None],&[true,false],&[1],true).is_err());
  let mut c=vec![None,None];assert!(validate_score_trial(&cycle,&mut c,&[None,None],&[true,false],&[1],true).is_err());
 }
}

#[cfg(test)]mod fused_score_benchmark{use super::*;
 #[test]#[ignore="explicit microbenchmark, no timing assertion"]fn fixed_trial_validation_microbenchmark(){
  let _guard=MetricEpochGuard::enter(true);let n=1024;let mut m=Vec::new();for s in 1..n{m.push(Instance{cell:format!("B{s}"),area:1.,inputs:vec![s/2],input_pins:vec!["A".into()],outputs:vec![OutputPin{sources:Vec::new(),name:"Y".into(),delays:vec![1.],function:2,targets:vec![s]}]});}
  let f:Vec<_>=(0..n).map(|s|s==0).collect();let w:Vec<_>=(0..n).map(|s|(s>0).then(||Binding{instance:s-1,pin:0})).collect();let roots:Vec<_>=(n/2..n).collect();let mut answers=Vec::new();
  for fused in [false,true]{let t=Instant::now();let mut value=(0.,0.);for _ in 0..4000{let mut c=w.clone();value=std::hint::black_box(validate_score_trial(&m,&mut c,&w,&f,&roots,fused).unwrap());}answers.push(value);println!("fused={fused} iterations=4000 nanos={} value={value:?}",t.elapsed().as_nanos());}assert_eq!(answers[0],answers[1]);
 }
}

fn shared_area_counters(instances:&[Instance],chosen:&[Option<Binding>],roots:&[usize])->(Vec<usize>,Vec<usize>){
 let mut refs=vec![0;chosen.len()];let mut users=vec![0;instances.len()];for b in chosen.iter().flatten(){users[b.instance]+=1;}for s in roots.iter().copied().collect::<BTreeSet<_>>(){refs[s]+=1;}for (i,m) in instances.iter().enumerate(){if users[i]>0 {for s in m.inputs.iter().copied().collect::<BTreeSet<_>>(){refs[s]+=1;}}}(refs,users)
}
// Exact physical-area accounting for replacements whose inputs already exist.
// This is ONLY a rejection filter. A surviving trial still uses the full
// witness/cycle/prune/pin-delay oracle; no timing label is an acceptance proof.
fn selected_input_area(instances:&[Instance],chosen:&[Option<Binding>],free:&[bool],i:usize,area:f64,signal_refs:&[usize],output_users:&[usize])->Option<f64>{
 let m=&instances[i];
 if m.inputs.iter().any(|&s|!free[s] && chosen[s].is_none()){return None;}
 let mut replace=HashMap::new();
 for (pin,o) in m.outputs.iter().enumerate(){for &s in &o.targets{if chosen[s].is_some(){replace.insert(s,Binding{instance:i,pin});}}}
 let targets:Vec<_>=replace.keys().copied().filter(|&s|chosen[s].is_some_and(|b|b.instance!=i || b.pin!=replace[&s].pin)).collect();
 if targets.is_empty(){return Some(area);}
 let mut refs=HashMap::new();let mut users=HashMap::new();let mut removed=HashSet::new();
 let mut delta=0.;
 // Activate BEFORE deref: even a producer exclusively used by the old cell
 // remains live when the new cell requires its output. Unique physical inputs
 // and output users charge aliases/joint pins exactly once per instance.
 if output_users[i]==0 {
  delta+=f64::from(m.area);
  for input in m.inputs.iter().copied().collect::<BTreeSet<_>>(){*refs.entry(input).or_insert(signal_refs[input])+=1;}
 }
 *users.entry(i).or_insert(output_users[i])+=targets.len();
 let mut pending=Vec::new();
 for &s in &targets {
  let old=chosen[s].unwrap().instance;
  let count=users.entry(old).or_insert(output_users[old]);*count-=1;
  if *count==0{pending.push(old);}
 }
 while let Some(old)=pending.pop(){
  if !removed.insert(old){continue;}
  delta-=f64::from(instances[old].area);
  for input in instances[old].inputs.iter().copied().collect::<BTreeSet<_>>(){
   let count=refs.entry(input).or_insert(signal_refs[input]);*count-=1;
   if *count!=0 || free[input]{continue;}
   let b=replace.get(&input).copied().or(chosen[input])?;
   let count=users.entry(b.instance).or_insert(output_users[b.instance]);*count-=1;
   if *count==0 {pending.push(b.instance);}
  }
 }
 Some(area+delta)
}
fn shared_area_lower_bound(instances:&[Instance],chosen:&[Option<Binding>],free:&[bool],targets:&[usize],i:usize,area:f64,refs:&[usize],users:&[usize],active:&BTreeSet<usize>)->f64{
 if targets.is_empty(){return area;}
 let removed=reclaimable_area(targets,instances,chosen,free,refs,users);let added=if active.contains(&i){0.}else{f64::from(instances[i].area)};
 let roundoff=8.*f64::EPSILON*(instances.len()+chosen.len()+1)as f64*(area.abs()+removed.abs()+added.abs()+1.);
 area-removed+added-roundoff
}
#[cfg(test)]mod shared_area_bound_tests{use super::*;
 fn cell(input:usize,target:usize,area:f32)->Instance{Instance{cell:format!("B{target}"),area,inputs:vec![input],input_pins:vec!["A".into()],outputs:vec![OutputPin{sources:Vec::new(),name:"Y".into(),delays:vec![1.],function:2,targets:vec![target]}]}}
 #[test]fn bound_never_exceeds_actual_complete_trial_area(){
  for seed in 0..64 {let mut m:Vec<_>=(1..8).map(|s|cell((s+seed)%s,s,1.+(s%3)as f32/10.)).collect();for target in 1..8{let mut v=cell((target*5+seed)%target,target,0.2+((seed+target)%8)as f32/10.);if target<7 && seed%2==0{let mut o=v.outputs[0].clone();o.name="Z".into();o.targets=vec![target+1];v.outputs.push(o);}m.push(v);}
   let free:Vec<_>=(0..8).map(|s|s==0).collect();let full:Vec<_>=(0..8).map(|s|(s>0).then(||Binding{instance:s-1,pin:0})).collect();let roots=[5,6,7];let mut chosen=full.clone();prune(&m,&mut chosen,&free,&roots).unwrap();let value=metrics(&m,&chosen,&free,&roots);let (refs,users)=shared_area_counters(&m,&chosen,&roots);let active=chosen.iter().flatten().map(|b|b.instance).collect();
   for (i,m0) in m.iter().enumerate(){let targets:Vec<_>=m0.outputs.iter().flat_map(|o|o.targets.iter().copied()).filter(|&s|chosen[s].is_some_and(|b|b.instance!=i)).collect::<BTreeSet<_>>().into_iter().collect();let lower=shared_area_lower_bound(&m,&chosen,&free,&targets,i,value.0,&refs,&users,&active);let mut trial=chosen.clone();for (pin,o) in m0.outputs.iter().enumerate(){for &s in &o.targets{if trial[s].is_some(){trial[s]=Some(Binding{instance:i,pin});}}}if let Ok(actual)=validate_score_trial(&m,&mut trial,&full,&free,&roots,true){assert!(lower<=actual.0+1e-10,"seed{seed} instance{i} lower{lower} actual{actual:?}");}}
  }
 }
 #[test]fn bounded_area_pass_matches_unfiltered_old_trajectory_when_finished(){
  for n in [4,8,16,32]{let mut m:Vec<_>=(1..n).map(|s|cell(0,s,8.)).collect();for s in 1..n{m.push(cell(0,s,1.));}let free:Vec<_>=(0..n).map(|s|s==0).collect();let full:Vec<_>=(0..n).map(|s|(s>0).then(||Binding{instance:s-1,pin:0})).collect();let roots:Vec<_>=(1..n).collect();let p=MapParams{fused_cover_scoring:true,refine_rounds:4,refine_trials:2000,objective:MapObjective::DelayArea,..Default::default()};let (old,old_value)=slack_area_pass(&m,&free,&roots,&full,full.clone(),((n-1)as f64*8.,1.),&p,&mut MapStats::default(),Instant::now()+std::time::Duration::from_secs(1),true);let on=MapParams{shared_area_bound:true,..p};let (new,new_value)=slack_area_pass(&m,&free,&roots,&full,full.clone(),((n-1)as f64*8.,1.),&on,&mut MapStats::default(),Instant::now()+std::time::Duration::from_secs(1),true);assert_eq!(old_value,new_value);assert_eq!(old.iter().map(|v|v.map(|b|(b.instance,b.pin))).collect::<Vec<_>>(),new.iter().map(|v|v.map(|b|(b.instance,b.pin))).collect::<Vec<_>>());}
 }
}

#[cfg(test)]mod live_shared_area_tests{use super::*;
 fn buf(input:usize,target:usize,area:f32)->Instance{Instance{cell:format!("B{input}_{target}_{area}"),area,inputs:vec![input],input_pins:vec!["A".into()],outputs:vec![OutputPin{sources:Vec::new(),name:"Y".into(),delays:vec![1.],function:2,targets:vec![target]}]}}
 #[test]fn refreshed_online_bound_skips_nonimproving_score_after_commit(){let m=vec![buf(0,1,8.),buf(0,1,1.),buf(0,1,9.)];let f=[true,false];let w=vec![None,Some(Binding{instance:0,pin:0})];let p=MapParams{objective:MapObjective::DelayArea,refine_rounds:1,refine_trials:0,online_area:true,shared_area_bound:true,online_shared_bound:true,fused_cover_scoring:true,..Default::default()};let mut stats=MapStats::default();let(_,v)=slack_area_pass(&m,&f,&[1],&w,w.clone(),(8.,1.),&p,&mut stats,Instant::now()+std::time::Duration::from_secs(1),true);assert_eq!(v,(1.,1.));assert_eq!(stats.online_area_updates,1);assert_eq!(stats.slack_local_scored,1,"fresh sharing snapshot should skip the expensive final complete score");}
}

#[cfg(test)]mod refreshed_shared_bound_tests{use super::*;
 fn cell(input:usize,target:usize,area:f32)->Instance{Instance{cell:format!("B{target}_{area}"),area,inputs:vec![input],input_pins:vec!["A".into()],outputs:vec![OutputPin{sources:Vec::new(),name:"Y".into(),delays:vec![1.],function:2,targets:vec![target]}]}}
 #[test]fn stale_active_set_can_prune_real_shared_area_saving(){let mut j=cell(0,1,5.);let mut out=j.outputs[0].clone();out.name="Z".into();out.targets=vec![3];j.outputs.push(out);let m=vec![cell(0,2,10.),cell(0,3,1.),j,cell(1,2,0.1)];let free=[true,false,false,false];let old=vec![None,None,Some(Binding{instance:0,pin:0}),Some(Binding{instance:1,pin:0})];let mut chosen=vec![None,Some(Binding{instance:2,pin:0}),Some(Binding{instance:3,pin:0}),Some(Binding{instance:1,pin:0})];prune(&m,&mut chosen,&free,&[2,3]).unwrap();let best=metrics(&m,&chosen,&free,&[2,3]);let(refs,users)=shared_area_counters(&m,&chosen,&[2,3]);let fresh=chosen.iter().flatten().map(|b|b.instance).collect();let stale=old.iter().flatten().map(|b|b.instance).collect();let good=shared_area_lower_bound(&m,&chosen,&free,&[3],2,best.0,&refs,&users,&fresh);let bad=shared_area_lower_bound(&m,&chosen,&free,&[3],2,best.0,&refs,&users,&stale);let mut trial=chosen.clone();trial[3]=Some(Binding{instance:2,pin:1});let actual=validate_score_trial(&m,&mut trial,&chosen,&free,&[2,3],true).unwrap();assert!(actual.0<best.0 && actual.1<=best.1);assert!(good<=actual.0+1e-9);assert!(bad>best.0,"stale active membership can reject the actual saving");}
 #[test]fn complete_online_pass_matches_unpruned_shared_dag_recovery(){for seed in 0..64{let mut m:Vec<_>=(1..8).map(|s|cell((s+seed)%s,s,1.+(s%3)as f32/10.)).collect();for target in 1..8{let mut v=cell((target*5+seed)%target,target,0.2+((seed+target)%8)as f32/10.);if target<7&&seed%2==0{let mut o=v.outputs[0].clone();o.name="Z".into();o.targets=vec![target+1];v.outputs.push(o);}m.push(v);}let free:Vec<_>=(0..8).map(|s|s==0).collect();let full:Vec<_>=(0..8).map(|s|(s>0).then(||Binding{instance:s-1,pin:0})).collect();let roots=[5,6,7];let mut seed_cover=full.clone();prune(&m,&mut seed_cover,&free,&roots).unwrap();let value=metrics(&m,&seed_cover,&free,&roots);let p=MapParams{objective:MapObjective::DelayArea,fused_cover_scoring:true,online_area:true,shared_area_bound:true,refine_rounds:4,refine_trials:2000,..Default::default()};let(a,av)=slack_area_pass(&m,&free,&roots,&full,seed_cover.clone(),value,&p,&mut MapStats::default(),Instant::now()+std::time::Duration::from_secs(1),true);let q=MapParams{online_shared_bound:true,..p};let(b,bv)=slack_area_pass(&m,&free,&roots,&full,seed_cover,value,&q,&mut MapStats::default(),Instant::now()+std::time::Duration::from_secs(1),true);assert_eq!(av,bv,"seed{seed}");assert!(bv.1<=value.1+1e-6);assert_eq!(a.iter().map(|v|v.map(|b|(b.instance,b.pin))).collect::<Vec<_>>(),b.iter().map(|v|v.map(|b|(b.instance,b.pin))).collect::<Vec<_>>(),"seed{seed}");}}
}

#[cfg(test)] mod delta_area_tests {
 use super::*;
 fn buf(input:usize,target:usize,area:f32)->Instance{Instance{cell:format!("B{input}_{target}_{area}"),area,inputs:vec![input],input_pins:vec!["A".into()],outputs:vec![OutputPin{sources:Vec::new(),name:"Y".into(),delays:vec![1.],function:2,targets:vec![target]}]}}
 #[test]fn protected_new_input_avoids_unnecessary_complete_score(){
  let m=vec![buf(0,1,8.),buf(1,2,1.),buf(1,2,2.)];let f=[true,false,false];let w=vec![None,Some(Binding{instance:0,pin:0}),Some(Binding{instance:1,pin:0})];
  let p=MapParams{objective:MapObjective::DelayArea,refine_rounds:1,refine_trials:0,online_area:true,shared_area_bound:true,online_shared_bound:true,delta_area_filter:true,fused_cover_scoring:true,..Default::default()};let mut stats=MapStats::default();
  let(c,v)=slack_area_pass(&m,&f,&[2],&w,w.clone(),(9.,2.),&p,&mut stats,Instant::now()+std::time::Duration::from_secs(1),true);
  assert_eq!(v,(9.,2.));assert_eq!(c.iter().map(|b|b.map(|v|(v.instance,v.pin))).collect::<Vec<_>>(),w.iter().map(|b|b.map(|v|(v.instance,v.pin))).collect::<Vec<_>>());assert_eq!(stats.slack_local_scored,0,"new producer needs the old ancestor: cannot reclaim its eight area units");
 }
}

#[cfg(test)]mod delta_area_differential_tests{use super::*;
 fn cell(input:Vec<usize>,target:Vec<usize>,area:f32)->Instance{Instance{cell:"B".into(),area,input_pins:input.iter().map(|_|"A".into()).collect(),outputs:vec![OutputPin{sources:Vec::new(),name:"Y".into(),delays:input.iter().map(|_|1.).collect(),function:2,targets:target}],inputs:input}}
 #[test]fn selected_delta_matches_full_oracle_on_shared_joint_and_alias_dags(){
  for seed in 0..256 {
   let mut m:Vec<_>=(1..12).map(|s|cell(vec![(s*7+seed)%s],vec![s],1.+(s%5)as f32/10.)).collect();
   for t in 1..12 {let mut v=cell(vec![(t+seed)%t,(t*3+seed)%t],vec![t],0.2+(seed%11)as f32/10.);if t<11&&seed%2==0{let mut o=v.outputs[0].clone();o.targets=vec![t+1];v.outputs.push(o);}m.push(v);}
   let free:Vec<_>=(0..12).map(|s|s==0).collect();let full:Vec<_>=(0..12).map(|s|(s>0).then(||Binding{instance:s-1,pin:0})).collect();let roots=[9,10,11];let mut c=full.clone();prune(&m,&mut c,&free,&roots).unwrap();let area=metrics(&m,&c,&free,&roots).0;let(refs,users)=shared_area_counters(&m,&c,&roots);
   for i in 0..m.len(){let Some(estimate)=selected_input_area(&m,&c,&free,i,area,&refs,&users)else{continue};let mut trial=c.clone();for(pin,o)in m[i].outputs.iter().enumerate(){for &s in &o.targets{if c[s].is_some(){trial[s]=Some(Binding{instance:i,pin});}}}if let Ok(v)=validate_score_trial(&m,&mut trial,&full,&free,&roots,true){assert!((v.0-estimate).abs()<1e-10,"seed{seed} i{i} expected{} delta{estimate}",v.0);}}
  }
 }
 #[test]fn missing_input_defers_to_original_full_oracle(){let m=vec![cell(vec![0],vec![1],1.),cell(vec![1],vec![2],1.)];let c=vec![None,None,Some(Binding{instance:1,pin:0})];let(refs,users)=shared_area_counters(&m,&c,&[2]);assert!(selected_input_area(&m,&c,&[true,false,false],1,1.,&refs,&users).is_none());}
}

#[cfg(test)] mod area_timing_refresh_tests {use super::*;
 fn cell(input:usize,target:usize,area:f32,delay:f32)->Instance{Instance{cell:format!("B{input}_{target}_{area}_{delay}"),area,inputs:vec![input],input_pins:vec!["A".into()],outputs:vec![OutputPin{sources:Vec::new(),name:"Y".into(),delays:vec![delay],function:2,targets:vec![target]}]}}
 #[test]fn committed_faster_input_must_unlock_fixed_bound_area_replacement(){
  let m=vec![cell(0,1,8.,6.),cell(1,2,8.,1.),cell(0,3,1.,10.),cell(0,1,1.,1.),cell(1,2,1.,8.)];let f=[true,false,false,false];let w=vec![None,Some(Binding{instance:0,pin:0}),Some(Binding{instance:1,pin:0}),Some(Binding{instance:2,pin:0})];let p=MapParams{objective:MapObjective::DelayArea,online_area:true,refresh_area_timing:true,refine_rounds:1,refine_trials:2000,fused_cover_scoring:true,..Default::default()};let(c,v)=slack_area_pass(&m,&f,&[2,3],&w,w.clone(),(17.,10.),&p,&mut MapStats::default(),Instant::now()+std::time::Duration::from_secs(1),true);assert_eq!(v,(3.,10.),"committed input arrival must refresh before filtering the next candidate");assert_eq!(metrics(&m,&c,&f,&[2,3]),v);
 }
}

#[cfg(test)]mod protected_tail_tests{use super::*;
 #[test]fn same_delay_area_reference_retains_complete_owned_predecessor(){let old=vec![None,Some(Binding{instance:7,pin:0}),Some(Binding{instance:7,pin:1})];let reference=(vec![None,Some(Binding{instance:2,pin:0}),Some(Binding{instance:3,pin:0})],(3.,10.));let mut chosen=old.clone();let mut best=(8.,10.);let ps=MapParams{objective:MapObjective::DelayArea,..Default::default()};let(saved,value)=activate_saved_tail(&mut chosen,&mut best,Some(&reference),&ps,Instant::now()+std::time::Duration::from_secs(1)).unwrap();assert_eq!(value,(8.,10.));assert_eq!(saved.iter().flatten().map(|b|(b.instance,b.pin)).collect::<Vec<_>>(),vec![(7,0),(7,1)]);chosen[1]=None;assert!(saved[1].is_some());assert_eq!(best,(3.,10.));}
 #[test]fn slower_or_expired_tail_cannot_change_incumbent(){let mut c=vec![None];let mut b=(1.,2.);let ps=MapParams{objective:MapObjective::DelayArea,..Default::default()};let candidate=(vec![None],(0.,3.));assert!(activate_saved_tail(&mut c,&mut b,Some(&candidate),&ps,Instant::now()+std::time::Duration::from_secs(1)).is_none());let candidate=(vec![None],(0.,2.));assert!(activate_saved_tail(&mut c,&mut b,Some(&candidate),&ps,Instant::now()).is_none());assert_eq!(b,(1.,2.));}
}

#[cfg(test)] mod component_ablation_tests {
 use super::*;
 #[test] fn input_phase_and_pin_order_are_independent_factors() {
  let canonical=vec!["A".to_string(),"B".to_string()]; let swapped=vec!["B".to_string(),"A".to_string()];
  let mut p=MapParams::default();
  assert!(admit_input_assignment(1,&swapped,&canonical,&p));
  p.no_input_phase_matches=true;
  assert!(!admit_input_assignment(1,&canonical,&canonical,&p));
  assert!(admit_input_assignment(0,&swapped,&canonical,&p));
  p.no_input_phase_matches=false;p.canonical_pin_matching=true;
  assert!(admit_input_assignment(1,&canonical,&canonical,&p));
  assert!(!admit_input_assignment(0,&swapped,&canonical,&p));
 }
 #[test] fn ablated_actual_area_returns_existing_cover_without_trials() {
  let cover=vec![None];let mut stats=MapStats::default();let p=MapParams{no_actual_area_tail:true,..Default::default()};
  let (out,v)=slack_area_pass(&[],&[true],&[0],&cover,cover.clone(),(0.,0.),&p,&mut stats,Instant::now(),true);
  assert_eq!(out.len(),cover.len());assert!(out.iter().all(Option::is_none));assert_eq!(v,(0.,0.));assert_eq!(stats.slack_local_scored,0);
 }
}

#[cfg(test)] mod component_source_fallback_tests {
 use super::*;
 #[test] fn cut_ablation_keeps_identical_mandatory_source_decomposition() {
  let mut g=egg::EGraph::<AigLang,()>::default();let a=g.add(AigLang::Pi(0));let b=g.add(AigLang::Pi(1));let c=g.add(AigLang::Pi(2));let ab=g.add(AigLang::And([a,b]));let r=g.add(AigLang::Cell("@egm_tt3_80".into(),vec![a,b,c]));g.rebuild();
  let classes=vec![a,b,c,ab,r];let ids:HashMap<_,_>=classes.iter().enumerate().map(|(i,&id)|(id,i)).collect();let nodes:Vec<_>=classes.iter().map(|&id|g[id].nodes.clone()).collect();
  let source=vec![regions::SourceAnd{root:r,left:ab,right:c,left_neg:false,right_neg:false}];let p=MapParams{single_structure_cuts:true,max_inputs:3,cuts_per_class:16,..Default::default()};
  let cuts=enumerate_mapping_cuts(&nodes,&nodes,&ids,&g,&TechLib::default(),&p,&mut MapStats::default(),&source);
  assert!(cuts[ids[&r]].contains(&Cut{leaves:vec![ids[&c],ids[&ab]],function:8}),"single-witness optional cuts must not remove the common original-source physical fallback");
 }
}

#[cfg(test)] mod terminal_area_tests {use super::*;
fn cell(targets:Vec<usize>,area:f32,delay:f32)->Instance{Instance{cell:"T".into(),area,inputs:vec![0],input_pins:vec!["A".into()],outputs:vec![OutputPin{sources:Vec::new(),name:"Y".into(),delays:vec![delay],function:2,targets}]}}
fn ids(c:&[Option<Binding>])->Vec<Option<(usize,usize)>>{c.iter().map(|v|v.map(|b|(b.instance,b.pin))).collect()}
#[test]fn terminal_recovers_shared_area_after_old_cover_comparison(){let m=vec![cell(vec![1],8.,1.),cell(vec![2],8.,1.),cell(vec![1,2],1.,1.)];let f=[true,false,false];let c=vec![None,Some(Binding{instance:0,pin:0}),Some(Binding{instance:1,pin:0})];let p=MapParams{terminal_area:true,objective:MapObjective::DelayArea,refine_rounds:4,refine_trials:1,fused_cover_scoring:true,online_area:true,..Default::default()};let mut st=MapStats{slack_rounds_started:1,..Default::default()};let out=terminal_area_continuation(&m,&f,&[1,2],&c,c.clone(),&p,&mut st,Instant::now()+std::time::Duration::from_secs(1));assert_eq!(metrics(&m,&out,&f,&[1,2]),(1.,1.));assert!(st.terminal_area_accepted);assert!(st.slack_rounds_started<=4);assert!(st.slack_local_trials<=3);}
#[test]fn exhausted_or_expired_terminal_keeps_every_old_binding(){let m=vec![cell(vec![1],8.,1.),cell(vec![1],1.,1.)];let f=[true,false];let c=vec![None,Some(Binding{instance:0,pin:0})];let p=MapParams{terminal_area:true,objective:MapObjective::DelayArea,refine_rounds:1,refine_trials:1,..Default::default()};for spent in [0,1]{let mut st=MapStats{slack_rounds_started:spent,..Default::default()};let deadline=if spent==0{Instant::now()}else{Instant::now()+std::time::Duration::from_secs(1)};let out=terminal_area_continuation(&m,&f,&[1],&c,c.clone(),&p,&mut st,deadline);assert_eq!(ids(&out),ids(&c));assert!(!st.terminal_area_attempted);}}
#[test]fn lower_area_cannot_spend_fixed_delay(){let m=vec![cell(vec![1],8.,1.),cell(vec![1],1.,2.)];let f=[true,false];let c=vec![None,Some(Binding{instance:0,pin:0})];let p=MapParams{terminal_area:true,objective:MapObjective::DelayArea,refine_rounds:1,refine_trials:1,..Default::default()};let out=terminal_area_continuation(&m,&f,&[1],&c,c.clone(),&p,&mut MapStats::default(),Instant::now()+std::time::Duration::from_secs(1));assert_eq!(ids(&out),ids(&c));}}

#[cfg(test)] mod functional_quotient_tests {use super::*;
#[test]fn full_care_quotients_reconstruct_all_three_input_functions(){
 for function in 0..256 {for g in 1..15 {let f=Cut{leaves:vec![0,1,2],function};let p=Cut{leaves:vec![1,2],function:g};
  let q=exact_functional_quotient(&f,3,&p);let possible=(0..4).all(|a|(0..8).filter(|&x|((x&1)|((((g>>(x>>1))&1)as usize)<<1))==a).map(|x|(function>>x)&1).collect::<BTreeSet<_>>().len()==1);
  assert_eq!(q.is_some(),possible);if let Some(q)=q{for x in 0..8 {let a=q.leaves.iter().enumerate().fold(0,|v,(i,&p)|v|((if p==3{((g>>(x>>1))&1)as usize}else{(x>>p)&1})<<i));assert_eq!((q.function>>a)&1,(function>>x)&1);}}
 }}
}
#[test]fn four_input_quotient_reconstructs_random_total_functions(){let mut state=11u64;for _ in 0..4096{state=state.wrapping_mul(6364136223846793005).wrapping_add(1);let f=Cut{leaves:vec![0,1,2,3],function:state&65535};let p=Cut{leaves:vec![1,2,3],function:(state>>16)&255};if let Some(q)=exact_functional_quotient(&f,4,&p){for x in 0..16{let a=q.leaves.iter().enumerate().fold(0,|v,(i,&id)|v|((if id==4{((p.function>>(x>>1))&1)as usize}else{(x>>id)&1})<<i));assert_eq!((q.function>>a)&1,(f.function>>x)&1);}}}}
#[test]fn unsafe_or_constant_partner_is_rejected(){let f=Cut{leaves:vec![0,1,2],function:0x80};for p in [Cut{leaves:vec![1,2],function:0},Cut{leaves:vec![1,2],function:15},Cut{leaves:vec![1,4],function:8}]{assert!(exact_functional_quotient(&f,3,&p).is_none());}assert!(exact_functional_quotient(&f,0,&Cut{leaves:vec![1,2],function:8}).is_none());}
}

#[cfg(test)]mod functional_phase_screen_tests {use super::*;
 #[test]fn one_fast_phase_must_not_hide_other_phase_regression(){assert!(!phase_screen_accepts([1.,9.],[3.,4.]),"min over phases hides a slower required phase");assert!(phase_screen_accepts([2.,4.],[3.,4.]));assert!(!phase_screen_accepts([3.,4.],[3.,4.]));assert!(!phase_screen_accepts([f64::INFINITY,1.],[3.,4.]));}
}

#[cfg(test)] mod fixed_bound_exchange_tests {
 use super::*;
 #[test] fn explicit_fixed_bound_permits_slack_area_but_not_delay_objective_trade() {
  let make=|area,delay|Instance{cell:"BUF".into(),area,inputs:vec![0],input_pins:vec!["A".into()],outputs:vec![OutputPin{sources:vec![],name:"Y".into(),delays:vec![delay],function:2,targets:vec![1]}]};
  let m=vec![make(8.,1.),make(1.,2.)];let free=[true,false];let w=vec![None,Some(Binding{instance:0,pin:0})];
  let p=MapParams{objective:MapObjective::AreaDelay,max_delay:Some(3.),fixed_area_bound:true,online_area:true,refine_rounds:1,refine_trials:1,..Default::default()};
  let (_,v)=slack_area_pass(&m,&free,&[1],&w,w.clone(),(8.,1.),&p,&mut MapStats::default(),Instant::now()+std::time::Duration::from_secs(1),true);
  assert_eq!(v,(1.,2.));assert!(better_cover((8.,1.),v,&MapParams{objective:MapObjective::DelayArea,..p.clone()}));
  let legacy=MapParams{fixed_area_bound:false,..p};let (_,v)=slack_area_pass(&m,&free,&[1],&w,w.clone(),(8.,1.),&legacy,&mut MapStats::default(),Instant::now()+std::time::Duration::from_secs(1),true);assert_eq!(v,(8.,1.));
 }
}

#[cfg(test)] mod feasible_visited_area_tests {use super::*;
#[test]fn fixed_delay_online_commit_uses_stage_limit_after_a_faster_point(){
 let p=MapParams{objective:MapObjective::AreaDelay,fixed_area_bound:true,max_delay:Some(10.),..Default::default()};let mut chosen=vec![None];let mut best=(9.,5.);let mut batch=Some((vec![None],(5.,8.),5.,0));assert!(commit_online_area(&mut batch,&mut chosen,&mut best,&p,&mut MapStats::default()));assert_eq!(best,(5.,8.));
}
#[test]fn area_trajectory_retains_fastest_eligible_already_scored_whole_cover(){
 let cell=|a,d|Instance{cell:"BUF".into(),area:a,inputs:vec![0],input_pins:vec!["A".into()],outputs:vec![OutputPin{sources:vec![],name:"Y".into(),delays:vec![d],function:2,targets:vec![1]}]};
 let m=vec![cell(10.,10.),cell(9.,5.),cell(5.,8.)];let f=[true,false];let c=vec![None,Some(Binding{instance:0,pin:0})];let p=MapParams{objective:MapObjective::AreaDelay,max_delay:Some(10.),fixed_area_bound:true,protected_area_limit:Some(10.),online_area:true,refresh_area_timing:true,refine_rounds:1,refine_trials:0,..Default::default()};
 let mut stats=MapStats::default();let(_,value)=slack_area_pass(&m,&f,&[1],&c,c.clone(),(10.,10.),&p,&mut stats,Instant::now()+std::time::Duration::from_secs(1),true);
 assert_eq!(value,(9.,5.));assert_eq!(stats.slack_local_trials,0);assert_eq!(stats.online_area_updates,2);
 let mut bad=m.clone();bad[1].area=12.;let(_,v)=slack_area_pass(&bad,&f,&[1],&c,c.clone(),(10.,10.),&p,&mut MapStats::default(),Instant::now()+std::time::Duration::from_secs(1),true);assert_eq!(v,(5.,8.),"over-area fast point cannot replace a legal intermediate");
}}

#[cfg(test)] mod generation_frontier_tests {use super::*;
 fn cell(area:f32,delay:f32)->Instance{Instance{cell:format!("BUF_{area}_{delay}"),area,inputs:vec![0],input_pins:vec!["A".into()],outputs:vec![OutputPin{sources:vec![],name:"Y".into(),delays:vec![delay],function:2,targets:vec![2]}]}}
 #[test]fn necessary_full_frontier_is_carried_and_raw_indices_are_remapped(){
  let mut pool=vec![cell(1.,10.),cell(2.,9.),cell(3.,1.)];let f=[true,false,false,false];
  let result=cap_instances_with_frontier(&mut pool,&f,&[2],2,false,true,true,false,true).unwrap();assert_eq!(result.pruned,1);
  let saved=result.frontier.expect("actual retention decision computed a better frontier but discarded it");let (c,w,origin)=saved.resolve(&pool,&f,&[2]).unwrap();assert_eq!(origin,"full-pool-retention");assert_eq!(w.iter().map(|v|v.map(|b|(b.instance,b.pin))).collect::<Vec<_>>(),finite_choices_queued(&pool,&f).iter().map(|v|v.map(|b|(b.instance,b.pin))).collect::<Vec<_>>());assert_eq!(c[2].unwrap().instance,1);assert_eq!(metrics(&pool,&c,&f,&[2]).1,1.);
 }
 #[test]fn necessary_legacy_guard_frontier_keeps_retained_index_space(){
  let mut pool=vec![cell(1.,10.),cell(2.,1.),cell(3.,3.)];let f=[true,false,false,false];let result=cap_instances_with_frontier(&mut pool,&f,&[2],2,false,true,true,true,true).unwrap();
  let (c,w,origin)=result.frontier.unwrap().resolve(&pool,&f,&[2]).unwrap();assert_eq!(origin,"legacy-pool-guard");assert_eq!(w.iter().map(|v|v.map(|b|(b.instance,b.pin))).collect::<Vec<_>>(),finite_choices_queued(&pool,&f).iter().map(|v|v.map(|b|(b.instance,b.pin))).collect::<Vec<_>>());assert_eq!(metrics(&pool,&c,&f,&[2]),(2.,1.));
 }
 #[test]fn carried_area_witness_equals_recomputation_after_ordered_compaction(){
  for seed in 0..64 {let n=8;let mut pool=Vec::new();for target in 1..=n {for variant in 0..4 {
   let mut m=cell(1.+((seed*11+target*3+variant)%13)as f32/10.,1.+((seed*7+target+variant*3)%11)as f32/10.);m.inputs=vec![(seed*3+target+variant)%target];m.outputs[0].targets=vec![target];pool.push(m);
  }}let free:Vec<_>=(0..=n).map(|s|s==0).collect();let result=cap_instances_with_frontier(&mut pool,&free,&[n],2*n+1,false,true,true,true,true).unwrap();let (_,w,_)=result.frontier.unwrap().resolve(&pool,&free,&[n]).unwrap();let fresh=finite_choices_queued(&pool,&free);assert_eq!(w.iter().map(|v|v.map(|b|(b.instance,b.pin))).collect::<Vec<_>>(),fresh.iter().map(|v|v.map(|b|(b.instance,b.pin))).collect::<Vec<_>>(),"seed{seed}");}
 }
 #[test]fn reporting_only_frontier_is_never_an_algorithmic_seed(){
  if std::env::var("_EGM_AUDIT_FRONTIER_CHILD").as_deref()!=Ok("1") {
   let out=std::process::Command::new(std::env::current_exe().unwrap()).args(["--exact","egraph_map::generation_frontier_tests::reporting_only_frontier_is_never_an_algorithmic_seed","--nocapture"]).env("_EGM_AUDIT_FRONTIER_CHILD","1").env("EGM_POOL_FRONTIER_AUDIT","1").env_remove("EGM_POOL_DIAG_PATH").output().unwrap();assert!(out.status.success(),"{}",String::from_utf8_lossy(&out.stdout));return;
  }
  let mut pool=vec![cell(1.,2.),cell(2.,1.)];let result=cap_instances_with_frontier(&mut pool,&[true,false,false,false],&[2],3,false,true,true,true,true).unwrap();assert!(result.frontier.is_none(),"uncapped reporting work cannot become a free mapping seed");
 }
}

#[cfg(test)]mod exchange_setup_budget_tests{use super::*;
 #[test]fn expired_setup_returns_no_partial_index(){let old=vec![BTreeSet::from([Cut{leaves:vec![1,2],function:8}])];assert!(functional_support_index_until(&old,Some(Instant::now())).is_none(),"expired exchange setup must stop before building the global support index");}
 #[test]fn unexpired_setup_keeps_deterministic_four_partner_cap(){let mut old=vec![BTreeSet::new();10];for g in 5..10{old[g].insert(Cut{leaves:vec![0,1],function:8});}let a=functional_support_index_until(&old,Some(Instant::now()+std::time::Duration::from_secs(1))).unwrap();assert_eq!(a[&vec![0,1]].iter().map(|x|x.0).collect::<Vec<_>>(),vec![5,6,7,8]);assert_eq!(a,functional_support_index(&old));}
}

#[cfg(test)]mod pool_report_summary_tests{use super::*;
 #[test]fn compact_report_preserves_counts_without_proxy_rows(){
 let m=Instance{cell:"BUF".into(),area:1.,inputs:vec![0],input_pins:vec!["A".into()],outputs:vec![OutputPin{sources:Vec::new(),name:"Y".into(),delays:vec![1.],function:2,targets:vec![1]}]};let pool=vec![m];let free=[true,false];let witness=finite_choices_queued(&pool,&free);let keep=BTreeSet::from([0]);let dir=std::env::temp_dir().join(format!("egm-pool-report-{}",std::process::id()));std::fs::create_dir_all(&dir).unwrap();let full=dir.join("full.json");let short=dir.join("short.json");
 write_pool_diagnostic_detail(&pool,&free,&[1],&witness,&keep,1,100,full.to_str().unwrap(),true).unwrap();write_pool_diagnostic_detail(&pool,&free,&[1],&witness,&keep,1,100,short.to_str().unwrap(),false).unwrap();
 let a:serde_json::Value=serde_json::from_slice(&std::fs::read(full).unwrap()).unwrap();let b:serde_json::Value=serde_json::from_slice(&std::fs::read(short).unwrap()).unwrap();for key in ["instances_before","instances_after","witness_reserved_instances","budget","roots"]{assert_eq!(a[key],b[key]);}assert!(!a["signals"].as_array().unwrap().is_empty());assert_eq!(b["signals"],serde_json::json!([]));assert_eq!(b["detailed"],false);std::fs::remove_dir_all(dir).unwrap();
 }
}
