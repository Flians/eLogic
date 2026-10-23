"""Write the current measured scope without replacing failures by old winners."""
import argparse,csv,json,math,re
from pathlib import Path
from run_time_only600 import ROOT,EPFL,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
    study=ROOT/'results_v2/time_only_600_20261022';original=(study/'paper_before600.tex').read_text()
    summary=json.loads((a.report/'validation.json').read_text());c=json.loads((a.report/'candidate.json').read_text())['results'];b=json.loads((a.report/'baselines.json').read_text())['results']
    head=original.split('\\section{Experimental Evaluation}')[0]
    abstract=r'''Technology mapping must choose a realizable circuit from equivalent Boolean structures while accounting for shared logic and timing.
We present \tool, a prototype for single-output standard-cell mapping over an e-graph.
Structural and functional cuts expose alternatives; physical matches retain input phase, pin assignment and conditional care information.
Whole-cover recovery charges shared instances once and reduces actual area under a validated timing limit.
Conditional matches remain outside the exact equivalence relation.
Evaluation uses all 20 EPFL arithmetic/control circuits, one shared \texttt{resyn2} input per design, and 47 supplied single-output ASAP7 cells.
A revised protocol limits total mapping time to 600 seconds and removes the earlier configurable node, pool, cut-count and iteration ceilings.
Failures and basic source covers are retained separately from completed optimized covers.
The results expose candidate-generation costs and do not establish full-suite dominance over the original baseline algorithms.
'''
    if summary['candidate_completed']==20:
        abstract=abstract.replace('Failures and basic source covers are retained separately from completed optimized covers.',
            f"The full sweep returns {summary['candidate_optimized']} complete optimized covers; the other {20-summary['candidate_optimized']} cases retain separately verified source implementations after mapping timeout.")
    head=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda m:r'\begin{abstract}'+'\n'+abstract+r'\end{abstract}',head,flags=re.S)
    start=head.index('\\subsection{Bounded cuts over exact alternatives}')
    end=head.index('\\subsection{Physical cell matching}',start)
    head=head[:start]+r'''\subsection{Cuts over exact alternatives}
The importer hash-conses Boolean operators and exact XOR/three-input truth-table contractions into one e-graph.
All original source helpers are retained in the time-only profile; compressed operators are not assumed to expose every AIG rewrite pattern.
Canonical e-classes support cut construction over alternative e-nodes.
Child boundaries are merged and their functions composed. Structural witnesses preserve a finite original implementation; cyclic e-graph alternatives do not permit cyclic physical covers.
Region ownership preserves complete physical boundaries and global target identifiers. It does not create separate per-region budgets.

The current policy removes configurable global count ceilings, rather than proving exhaustive enumeration. The algorithm still restricts the families of structures considered.
Small cuts use truth tables on actual support of at most six signals.
Seven-to-nine-input cells use directed structural matching, without constructing a table at the maximum library width for every cut.
Functional quotients use an existing class for $G(T)$ to form
$F(S)=Q(S\setminus T,G(T))$ only when all assignments uniquely determine every quotient entry.
The current proposal generator considers small quotient families, with fixed partner/batch rules. Safe exchange likewise uses a finite transaction batch; these algorithmic restrictions remain distinct from the removed global node and retained-pool ceilings.

Exact Not edges identify physical signal aliases; raw logical cut banks and physical pin interfaces remain explicit. Conditional matches establish no aliases.
Generation deduplicates physical instances before recovery. Delay-frontier propagation uses dependency notifications rather than repeated unchanged sweeps.
No optimality claim extends beyond the finite candidate problem actually constructed.

''' +head[end:]
    head=head.replace('A borrowed preparation view copies only changed\nrecords, with a 2048-record limit. The commit moves unchanged records.',
        'A borrowed preparation view copies only changed records within the current safe-exchange transaction family. The commit moves unchanged records.')
    protocol=r'''\section{Experimental Evaluation}
\subsection{Protocol and scope}
The suite contains all 20 EPFL arithmetic/control circuits \cite{epfl}; MtM is not included.
Each method reuses the same named AIG after one fixed ABC \texttt{resyn2} sequence and a preparation CEC. No method repeats this front end.
The common inventory contains all 47 single-output records in the supplied ASAP7 genlib \cite{asap7}: 45 named physical cell types and two zero-area constants. Seven cells have seven to nine inputs. This is not the full foundry PDK.
Earlier three-/four-input inventory results are archived as historical studies and are not substituted into the current table.

The current per-design/method allowance is 600 seconds of accumulated mapping wall time. Our clock includes source construction, import, saturation, cuts, matching, recovery and export.
There is no separate native-process, stage, address-space, node, search-ratio, retained-pool, cut-count or iteration ceiling in this execution policy.
The existing candidate families and safe-exchange batch rules remain algorithmic restrictions, as described above. CBC is disabled.
External preparation, conversion, independent CEC and metric reporting are excluded. Algorithmic intermediate metrics and mapping feedback remain inside the corresponding mapping call.
Stage work does not obtain a renewed outer deadline. Process termination and output draining can add small wall-time overhead after the deadline; such cases remain failures.

Baseline algorithms, source commits, models and author search parameters are unchanged. Comparisons include ABC map/map-a, independent emap in both modes, EqMap followed by same-library ABC amap with staging controls, frozen-model E-Syn and final native E-morphic SA.
Only necessary port adaptation, library-range adaptation and timing instrumentation are applied.
Original completed groups may be reused when their outputs pass CEC and observed mapping time, or a documented conservative upper bound, is below the old 300-second guard. They retain their original policy and runtime labels; they are not reported as fresh 600-second runs.
Incomplete groups are retested, with every original failure retained.

New EqMap runs meter the actual ABC process inside the initial Yosys LUT conversion. That shared mapping prefix is charged once to each relevant downstream method; conversion is excluded.
Historical EqMap runtimes retain the enclosing-Yosys upper-bound label.
E-morphic mapping feedback is charged as the union of concurrent active ABC intervals; idle and other SA work are excluded. Its unseeded native winner is reported without reselection using public metrics.
Native E-Syn/EqMap two-second, 20,000-node, 16-iteration author settings remain unchanged; thus the common external ceiling does not imply identical internal search budgets or end-to-end compute.
Times are observed wall times on the same host, without CPU pinning; author parallel settings are retained. They do not establish an isolated or machine-independent runtime gain.

Physical area counts every emitted instance once. Delay is the maximum output arrival using the maximum rise/fall unloaded block delay for functionally dependent pins; load, slew and interconnect are omitted.
Every numeric current output has independent named-port CEC against the original AIG, using explicit functional cell models.
Verification-only normalization never supplies candidates or scores to mapping.
Proof reuse requires identical physical netlist bytes, semantic connectivity, original input and library hashes; it is marked separately from a fresh CEC run. Interrupted redundant checks remain in the raw record.
Source fallback has its own metrics and CEC record, but does not count as completed optimized coverage.
Manifests retain commands, parameters, source/input/library/binary hashes, netlists and failed logs.

'''
    columns=[('ABC map','abc_map_delay_resyn2'),('emap','mockturtle_emap_delay_resyn2'),('EqMap','eqmap_lut_amap__delay-area'),('E-Syn','esyn_frozen_model_amap_bounded_dag__delay-area'),('E-morphic','emorphic_final_sa'),(r'\tool','egraph_time600')]
    allrows={(r['bench'],r['method']):r for r in b+c}
    table=r'''\begin{table*}[!t]\centering\fontsize{7.0}{8.5}\selectfont
\setlength{\tabcolsep}{1.3pt}\renewcommand{\arraystretch}{1.0}
\caption{Full 47-cell single-output inventory, one uniform delay-area eGraphMap configuration. T is mapping seconds; $\leq$ marks an inherited conservative upper bound. F retains failed or source-only optimization status; P is pending. A failed optimizer's separately verified source metrics remain in the data package. Reused completed baseline groups retain their original timing policy.}
\label{tab:epfl600}\begin{tabular}{l*{18}{r}}\toprule
Design & '''+' & '.join(r'\multicolumn{3}{c}{'+label+'}'for label,_ in columns)+r''' \\
 & '''+' & '.join(['A & D & T']*6)+r''' \\ \midrule
'''
    for bench in EPFL:
        values=[bench.replace('_',r'\_')]
        for _,method in columns:
            r=allrows[bench,method];m=r.get('measured')or{};time=r.get('mapping_wall_secs')
            t='--'if time is None else f'{time:.2f}'
            if r.get('T_kind')=='upper_bound_enclosing_LUT_conversion':t=r'$\leq$'+f'{math.ceil(time*100)/100:.2f}'
            if r['status']=='ok':values += [f"{m['area']:.2f}",f"{m['delay']:.2f}",t]
            else:values += [('P'if r['status']=='pending600'else'F')]*2+[t]
        table+=' & '.join(values)+r' \\'+'\n'
    table+=r'\bottomrule\end{tabular}\end{table*}'+'\n'
    results='\\subsection{Measured coverage and limitations}\n'
    if summary['candidate_completed']==20:
        results+=f"All 20 eGraphMap cases were attempted: {summary['candidate_verified_outputs']} emitted outputs pass independent verification and {summary['candidate_optimized']} return complete optimized covers. "
    else:results+=f"The current checkpoint contains {summary['candidate_completed']} of 20 eGraphMap attempts, {summary['candidate_verified_outputs']} independently verified emitted outputs and {summary['candidate_optimized']} completed optimized covers. "
    if summary['baseline_completed_groups']==120:
        results+=f"All 120 baseline groups were attempted; {summary['baseline_qualified_observations']} of 260 method observations are qualified, with failures or unmeasured slots retained separately. "
    else:results+=f"The baseline checkpoint contains {summary['baseline_completed_groups']} of 120 groups and {summary['baseline_qualified_observations']} qualified method observations out of 260 slots. Pending slots remain explicit. "
    results+='Attempted coverage is not equivalent to valid PPA coverage. Table~\\ref{tab:epfl600} uses the first observation for each fixed configuration, never a best repetition or a per-design profile choice. Area-oriented baselines and all staging controls remain in the machine-readable full-scope gates.\n\n'
    for bench in ['hyp','log2']:
        r=allrows[bench,'egraph_time600']
        if r['status']=='pending600':results+=bench.capitalize()+' has no completed current-policy observation yet. '
        elif r.get('optimizer_succeeded'):results+=bench.capitalize()+' returns a complete optimized cover; its final emitted output passes independent CEC. '
        else:results+=bench.capitalize()+' does not return a complete optimized cover under the current policy; any verified source fallback remains separate. '
    results+='Full-scope delay-area, area-delay and Pareto gates remain false. A strict delay improvement is a lexicographic win even if area increases; it is not simultaneous Pareto dominance.\n\n'
    results+=r'''\subsection{Baseline interpretation}
EqMap retains original six-input LUT remapping before standard-cell mapping; these results measure a cross-target composition, not native LUT utilization.
E-Syn retains original DAG features, frozen models and up to 30 ranked candidates. Its input conversion, candidate indexing and full-library calibration limit interpretation; no optimized feature/index probe enters this table.
E-morphic uses five rewriting iterations, ten initial samples and six rounds of ten neighbors at probability 0.8. Its feedback Liberty names the same physical cells, but loaded feedback delay differs from public unloaded delay.
Native exits, conversion failures, mapping timeouts and unproved CEC remain separate in the data package.

\subsection{Historical independent removals}
The earlier fixed-20,000-node study is retained in Table~\ref{tab:ablation}; it is not a same-binary ablation of the current 600-second policy or 47-cell inventory.
Functional quotients, witness-only cuts, input-phase matching, pin assignment, SDC, post-phase area recovery and seed protection were removed independently.
Area and delay counts use jointly legal pairs; failed coverage remains separate. The targeted repetitions use sqrt, sin and priority across both historical inventories, with alternating case order and rotated settings.
They neither establish full-suite stability nor isolate the later compact representation and timing-policy changes.

'''
    ablation=original[original.index('\\begin{table*}[!t]\\centering\\footnotesize',original.index('% END EPFL COMPLETE TABLES')):]
    ablation=ablation[:ablation.index('\\end{table*}')+len('\\end{table*}')]
    ablation=ablation.replace('these are not ablations of the proportional profile','these are not ablations of the current 600-second profile')
    conclusion=r'''\section{Discussion and Conclusion}
The current experiment separates representability from optimization: an AND/NAND/INV source implementation can pass CEC while search fails to produce an optimized physical cover.
Removing import rejection therefore cannot by itself establish a mapping-quality improvement.
In the current large-design traces, saturation can finish quickly while later cut expansion consumes the available time. Additional retained alternatives increase both opportunity and the cost of reaching extraction.
These observations motivate duplicate and dominance filtering and earlier commitment of valid optimized incumbents, rather than a claim that a larger allowance necessarily improves delay.

Shared-area recovery remains coupled to candidate coverage, phase and pin choices. Accurate whole-cover metrics prevent double charging of reused cells, but do not prove globally minimum area.
Conservative SDC matches do not solve ODC or sequential reachability. Directed wide matching does not exhaust every wide function or compact/AIG boundary.
Current measurements do not establish full-suite dominance or repeatable gains for every design. Complete same-policy component removals and repetitions remain necessary before attributing improvements to individual mechanisms or promoting the research configuration to the main mapper.
'''
    refs=original[original.index('\\clearpage\n\\twocolumn\n\\begin{thebibliography}'):]
    target=ROOT/'emapper_paper/DAC-eGraphMap-draft/main.tex'
    target.write_text(head+protocol+table+results+ablation+'\n'+conclusion+'\n'+refs)
    (study/'paper600_provenance.json').write_text(json.dumps(dict(report=str(a.report.resolve()),report_sha256=sha(a.report/'validation.json'),source_before=str(study/'paper_before600.tex'),source_before_sha256=sha(study/'paper_before600.tex'),main_tex_sha256=sha(target),generator_sha256=sha(Path(__file__))),indent=2)+'\n')
if __name__=='__main__':main()
