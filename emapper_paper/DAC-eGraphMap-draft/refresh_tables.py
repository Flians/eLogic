#!/usr/bin/env python3
"""Refresh standalone paper tables only from CEC-passing saved reports."""
import csv
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BENCHES = ["c17", "c432", "c499", "c880", "c1355", "c1908", "c2670"]
SOURCES = [ROOT / "results_v2" / p / "report.json" for p in
           ["fixed_all_area_20261012", "fixed_4in_area_20261012"]]
COMPLETE = ROOT / "results_v2/fixed_complete_20261012"
EQMAP = COMPLETE / "eqmap_3in_area.json"
FOLLOWUPS = [ROOT / "results_v2" / folder / "report.json" for folder in [
    "egraph_cumulative_controlled_20261012", "egraph_structural_refs_20261012",
    "egraph_c17_timing_feasibility_20261012"]]
OBJECTIVE_REPORTS = [ROOT / "results_v2" / p / "report.json" for p in [
    "fixed_all_area_20261012", "fixed_all_delay_20261012"]]
PIPELINE_REPORTS = [ROOT / "results_v2" / p / "report.json" for p in [
    "egraph_resyn2_area_20261012", "egraph_resyn2_delay_20261012",
    "esyn_c17_probe_20261012", "esyn_weak_bounded_20261012",
    "esyn_c17_bounded_fixed_20261012", "esyn_weak_dag_bounded_20261012",
    "emorphic_random_area_20261012", "emorphic_random_delay_20261012",
    "emorphic_final_sa_20261012", "emorphic_final_sa_resyn2_20261012",
    "egraph_fixed_resyn2_c17_area_20261012", "egraph_fixed_resyn2_c17_delay_20261012",
    "fixed_all_delay_20261012", "fixed_all_eqmap_20261012", "fixed_all_eqmap_delay_20261012",
    "fixed_all_esyn_20261012", "fixed_all_emorphic_20261012",
    "fixed_4in_delay_20261012", "fixed_4in_eqmap_20261012", "fixed_4in_eqmap_delay_20261012",
    "fixed_4in_esyn_20261012", "fixed_4in_emorphic_20261012",
    "fixed_eqmap_delay_c2670_repair_20261012", "fixed_eqmap_delay_c2670_repair2_20261012",
    "fixed_esyn_c2670_repair_20261012", "worklist_area_20261012",
    "worklist_delay_20261012", "worklist_area_expanded_20261012",
    "worklist_delay_expanded_20261012", "flow_compare_area_20261012",
    "flow_compare_delay_20261012", "flow_compare_4in_area_20261012",
    "flow_compare_4in_delay_20261012", "critical_area_20261012",
    "critical_delay_20261012", "depth_delay_probe_20261012",
    "depth_delay_rest_20261012", "depth_area_20261012",
    "cut_mix_delay_probe_20261012", "cut_mix_delay_rest_20261012",
    "cut_mix_area_20261012", "library_timing_delay_probe2_20261012",
    "library_timing_delay_rest_20261012", "library_timing_area_20261012",
    "library_timing_4in_delay_20261012", "library_timing_4in_area_20261012",
    "slack_flow_delay_probe_20261012", "slack_flow_delay_rest_20261012",
    "slack_flow_area_20261012", "slack_flow_4in_delay_20261012",
    "slack_flow_4in_area_20261012", "timing_cap_3in_delay-area_20261012",
    "timing_cap_3in_area-delay_20261012", "timing_cap_4in_delay-area_20261012",
    "timing_cap_4in_area-delay_20261012"]]


def load(path):
    data = json.loads(path.read_text())
    for row in data["results"]:
        if row["status"] == "ok" and row["measured"]["cec"] != 1:
            raise ValueError(f"unverified record in {path}")
    return data


def pair(row):
    if row is None or row["status"] != "ok":
        return "failed"
    return f"{row['measured']['area']:.2f} / {row['measured']['delay']:.2f}"


def metrics(row):
    if row is None or row["status"] != "ok":
        return ["failed", "failed"]
    return [f"{row['measured'][key]:.2f}" for key in ["area", "delay"]]


def texrow(cells):
    return " & ".join(cells) + r" \\" + "\n"


def standard(data, delay_data, k):
    rows = {(r["bench"], r["method"].removesuffix("_resyn2")): r for r in data["results"]}
    delayed = {r["bench"]: r for r in delay_data["results"] if r["method"] == "egraph_resyn2"}
    methods = ["abc_map_area", "mockturtle_emap_area", "egraph", "abc_map_delay", "mockturtle_emap_delay"]
    label = "three" if k == 3 else "four"
    tex = ("\\begin{table*}[t]\n\\centering\\small\n\\setlength{\\tabcolsep}{3pt}\n"
           f"\\caption{{Fixed resyn2, {k}-input library results. Area and unloaded block delay are "
           "reported separately. ABC area-first uses map -a; ABC delay-first uses map. All netlists pass CEC.}\n"
           f"\\label{{tab:{label}}}\n"
           "\\begin{tabular}{lrrrrrrrrrrrr}\\toprule\n")
    tex += texrow(["", r"\multicolumn{6}{c}{Area-first $(A,D)$}", r"\multicolumn{6}{c}{Delay-first $(D,A)$}"])
    tex += "\\cmidrule(lr){2-7}\\cmidrule(lr){8-13}\n"
    tex += texrow(["Design"] + [r"\multicolumn{2}{c}{" + name + "}" for name in ["ABC map", "emap", r"\tool{}", "ABC map", "emap", r"\tool{}"]])
    tex += "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}\\cmidrule(lr){12-13}\n"
    tex += texrow([""] + ["Area", "Delay"] * 6) + "\\midrule\n"
    for b in BENCHES:
        if (b, "egraph") in rows:
            records = [rows.get((b, m)) for m in methods] + [delayed[b]]
            tex += texrow([b] + [v for row in records for v in metrics(row)])
    return tex + "\\bottomrule\\end{tabular}\n\\end{table*}\n"


def eqmap(data):
    rows = {(r["bench"], r["method"]): r for r in data["results"]}
    tex = ("\\begin{table*}[t]\n\\centering\\small\n"
           "\\caption{EqMap cross-target composition: 6-input LUTs, 3-input standard-cell library. "
           "Area and delay use the common physical evaluator. LUT count and depth are native stage metrics.}\n"
           "\\label{tab:eqmap}\n\\begin{tabular}{lrrrrrrrr}\\toprule\n")
    tex += texrow(["Design", r"\multicolumn{4}{c}{LUT stage + amap}", r"\multicolumn{4}{c}{EqMap LUT + amap}"])
    tex += "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
    tex += texrow([""] + ["Area", "Delay", "LUTs", "Depth"] * 2) + "\\midrule\n"
    for b in BENCHES:
        if (b, "abc_amap") in rows:
            cells = [b]
            for method in ["lut_stage_amap", "eqmap_lut_amap"]:
                row = rows.get((b, method))
                cells += metrics(row)
                cells += ([str(row['native_lut'][key]) for key in ["lut_count", "lut_depth"]]
                          if row and row["status"] == "ok" else ["failed", "failed"])
            tex += texrow(cells)
    return tex + "\\bottomrule\\end{tabular}\n\\end{table*}\n"


def main():
    standard_data = [load(p) for p in SOURCES]
    eqmap_data = load(EQMAP)
    if eqmap_data["library_sha256"] != standard_data[0]["library_sha256"]:
        raise ValueError("EqMap library differs from the standard comparison")
    for row in eqmap_data["results"]:
        references = [r for r in standard_data[0]["results"] if r["bench"] == row["bench"]]
        if any(r["input_sha256"] != row["input_sha256"] for r in references):
            raise ValueError("benchmark input mismatch")
    source = HERE / "main.tex"
    text = source.read_text()
    standard_delay = [load(ROOT / "results_v2" / p / "report.json") for p in
                      ["fixed_all_delay_20261012", "fixed_4in_delay_20261012"]]
    blocks = {"STANDARD TABLES": "\n".join(standard(d, delayed, k) for d, delayed, k in zip(standard_data, standard_delay, [3, 4])),
              "EQMAP TABLE": eqmap(eqmap_data)}
    objectives = [load(p) for p in OBJECTIVE_REPORTS]
    if any(d["library_sha256"] != standard_data[0]["library_sha256"] for d in objectives):
        raise ValueError("objective profile library mismatch")
    rows = [{r["bench"]: r for r in d["results"] if r["method"] == "egraph_resyn2"} for d in objectives]
    library_profile = load(ROOT / "results_v2/slack_flow_complete_20261012/delay_3in.json")
    if library_profile["library_sha256"] != standard_data[0]["library_sha256"]:
        raise ValueError("library-aware profile uses a different library")
    library_rows = {r["bench"]: r for r in library_profile["results"]
                    if r["method"] == "egraph_library_slack_resyn2"}
    table = ("\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{3pt}\n"
             "\\caption{Default priorities and optional library-aware slack recovery, 3-input library. "
             "Area and delay are separate; CBC is disabled and all covers pass CEC.}\n"
             "\\label{tab:objectives}\\begin{tabular}{lrrrrrr}\\toprule\n")
    table += texrow(["Design", r"\multicolumn{2}{c}{Default $(A,D)$}", r"\multicolumn{2}{c}{Default $(D,A)$}", r"\multicolumn{2}{c}{Slack $(D,A)$}"])
    table += "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
    table += texrow([""] + ["Area", "Delay"] * 3) + "\\midrule\n"
    for b in BENCHES:
        if len({rows[0][b]["input_sha256"], rows[1][b]["input_sha256"], library_rows[b]["input_sha256"]}) != 1:
            raise ValueError("objective profile inputs differ")
        table += texrow([b] + [v for r in [rows[0][b], rows[1][b], library_rows[b]] for v in metrics(r)])
    blocks["OBJECTIVE TABLE"] = table + "\\bottomrule\\end{tabular}\\end{table}\n"
    baselines = load(COMPLETE / "baselines_3in.json")
    depth = load(COMPLETE / "eqmap_3in_delay.json")
    extra = ("\\begin{table*}[t]\\centering\\small\n"
        "\\caption{Structural-synthesis adapters after the identical resyn2 subject, 3-input library. "
        "Entries are common area / delay, not native LUT or Liberty timing scores.}\n"
        "\\label{tab:adapters}\\begin{tabular}{lrrrr}\\toprule\n"
        "Design & EqMap depth + amap & E-Syn $(A,D)$ & E-Syn $(D,A)$ & E-morphic final SA \\\\\n\\midrule\n")
    for b in BENCHES:
        eq = next(r for r in depth["results"] if r["bench"] == b and r["method"] == "eqmap_lut_amap")
        es = [next(r for r in baselines["results"] if r["bench"] == b and
                   r["method"].startswith("esyn_") and r.get("selection_objective") == obj)
              for obj in ["area-delay", "delay-area"]]
        sa = next(r for r in baselines["results"] if r["bench"] == b and r["method"] == "emorphic_final_sa")
        extra += b + " & " + " & ".join(pair(r) for r in [eq, *es, sa]) + " \\\\\n"
    blocks["ADAPTER TABLE"] = extra + "\\bottomrule\\end{tabular}\\end{table*}\n"
    for label, content in blocks.items():
        pattern = rf"(% BEGIN AUTO {label}\n).*?(% END AUTO {label})"
        text, count = re.subn(pattern, lambda m: m[1] + content + m[2], text, flags=re.S)
        if count != 1:
            raise ValueError("missing unique table marker")
    source.write_text(text)
    paths = SOURCES + [EQMAP, COMPLETE / "eqmap_3in_delay.json", COMPLETE / "baselines_3in.json", COMPLETE / "baselines_4in.json",
                      ROOT / "results_v2/abc_map_primary_20261012/baselines_3in.json",
                      ROOT / "results_v2/abc_map_primary_20261012/baselines_4in.json"]
    followup_paths = FOLLOWUPS + OBJECTIVE_REPORTS + PIPELINE_REPORTS
    followups = [load(p) for p in followup_paths]
    (HERE / "table_provenance.json").write_text(json.dumps({
        "sources": [{"path": str(p.relative_to(ROOT)),
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
        "successful_standard_records": sum(r["status"] == "ok" for d in standard_data for r in d["results"]),
        "successful_eqmap_adapter_records": sum(r["status"] == "ok" for r in eqmap_data["results"]),
        "failed_eqmap_adapter_records": sum(r["status"] != "ok" for r in eqmap_data["results"]),
        "additional_evidence": [{"path": str(p.relative_to(ROOT)),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "library_sha256": d["library_sha256"],
            "successful_records": sum(r["status"] == "ok" for r in d["results"]),
            "failed_records": sum(r["status"] != "ok" for r in d["results"])}
            for p, d in zip(followup_paths, followups)],
        "note": "Primary tables use the completed fixed-resyn2 suite. Historical and failed adapter evidence remains separate; synthetic libraries are not aggregated with ASAP7."
    }, indent=2) + "\n")
    with (HERE / "all_results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["library_max_inputs", "design", "method", "status", "area", "delay", "cec"])
        for k, data in [(3, standard_data[0]), (4, standard_data[1]),
                        (3, load(COMPLETE / "baselines_3in.json")),
                        (4, load(COMPLETE / "baselines_4in.json")),
                        (3, objectives[1]), (4, load(ROOT / "results_v2/fixed_4in_delay_20261012/report.json"))]:
            for r in data["results"]:
                m = r.get("measured", {})
                writer.writerow([k, r["bench"], r["method"], r["status"],
                                 m.get("area"), m.get("delay"), m.get("cec")])


if __name__ == "__main__":
    main()
