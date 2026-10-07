# Verified ePMapping paper experiment archive

This directory preserves the audited measurements, raw engine/CEC logs, mapped
netlists, input snapshots, recorded batch specifications and SHA256 manifests.
`verified_metrics.json` and `tables.md` summarize the verified results.
`inputs/epmapping_check` is the frozen experiment executable, not a build cache;
its source snapshot and provenance are recorded in `manifest.json`.

`cancelled_replica/` and `validation_intermediate/` retain historical evidence;
they do not contribute to the published result tables. The independent paper
repository is located at `epmapping_paper/`.

The original manifest describes the experiment-time files. Subsequent integration
of the mockturtle patch into CMake and the new patch-application test does not
change the optimizer or the archived experimental results. A current rebuild
can be compared with the recorded engine hash before regenerating tables:

```
python3 scripts/experiments/export_paper.py --root results_v3_audit \
  --out epmapping_paper/DATE27-ePMapping-sub/doc/generated_results.tex
```

For a fresh run, use `scripts/experiments/reproduce_paper.sh` with a new
`EPM_ROOT`, as documented in `epmapping/README.md`.
