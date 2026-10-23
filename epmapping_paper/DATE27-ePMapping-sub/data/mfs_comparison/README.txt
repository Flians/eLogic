Numerical sources for the horizontally grouped mfs3 comparison.
Copied byte-for-byte from results_v20_three_parts_chain. The retained
verified_metrics.json records the original campaign's metrics and hashes;
the three TSV sources and sixteen own-run logs needed for this table
are included here. Initial native costs come from the TSVs; own optimizer
runtimes come from the hashed logs. ABC runtime is its measured command
duration, excluding CEC but including import and output writing.

From DATE27-ePMapping-sub, run `make verify-mfs` to check source hashes,
CEC/status fields, the 13-design paired means and the 16-design own mean,
and regenerate doc/generated_mfs.tex. The full campaign audit and raw
netlists remain in the original experiment archive.
