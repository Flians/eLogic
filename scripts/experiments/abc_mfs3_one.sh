#!/usr/bin/env bash
# One measured mfs3 run and a separately timed CEC against the original AIGER.
set -uo pipefail
wd="$1" design="$2" lib="$3" baseline="$4" benchmark="$5" cap="${6:-1000}"
abc="${ABC:-abc}"
mkdir -p "$wd"
out="$wd/mapped.v"
start=$SECONDS
perl -e 'alarm shift; exec @ARGV' "$cap" "$abc" -c "read_genlib '$lib'; read -m '$baseline'; mfs3 -a; print_stats; write_verilog '$out'" > "$wd/log" 2>&1
rc=$?
secs=$(( SECONDS-start ))
area=$(sed -n 's/.*[^a-z]area *= *\([0-9.]*\).*/\1/p' "$wd/log" | tail -1)
delay=$(sed -n 's/.*[^a-z]delay *= *\([0-9.]*\).*/\1/p' "$wd/log" | tail -1)
cec=-; validation_secs=0
if [ "$rc" = 0 ] && [ -f "$out" ]; then
  start=$SECONDS
  perl -e 'alarm shift; exec @ARGV' "$cap" "$abc" -c "read_genlib '$lib'; read -m '$out'; cec -n '$benchmark'" > "$wd/cec.log" 2>&1
  validation_secs=$(( SECONDS-start ))
  cec=0
  if grep -q 'Networks are equivalent' "$wd/cec.log"; then cec=1; fi
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$design" "${area:--}" "${delay:--}" "$secs" "$rc" "$cec" "$validation_secs"
