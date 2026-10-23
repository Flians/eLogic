"""Replay native stage events: recovery/cuts must run while wide search is unfinished."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('trace',type=Path);a=p.parse_args();events=[json.loads(x)for x in a.trace.read_text().splitlines()];started=False;finished=False;yielded=False;recovery=False;cuts=False
for e in events:
 s=e['stage']
 if s in ['online_wide_start','online_stream_wide_start']:started=True
 elif s in ['online_wide_done','online_stream_wide_done']:finished=True
 elif started and not finished:
  yielded |= s=='online_stream_wide_yield'
  recovery |= s in ['online_recovery_slice','online_cone_area_start']
  cuts |= s.startswith('cut_round_')and s.endswith('_start')
print(json.dumps(dict(wide_started=started,wide_yielded=yielded,recovery_while_wide_pending=recovery,cuts_while_wide_pending=cuts)))
raise SystemExit(0 if started and yielded and recovery and cuts else 1)
