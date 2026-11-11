"""Telemetry-only JSONL reader: preserve a killed final fragment explicitly."""
import hashlib,json

def read(path):
 data=path.read_bytes();parts=data.splitlines(keepends=True);rows=[];tail=None;offset=0
 for i,line in enumerate(parts):
  try:rows.append(json.loads(line))
  except (ValueError,UnicodeDecodeError):
   if i!=len(parts)-1 or line.endswith(b'\n'):raise
   tail=dict(offset=offset,bytes=len(line),sha256=hashlib.sha256(line).hexdigest(),reason='unterminated last telemetry record; raw file preserved; never a coverage certificate')
  offset+=len(line)
 return rows,tail
