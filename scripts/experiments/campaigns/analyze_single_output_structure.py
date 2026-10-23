#!/usr/bin/env python3
"""Read-only structural diversity/cut-bank audit after all timed work has ended."""
import argparse,hashlib,json,re
from pathlib import Path
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def slice_field(raw,key,nextkey):
 a=raw.index(('"'+key+'":').encode())+len(key)+3;b=raw.index((',"'+nextkey+'":').encode(),a);return raw[a:b]
def split_nodes(s):
 assert s.startswith('[') and s.endswith(']');s=s[1:-1];out=[];start=0;depth=0;quoted=False;escaped=False
 for i,ch in enumerate(s):
  if quoted:
   if escaped:escaped=False
   elif ch=='\\':escaped=True
   elif ch=='"':quoted=False
  elif ch=='"':quoted=True
  elif ch in '([':depth+=1
  elif ch in ')]':depth-=1;assert depth>=0
  elif ch==',' and depth==0:out.append(s[start:i].strip());start=i+1
 assert depth==0 and not quoted
 if s[start:].strip():out.append(s[start:].strip())
 return out
def diversity(rows):
 out=dict(classes=len(rows),AND_classes=0,AND_nodes=0,commutative_duplicates=0,classes_with_distinct_AND_child_pairs=0,extra_distinct_AND_child_pairs=0,mixed_nonfree_node_forms=0)
 for row in rows:
  ns=split_nodes(row['nodes']);pairs=[];forms=set();free=any(n.startswith('Pi(') or n in ['Zero','One'] for n in ns)
  for n in ns:
   if n.startswith('And('):
    m=re.fullmatch(r'And\(\[\s*(\d+)\s*,\s*(\d+)\s*\]\)',n);assert m,n
    pair=tuple(sorted(map(int,m.groups())));pairs.append(pair);forms.add(('And',pair))
   else:forms.add(('other',n))
  if pairs:
   out['AND_classes']+=1;out['AND_nodes']+=len(pairs);out['commutative_duplicates']+=len(pairs)-len(set(pairs))
   if len(set(pairs))>1:out['classes_with_distinct_AND_child_pairs']+=1;out['extra_distinct_AND_child_pairs']+=len(set(pairs))-1
  if not free and len(forms)>1:out['mixed_nonfree_node_forms']+=1
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('campaign',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();c=json.loads(a.campaign.read_text());assert c['status']=='attempted' and len(c['jobs'])==c['expected_jobs']
 assert diversity([{'nodes':'[And([1, 2]), And([2, 1])]'}])['classes_with_distinct_AND_child_pairs']==0
 assert diversity([{'nodes':'[And([1, 2]), And([3, 4])]'}])['classes_with_distinct_AND_child_pairs']==1
 assert split_nodes('[Cell("quoted, cell", [1, 2]), Not(1)]')==['Cell("quoted, cell", [1, 2])','Not(1)']
 records=[]
 for j in c['jobs']:
  if j['profile']!='protected_area':continue
  r=json.loads(Path(j['report']).read_text())['results'][0]
  if r['status']!='ok':continue
  b=next(x for x in c['jobs'] if (x['k'],x['bench'],x['profile'])==(j['k'],j['bench'],'single_structure'));rb=json.loads(Path(b['report']).read_text())['results'][0]
  raw=Path(j['candidates']).read_bytes();graph=slice_field(raw,'egraph_nodes','free');cuts=slice_field(raw,'cuts','delay');rec=dict(k=j['k'],bench=j['bench'],source_diagnostic=str(j['candidates']),source_sha256=sha(j['candidates']),graph_sha256=hashlib.sha256(graph).hexdigest(),diversity=diversity(json.loads(graph)),full_cut_sha256=hashlib.sha256(cuts).hexdigest(),witness_status=rb['status'])
  if rb['status']=='ok':
   wb=Path(b['candidates']).read_bytes();wc=slice_field(wb,'cuts','delay');assert slice_field(wb,'egraph_nodes','free')==graph
   rec['witness_cut_sha256']=hashlib.sha256(wc).hexdigest();rec['cut_banks_byte_equal']=cuts==wc
   if cuts!=wc:
    key=lambda z:(z['class'],tuple(z['leaves']),z['function']);full={key(z) for z in json.loads(cuts)};wit={key(z) for z in json.loads(wc)};rec.update(full_only_cuts=len(full-wit),witness_only_cuts=len(wit-full))
  records.append(rec)
 out=dict(campaign=str(a.campaign),campaign_sha256=sha(a.campaign),records=records,valid_references=len(records),identical_optional_cut_banks=sum(r.get('cut_banks_byte_equal',False) for r in records),different_optional_cut_banks=sum('cut_banks_byte_equal' in r and not r['cut_banks_byte_equal'] for r in records),scope='Collapses AND input commutation only; not a semantic inequivalence proof. Zero distinct AND alternatives does not exclude phase/constant simplification or non-AND structural opportunities. Witness control retains identical source fallbacks; equality does not show saturation itself is useless. Read-only audit, never mapper advice.')
 a.output.write_text(json.dumps(out,indent=2)+'\n');print({k:out[k] for k in ['valid_references','identical_optional_cut_banks','different_optional_cut_banks']})
if __name__=='__main__':main()
