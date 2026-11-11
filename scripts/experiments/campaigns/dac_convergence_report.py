"""Read-only progress snapshot. Failed/missing pairs remain visible, no winners selected."""
import argparse,csv,fcntl,hashlib,json,os,pathlib,socket,sys
R=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(R/'scripts/experiments/diagnostics/emapper_stamp_cost'));from resources import summarize
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args();assert not a.output.exists()
 locks=[]
 for p in [R/'results_v2/independent_components_20261019/global.lock',a.campaign.parent/'runner.lock']:
  f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 c=json.loads(a.campaign.read_text());assert c['host']==socket.gethostname(),'sample tick/page units require the same recorded host';a.output.mkdir(parents=True);rows=[];checked=0;invalid=[]
 (a.output/'campaign-snapshot.json').write_bytes(a.campaign.read_bytes())
 (a.output/'report-helper.py').write_bytes(pathlib.Path(__file__).read_bytes())
 for n,h in c['frozen'].items():
  p=a.campaign.parent/'frozen'/n;checked+=1
  if not p.is_file()or sha(p)!=h:invalid.append(str(p))
 for key,v in c['jobs'].items():
  for n,h in v['artifacts'].items():
   p=pathlib.Path(v['case'])/n;checked+=1
   if not p.is_file()or sha(p)!=h:invalid.append(str(p))
  if sha(pathlib.Path(v['sample_file']))!=v['sample_sha256']:invalid.append(v['sample_file'])
  row={k:v.get(k)for k in ['key','design','rep','variant','phase','status','error','report_status','A','D','T','CEC','initial_AD','first_cover_time','first_independent_CEC','search_net_gain','search_timeout','search_finished','qualified_clock_chain','output_sha256']}
  row['initial_cover_origin']='native mapping, no external seed'
  if v['status']=='ok':
   prefix=v.get('observed_prefix',{});details=prefix.get('details',{});row['prefix']=dict(elapsed=prefix.get('elapsed'),exclusive_seconds=prefix.get('exclusive_seconds'),cuts=details.get('matched_cuts'),pool=details.get('pool'),selected_capture=details.get('selected_evidence'),trial_diagnostics=details.get('trial_diagnostics'),telemetry=details.get('telemetry'))
   row['resources']=summarize(v['sample_file'],os.sysconf('SC_CLK_TCK'));row['observed_peak_RSS_bytes_lower_bound']=max(json.loads(l).get('native',{}).get('rss_pages',0)for l in pathlib.Path(v['sample_file']).read_text().splitlines())*os.sysconf('SC_PAGE_SIZE')
  rows.append(row)
 interrupted=[]
 for item in c.get('interruptions',[]):
  p=pathlib.Path(item['audit']);checked+=1
  if not p.is_file()or sha(p)!=item['sha256']:invalid.append(str(p));continue
  audit=json.loads(p.read_text());assert not audit['eligible_for_PPA']
  for n,h in audit['artifacts'].items():
   f=pathlib.Path(audit['original_case'])/n;checked+=1
   if not f.is_file()or sha(f)!=h:invalid.append(str(f))
  entry={k:audit.get(k)for k in ['original_case','logical_key','retry_key','status','reason','last_observed_seconds','last_observed_AD','full_mapping_T','CEC','eligible_for_PPA']}
  supplement=p.with_name(p.stem+'-resources.json')
  if supplement.exists():
   resource=json.loads(supplement.read_text());sample=pathlib.Path(resource['source']);checked+=1
   if not sample.is_file()or sha(sample)!=resource['sha256']:invalid.append(str(sample))
   entry['original_resource_samples']=resource
  interrupted.append(entry)
 by={v['key']:v for v in rows};pairs=[]
 for rep in [0,1,2]:
  for design in ['dec','sin','sqrt','div']:
   x,y=[by.get(f'{design}_r{rep}_{v}')for v in ['off','on']]
   if x is None and y is None:continue
   pair=dict(design=design,rep=rep,complete=x is not None and y is not None,off_status=x.get('status')if x else'missing',on_status=y.get('status')if y else'missing')
   if all(v and v['status']=='ok'for v in [x,y]):
    pair.update(off_AD=[x['A'],x['D']],on_AD=[y['A'],y['D']],off_T=x['T'],on_T=y['T'],T_percent=100*(y['T']/x['T']-1),DA='better'if(y['D'],y['A'])<(x['D'],x['A'])else'equal'if(y['D'],y['A'])==(x['D'],x['A'])else'worse',same_netlist=x['output_sha256']==y['output_sha256'],both_search_deadline=x['search_timeout']and y['search_timeout'])
   pairs.append(pair)
 allplan=[j for p in c['plans'].values()for j in p];missing=[j['key']for j in allplan if j['key']not in c['jobs']]
 out=dict(completed=len(rows),planned_representatives=8,planned_repeats=8,rows=rows,pairs=pairs,interruptions=interrupted,missing=missing,invalid_artifacts=invalid,verified_artifact_references=checked,baseline_runs=0,default_promoted=False,uniform_config_selected=False,GitHub_push=False,source_campaign_sha256=sha(a.campaign),helper_sha256=sha(pathlib.Path(__file__)),scope='completed rows only, interrupted attempts retained separately and ineligible for full-budget PPA; sampled CPU spans differ from total mapping T; observation counters are prefixes; deadline completion is not convergence; any-time internal observations do not replace official delivered A/D')
 (a.output/'validation.json').write_text(json.dumps(out,indent=2)+'\n')
 with(a.output/'results.csv').open('w')as f:
  fields=['key','design','rep','variant','phase','status','A','D','T','CEC','first_cover_time','search_net_gain','search_timeout','search_finished'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
 lines=['# DAC convergence progress','',f'Completed {len(rows)} / 8 representative + 8 repeat jobs; no baseline runs, no default promotion.','', '| design | rep | off T | on T | T change | (D,A) |','|---|---:|---:|---:|---:|---|']
 for p in pairs:lines.append(f"| {p['design']} | {p['rep']} | {p.get('off_T','missing')} | {p.get('on_T','missing')} | {p.get('T_percent','unknown')} | {p.get('DA','incomplete/failed')} |")
 lines+=['','All failures, regressions and missing jobs remain in validation.json. This is not a full-EPFL claim.']
 (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(completed=len(rows),pairs=pairs,invalid_artifacts=len(invalid)),indent=2));assert not invalid
if __name__=='__main__':main()
