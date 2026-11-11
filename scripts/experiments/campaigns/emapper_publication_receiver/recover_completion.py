"""Recover a completed report after a *post-run* Cargo.lock ordering assertion.
Never resumes mapping, edits the original checkpoint, or replaces an incumbent.
"""
import copy,fcntl,hashlib,importlib.util,json,pathlib,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[4];P=R/'results_v2/emapper_publication_receiver_20261111';O=P/'completion-recovery'
sys.path.insert(0,str(pathlib.Path(__file__).parent));import run as driver
sys.path.insert(0,str(R/'scripts/emapper'));from frontend import verify_receipt
from provenance import sha
try:import tomllib as tomli
except ImportError:from pip._vendor import tomli

def normalized_lock(data):
 j=tomli.loads(data.decode());j['package'].sort(key=lambda r:(r['name'],r['version'],r.get('source','')));return j

def main():
 locks=[]
 for path in [R/'results_v2/independent_components_20261019/global.lock',P/'benchmark/runner.lock']:
  f=path.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 assert not O.exists(),'preserve partial recovery; use an explicitly audited new output'
 c=json.loads((P/'benchmark/campaign.json').read_text());assert len(c['jobs'])==7
 current=driver.fingerprint();drift=[k for k in sorted(set(current)|set(c['frozen']))if current.get(k)!=c['frozen'].get(k)]
 assert drift==['Cargo.lock'],drift
 old=subprocess.check_output(['git','show',c['implementation_commit']+':Cargo.lock'],cwd=R);new=(R/'Cargo.lock').read_bytes()
 assert hashlib.sha256(old).hexdigest()==c['frozen']['Cargo.lock']
 a,b=normalized_lock(old),normalized_lock(new);assert a==b,'dependency semantic change is not an ordering repair'
 job=next(j for j in c['plan']if j['key']not in c['jobs']);assert job['key']=='sin_r1_events'
 folder=P/'benchmark/cases'/job['key'];r=json.loads((folder/'report.json').read_text());inp=c['subjects']['sin']
 assert r['status']=='ok'and r['cec']==1 and r['evaluation']['exact_agreement']and r['selected_evidence']['valid']and r['selected_evidence']['netlist_binding_agreement']
 assert r['configuration']==c['jobs']['sin_r0_events']['configuration']
 assert r['direct_native']and not r.get('external_seed')and not r['seed_generation_in_mapping_clock']
 assert not [(k,v)for k,v in r['dependencies'].items()if k in c['frozen']and v!=c['frozen'][k]]
 for k in ['target/release/emapper_kernel','target/release/emapper_measure','target/release/emapper-build.json']:
  assert r['dependencies'][k]==c['frozen'][k]==sha(R/k)
 archived=json.loads((P/'executed-release/emapper-build.json').read_text());assert archived['source']['Cargo.lock']==c['frozen']['Cargo.lock']
 affinity=json.loads((folder/'native-affinity.json').read_text());assert affinity['cpus_allowed_list']=='6'and affinity['nonce']==hashlib.sha256(str(folder).encode()).hexdigest()
 verified=verify_receipt(folder/'raw.incumbent.receipt.json',R/'emapper/libraries/asap7/asap7_full.genlib',pathlib.Path(inp['prepared']),R/'target/release/emapper_kernel')
 assert [verified['area'],verified['delay']]==[r['measured']['area'],r['measured']['delay']]
 first=r['first_native_cover'];ad0=[first['area'],first['delay']];ad1=[r['measured']['area'],r['measured']['delay']]
 artifacts=dict(r['artifacts']);artifacts['report.json']=sha(folder/'report.json')
 # Hashes captured by the original mapping run are independently reread by the
 # completion audit. They are not falsely labelled verified by this recovery.
 row=dict(**job,case=str(folder),exit=None,status='ok',report_status='ok',T=r['mapping_wall_secs'],A=ad1[0],D=ad1[1],initial_AD=ad0,CEC=1,search_timeout=r['search_timeout'],search_finished=r['search_finished'],first_native_cover=first,selected_evidence=r['selected_evidence'],output_sha256=r['output_sha256'],configuration=r['configuration'],search_net_gain='better'if(ad1[1],ad1[0])<(ad0[1],ad0[0])else'equal',artifacts=artifacts,recovery='completed native report and receipt independently qualified after original post-run assertion; child wrapper exit not persisted; original batch exit failure retained')
 proof=dict(original_registry_sha256=sha(P/'benchmark/campaign.json'),original_failure_log_sha256=sha(P/'batch-004.log'),original_registered_jobs=7,completed_native_reports=8,changed_path='Cargo.lock',frozen_byte_sha256=hashlib.sha256(old).hexdigest(),current_byte_sha256=hashlib.sha256(new).hexdigest(),normalized_semantics_sha256=hashlib.sha256(json.dumps(a,sort_keys=True,separators=(',',':')).encode()).hexdigest(),same_lock_dependency_semantics=True,all_other_frozen_dependencies_unchanged=True,original_report_sha256=sha(folder/'report.json'),executed_binary_and_build_receipt_unchanged=True,independent_whole_artifact_verification_pending=True,mapping_rerun=False,original_checkpoint_modified=False,old_incumbents_replaced=False,helper_sha256=sha(pathlib.Path(__file__)))
 O.mkdir();(O/'original-Cargo.lock').write_bytes(old);(O/'current-Cargo.lock').write_bytes(new)
 (O/'recovery.json').write_text(json.dumps(proof,indent=2)+'\n');c=copy.deepcopy(c);c['jobs'][job['key']]=row;c['completion_recovery']=proof;c['original_registry_count']=7
 (O/'campaign.json').write_text(json.dumps(c,indent=2)+'\n');(O/'state.json').write_text(json.dumps(dict(active=None,completed=8,expected=8,recovered_completed_report=1,baseline_runs=0,default_promoted=False,GitHub_push=False),indent=2)+'\n')
 print(json.dumps(proof,indent=2))
if __name__=='__main__':main()
