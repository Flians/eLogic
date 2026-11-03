"""Controller-only simulated jobs: verifies all four names survive diagnostics.
No native mapping or real CEC is performed by this unit test.
"""
import importlib.util,json,os,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/emapper'))
class DriverTests(unittest.TestCase):
 def check_driver(self,path):
  spec=importlib.util.spec_from_file_location('tested_p0_driver',path);d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
  with tempfile.TemporaryDirectory(prefix='emapper-p0-driver-')as td:
   root=Path(td);subject=root/'tiny.aig';subject.write_text('controller fixture, not a real AIG')
   protocol=root/'results_v2/emapper_seeded_epfl_20261101/report.json';protocol=protocol.with_name('protocol.json');protocol.parent.mkdir(parents=True)
   inp={k:str(subject)for k in ['prepared','golden','seed']};inp.update({k+'_sha256':d.sha(subject)for k in ['prepared','golden','seed']});protocol.write_text(json.dumps({'inputs':{n:inp for n in ['sin','max']}}))
   visited=[]
   def run(argv,**kwargs):
    case=Path(argv[argv.index('--output')+1]);case.mkdir(parents=True);visited.append(case.name)
    (case/'incumbents/00000001').mkdir(parents=True);(case/'incumbents/00000001/manifest.json').write_text(json.dumps({'kind':'complete-source-cover','area':1.,'delay':1.}))
    (case/'report.json').write_text(json.dumps({'status':'ok','cec':1,'measured':{'area':1.,'delay':1.},'external_native_import':{'applied':True}}));return SimpleNamespace(returncode=0)
   out=root/'study'
   with patch.object(d,'R',root),patch.object(d,'git_revision',return_value='fixture'),patch.object(d,'fingerprint',return_value={'target/release/emapper_kernel':'fixture'}),patch.object(d.shutil,'which',return_value='/usr/bin/true'),patch.object(d.subprocess,'run',side_effect=run),patch.object(d,'measure_cover',return_value={'measured':{'area':1.,'delay':1.}}),patch.object(d,'verify',return_value=True),patch.object(d,'seal',return_value=None),patch.object(d,'verify_files'),patch.object(sys,'argv',['runner','--output',str(out),'--max-jobs','4']):d.main()
   c=json.loads((out/'campaign.json').read_text());self.assertEqual(len(c['jobs']),4);self.assertEqual(visited,['sin_legacy','sin_window8','max_window8','max_legacy']);self.assertEqual(json.loads((out/'state.json').read_text())['status'],'complete')
 def test_heavy_controller_completes_all_names(self):self.check_driver(Path(os.environ.get('EMAPPER_TEST_P0_DRIVER',Path(__file__).with_name('run_emapper_window_p0.py'))))
 def test_light_controller_completes_all_names(self):self.check_driver(Path(__file__).with_name('run_emapper_window_p0_light.py'))

class ReplayDriverTests(unittest.TestCase):
 def test_eight_replays_keep_source_keys_and_validate_initial(self):
  path=Path(__file__).with_name('run_emapper_p0_replay.py');spec=importlib.util.spec_from_file_location('tested_p0_replay',path);d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
  with tempfile.TemporaryDirectory(prefix='emapper-p0-replay-')as td:
   root=Path(td);subject=root/'tiny';subject.write_text('controller fixture only');snap=root/'recovery-snapshot.json';snap.write_text('{}');report=root/'origin.json';report.write_text('{"cec":1}')
   inp={k:str(subject)for k in ['prepared','golden','seed']};inp.update({k+'_sha256':d.sha(subject)for k in ['prepared','golden','seed']})
   origin=root/'origin';origin.mkdir();(origin/'recovery-snapshot.json').write_bytes(snap.read_bytes())
   source=root/'source.json';source.write_text(json.dumps({'jobs':{k:{'case':str(origin),'diagnostic':{'errors':[],'snapshot_summary':{'initial_AD':[1.,1.]}},'initial':{'CEC':1},'report':str(report),'report_sha256':d.sha(report),'artifacts':{'recovery-snapshot.json':d.sha(snap)}}for k in ['sin_legacy','sin_window8','max_window8','max_legacy']},'inputs':{'sin':inp,'max':inp},'arms':{'legacy':0,'window8':8}}))
   def run(argv,**kwargs):
    case=Path(argv[argv.index('--output')+1]);case.mkdir(parents=True);(case/'incumbents/00000001').mkdir(parents=True);(case/'incumbents/00000001/manifest.json').write_text('{"area":1.0,"delay":1.0}')
    (case/'report.json').write_text(json.dumps({'status':'ok','cec':1,'measured':{'area':1.,'delay':1.},'evaluation':{'exact_agreement':True},'diagnostic_scope':'fixed candidate/proof snapshot fixture','snapshot_sha256':d.sha(Path(argv[argv.index('--replay-snapshot')+1]))}));return SimpleNamespace(returncode=0)
   out=root/'study'
   with patch.object(d,'R',root),patch.object(d,'SOURCE',source),patch.object(d,'git_revision',return_value='fixture'),patch.object(d,'fingerprint',return_value={}),patch.object(d.shutil,'which',return_value='/usr/bin/true'),patch.object(d.subprocess,'run',side_effect=run),patch.object(d,'measure_cover',return_value={'measured':{'area':1.,'delay':1.},'netjson':str(subject)}),patch.object(d,'verify',return_value=True),patch.object(d,'seal',return_value=None),patch.object(d,'verify_files'),patch.object(sys,'argv',['replay','--output',str(out),'--max-jobs','8']):d.main()
   c=json.loads((out/'campaign.json').read_text());self.assertEqual(len(c['jobs']),8);self.assertTrue(all(j['valid_replay']for j in c['jobs'].values()));self.assertEqual(json.loads((out/'state.json').read_text())['status'],'complete')

if __name__=='__main__':unittest.main()
