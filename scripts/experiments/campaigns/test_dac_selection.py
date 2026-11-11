import copy,unittest
from run_dac_frozen_epfl import selection

def cohort():
 jobs={};plan=[]
 for n,reps in [('dec',[0,1,2]),('sin',[0,1,2]),('sqrt',[0]),('div',[0])]:
  for rep in reps:
   for v in ['off','on']:
    key=f'{n}_r{rep}_{v}';plan.append(dict(key=key));jobs[key]=dict(status='ok',CEC=1,first_independent_CEC=True,qualified_clock_chain=True,T=(90 if v=='on' else 100),D=10,A=10,search_finished=True,configuration=dict(trace_write='line'if v=='on'else'fragmented',evidence_arrays='move'if v=='on'else'copy',mapping_seconds=600,environment=dict(EGM_TRACE_LINE_WRITE='1'if v=='on'else'0',EGM_EVIDENCE_ARRAY_MOVE='1'if v=='on'else'0',EGM_K='6')))
 return dict(jobs=jobs,plans=dict(all=plan),selection_policy='fixed test policy')
class Selection(unittest.TestCase):
 def test_uniform_policy_passes_but_never_picks_repeat_winners(self):
  c=cohort();self.assertEqual(selection(c)['variant'],'on');c['jobs']['dec_r1_on']['T']=120
  self.assertEqual(selection(c)['variant'],'off','positive median cannot hide worse total runtime')
 def test_any_quality_regression_or_missing_evidence_blocks_promotion(self):
  for field,value in [('D',11),('CEC',0),('qualified_clock_chain',False)]:
   c=cohort();c['jobs']['sin_r2_on'][field]=value;self.assertEqual(selection(c)['variant'],'off')
 def test_deadline_runs_and_unplanned_differences_are_not_speed_evidence(self):
  c=cohort();c['jobs']['dec_r0_on']['search_finished']=False;self.assertEqual(selection(c)['variant'],'off')
  c=cohort();c['jobs']['sqrt_r0_on']['configuration']['environment']['EGM_K']='5'
  with self.assertRaises(AssertionError):selection(c)
 def test_incomplete_scope_cannot_start_full_epfl(self):
  c=cohort();c['jobs'].pop('div_r0_on')
  with self.assertRaises(AssertionError):selection(c)
if __name__=='__main__':unittest.main()
