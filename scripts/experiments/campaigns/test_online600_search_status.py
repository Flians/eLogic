import unittest
from online600_search_status import classify
class SearchStatus(unittest.TestCase):
 def test_native_failure_cannot_be_hidden_by_frontend_exit_zero(self):
  s=classify({'mapping':{'exit_code':0,'timeout':False},'native':{'search_finished':True,'native_exit_code':7},'status':'ok','measured':{'cec':1}})
  self.assertFalse(s['search_finished']);self.assertTrue(s['optimized_output_independent_CEC'])
 def test_outer_timeout_without_native_summary_is_incomplete(self):
  self.assertFalse(classify({'mapping':{'exit_code':-9,'timeout':True}})['search_finished'])
 def test_complete_requires_both_successes(self):
  self.assertTrue(classify({'mapping':{'exit_code':0,'timeout':False},'native':{'native_exit_code':0,'search_deadline_hit':False}})['search_finished'])
 def test_native_deadline_and_native_kill_are_never_complete(self):
  for code,hit in [(0,True),(-9,False)]:
   self.assertFalse(classify({'mapping':{'exit_code':0,'timeout':False},'native':{'native_exit_code':code,'search_deadline_hit':hit}})['search_finished'])
if __name__=='__main__':unittest.main()
