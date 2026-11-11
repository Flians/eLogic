import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'campaigns'))
from run_dac_fixed_pool_diagnosis import inventory, summarize_events


class PoolInventory(unittest.TestCase):
    def snapshot(self):
        return dict(schema=1,selected_context={'logic':{}},free=[True,False,False,False],
                    constants=[None]*4,chosen=[None,None,dict(instance=0,pin=0),None],
                    witness=None,bank=[[],[]],initial_ad=[1,1],context={'cover_revision':1},
                    pool=[dict(inputs=[0],input_pins=['A'],
                               outputs=[dict(delays=[1],targets=[2],sources=[])])])

    def run_inventory(self, data):
        with tempfile.TemporaryDirectory()as folder:
            path=Path(folder)/'snapshot.json';path.write_text(json.dumps(data))
            return inventory(path)

    def test_absent_source_is_unresolved_not_proved_usable(self):
        result=self.run_inventory(self.snapshot())
        self.assertEqual(result['targets_without_explicit_source'],1)
        self.assertTrue(result['complete_admitted_pool_at_capture'])
        self.assertFalse(result['all_future_candidates_enumerated'])
        self.assertFalse(result['producer_cursor_serialized'])

    def test_source_record_retains_full_pool_identity(self):
        data=self.snapshot();data['pool'][0]['outputs'][0]['sources']=[{'target':2}]
        before=copy.deepcopy(data);result=self.run_inventory(data)
        self.assertEqual(result['recorded_binding_sources'],1)
        self.assertEqual(result['targets_without_explicit_source'],0)
        self.assertEqual(data,before)

    def test_invalid_binding_or_physical_input_rejected(self):
        data=self.snapshot();data['chosen'][2]['instance']=1
        with self.assertRaises(AssertionError):self.run_inventory(data)
        data=self.snapshot();data['pool'][0]['inputs']=[4]
        with self.assertRaises(AssertionError):self.run_inventory(data)

    def test_partial_diagnostic_retains_negative_rows_and_tail(self):
        with tempfile.TemporaryDirectory()as folder:
            path=Path(folder)/'events.jsonl'
            rows=[{'instance':2,'physical_unavailable':'cycle'},
                  {'instance':3,'oracle_area_improves_under_initial_D':True,'source_error':'missing proof'},
                  {'instance':4,'verified_missed_proposal':True}]
            path.write_bytes(('\n'.join(json.dumps(r)for r in rows)+'\n').encode()+b'{"instance":')
            result=summarize_events(path)
            self.assertEqual(result['complete_rows'],3)
            self.assertEqual(result['physical_unavailable'],1)
            self.assertEqual(result['source_blocked_improvements'],1)
            self.assertEqual(result['verified_missed_instances'],[4])
            self.assertGreater(result['truncated_tail_bytes'],0)

    def test_corrupt_completed_row_rejected(self):
        with tempfile.TemporaryDirectory()as folder:
            path=Path(folder)/'events.jsonl';path.write_bytes(b'not JSON\n')
            with self.assertRaises(AssertionError):summarize_events(path)


if __name__=='__main__':unittest.main()
