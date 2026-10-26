"""Keep same-binary ablations limited to explicit joint-search choices."""
import json
import tempfile
import unittest
from pathlib import Path
from run import ablation_profile, select_output
from frontend import fnv, sha


class OutputSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root/'raw.v'
        self.raw.write_text('source cover')
        self.observed = dict(optimizer_succeeded=False, selected='source_witness',
                             source_sha256=sha(self.raw))
        self.library, self.subject, self.native = [self.root/n for n in ['lib','aig','bin']]
        for p in [self.library,self.subject,self.native]:
            p.write_text(p.name)

    def receipt(self):
        net = self.root/'incumbent.v';net.write_text('optimized complete cover')
        manifest = self.root/'manifest.json'
        values = dict(kind='complete-generated-cover', fnv64=fnv(net.read_bytes()),
                      area=3.,delay=2.,sequence=1,instances=2,matched_cuts=4)
        manifest.write_text(json.dumps(values))
        r = dict(values, netlist=str(net),manifest=str(manifest),
                 netlist_sha256=sha(net),manifest_sha256=sha(manifest),
                 observed_monotonic=9.,deadline_monotonic=10.,
                 provenance=dict(library_sha256=sha(self.library),
                                 prepared_sha256=sha(self.subject),native_sha256=sha(self.native)))
        self.raw.with_suffix('.incumbent.receipt.json').write_text(json.dumps(r))
        return r

    def select(self):
        return select_output(self.raw,self.observed,self.library,self.subject,self.native)

    def test_killed_child_initial_metadata_does_not_hide_verified_incumbent(self):
        r=self.receipt();saved=json.dumps(self.observed,sort_keys=True)
        effective,selection=self.select()
        self.assertTrue(effective['optimizer_succeeded'])
        self.assertEqual(effective['selected'],'validated_online_incumbent')
        self.assertEqual(selection['incumbent'],r)
        self.assertEqual(selection['kind'],'optimized')
        self.assertEqual(self.raw.read_bytes(),Path(r['netlist']).read_bytes())
        self.assertEqual(json.dumps(self.observed,sort_keys=True),saved)

    def test_source_without_receipt_stays_source_only(self):
        effective,selection=self.select()
        self.assertFalse(effective['optimizer_succeeded'])
        self.assertEqual(selection['kind'],'source_only')

    def test_child_success_claim_without_receipt_does_not_authorize_optimization(self):
        self.observed.update(optimizer_succeeded=True,selected='validated_online_incumbent')
        effective,selection=self.select()
        self.assertFalse(effective['optimizer_succeeded'])
        self.assertEqual(selection['kind'],'source_only')

    def test_unreceipted_modified_file_fails_closed(self):
        self.raw.write_text('partial worker output')
        with self.assertRaisesRegex(RuntimeError,'no verified incumbent'):
            self.select()

    def test_corrupt_receipt_fails_without_overwriting_source(self):
        r=self.receipt();Path(r['netlist']).write_text('corrupted')
        with self.assertRaises(AssertionError):self.select()
        self.assertEqual(self.raw.read_text(),'source cover')

    def test_late_receipt_fails_without_overwriting_source(self):
        r=self.receipt();r['observed_monotonic']=10.
        self.raw.with_suffix('.incumbent.receipt.json').write_text(json.dumps(r))
        with self.assertRaises(AssertionError):self.select()
        self.assertEqual(self.raw.read_text(),'source cover')

    def test_different_input_or_library_cannot_reuse_receipt(self):
        self.receipt()
        for path in [self.subject,self.library,self.native]:
            with self.subTest(changed=path.name):
                before=path.read_bytes();path.write_text('different dependency')
                with self.assertRaises(AssertionError):self.select()
                self.assertEqual(self.raw.read_text(),'source cover')
                path.write_bytes(before)

class AblationProfileTests(unittest.TestCase):
    def test_variants_preserve_every_other_algorithm_setting_and_base(self):
        base = json.loads((Path(__file__).resolve().parents[2]/'emapper/profiles/delivery.json').read_text())
        saved = json.dumps(base, sort_keys=True)
        self.assertEqual(ablation_profile(base, None), base)
        for arm, windows in [('joint-off', '0'), ('joint-filtered', '1')]:
            actual = ablation_profile(base, arm)
            self.assertEqual(actual['mapping_seconds'], 600)
            expected_env = dict(base['environment'], EGM_JOINT_WINDOWS=windows, EGM_JOINT_TARGET_GROUPS='0')
            self.assertEqual(actual['environment'], expected_env)
            for key in base.keys()-{'environment','name','status'}:
                self.assertEqual(actual[key], base[key])
        area_only = ablation_profile(base, 'joint-area-only')
        self.assertEqual(area_only['environment'], dict(base['environment'], EGM_JOINT_DELAY_AREA='0'))
        for key in base.keys()-{'environment','name','status'}:
            self.assertEqual(area_only[key], base[key])
        functional = ablation_profile(base, 'functional-stream')
        self.assertEqual(functional['environment'], dict(base['environment'], EGM_FUNCTIONAL_CURSOR='1'))
        self.assertEqual(functional['mapping_seconds'],600)
        sparse = ablation_profile(base, 'functional-sparse')
        self.assertEqual(sparse['environment'], dict(functional['environment'], EGM_PRIMITIVE_SPARSE='1'))
        self.assertEqual(sparse['mapping_seconds'],600)
        fine = ablation_profile(base,'functional-fine')
        self.assertEqual(fine['environment'],dict(sparse['environment'],EGM_PROOF_AWARE_JOINT='1'))
        self.assertEqual(fine['mapping_seconds'],600)
        lazy = ablation_profile(base,'functional-lazy')
        self.assertEqual(lazy['environment'],dict(fine['environment'],EGM_SDC_LAZY_LAYOUT='1'))
        self.assertEqual(lazy['mapping_seconds'],600)
        self.assertEqual(json.dumps(base, sort_keys=True), saved)
        with self.assertRaises(ValueError):
            ablation_profile(base, 'unknown')

if __name__ == '__main__':
    unittest.main()
