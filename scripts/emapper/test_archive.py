import json,tempfile,unittest
from pathlib import Path
from archive import seal,verify_files,sha
class DurableCapsuleTests(unittest.TestCase):
 def test_seed_files_selected_snapshots_and_resume_remain_verifiable(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);case=root/'case';case.mkdir();(case/'incumbents/0001').mkdir(parents=True);(case/'incumbents/0002').mkdir();(case/'incumbents/0001/mapped.v').write_text('old');selected=case/'incumbents/0002/manifest.json';selected.write_text('selected');(case/'incumbents/seed-recovery.json').write_text('seed');(case/'raw.v').write_text('final');(case/'raw.incumbent.receipt.json').write_text(json.dumps({'manifest':str(selected)}));manifest={str(p.relative_to(case)):sha(p)for p in case.rglob('*')if p.is_file()};capsule=seal(case,root/'archives');verify_files(case,manifest,capsule);self.assertTrue(selected.is_file());self.assertTrue((case/'incumbents/seed-recovery.json').is_file());self.assertFalse((case/'incumbents/0001').exists());self.assertEqual(seal(case,root/'archives'),capsule)
 def test_unverified_partial_archive_is_preserved_before_new_archive(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);case=root/'case';case.mkdir();(case/'raw.v').write_text('retained');archives=root/'archives';archives.mkdir();(archives/'case.tar.zst').write_bytes(b'partial');capsule=seal(case,archives);self.assertEqual((archives/'case.tar.zst.interrupted-0').read_bytes(),b'partial');verify_files(case,{'raw.v':sha(case/'raw.v')},capsule)
if __name__=='__main__':unittest.main()
