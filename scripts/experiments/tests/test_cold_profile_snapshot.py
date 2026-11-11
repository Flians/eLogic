#!/usr/bin/env python3
"""Check cold-only training when a warm control adds cache records."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from preserve_cold_profile_pools import preserve
from build_readonly_profile_library import snapshot


class ColdSnapshot(unittest.TestCase):
    def test_warm_extras_stay_out_of_frozen_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();pool=root/'cache_sources/main/ours/test';pool.mkdir(parents=True)
            old=pool/'old.json';old.write_bytes(b'old record');os.utime(old,ns=(10,10))
            fresh=pool/'new.json';fresh.write_bytes(b'warm record');os.utime(fresh,ns=(200,200))
            for mode,count,mtime in [('cold',1,100),('warm',2,300)]:
                p=root/'main/ours'/(mode+'.test')/'result.json';p.parent.mkdir(parents=True)
                p.write_text(json.dumps(dict(cache_files=count)));os.utime(p,ns=(mtime,mtime))
            original=snapshot(pool);r=preserve(root,'main','ours','test')
            self.assertTrue(pool.is_symlink())
            self.assertEqual(set(snapshot(pool)),{'old.json'})
            self.assertEqual(snapshot(root/r['warm_pool']),original)
            self.assertEqual(r['warm_extra_records'],1)
            self.assertEqual(r['cold_record_sha256']['old.json'],original['old.json'])
            self.assertEqual(preserve(root,'main','ours','test'),r)

    def test_ambiguous_cold_boundary_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();pool=root/'cache_sources/main/ours/test';pool.mkdir(parents=True)
            (pool/'record.json').write_text('record')
            for mode in ['cold','warm']:
                p=root/'main/ours'/(mode+'.test')/'result.json';p.parent.mkdir(parents=True)
                p.write_text(json.dumps(dict(cache_files=2)))
            with self.assertRaises(AssertionError):preserve(root,'main','ours','test')
            self.assertFalse(pool.is_symlink())
            self.assertEqual((pool/'record.json').read_text(),'record')


if __name__=='__main__':unittest.main()
