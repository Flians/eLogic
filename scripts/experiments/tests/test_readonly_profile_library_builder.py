#!/usr/bin/env python3
"""Check deterministic deduplication and refusal to merge conflicting records."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from build_readonly_profile_library import build,snapshot


class LibraryBuilder(unittest.TestCase):
    def test_duplicate_records_are_byte_preserved_and_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=root/'a';b=root/'b';a.mkdir();b.mkdir()
            key=hashlib.sha256(b'query').hexdigest()
            data=json.dumps(dict(schema=1,key=key,payload_sha256=hashlib.sha256(b'[]').hexdigest(),covers=[])).encode()
            for folder in [a,b]:(folder/(key+'.json')).write_bytes(data)
            library=root/'library';m=build(library,dict(first=a,second=b),{})
            try:
                self.assertEqual(m['unique_queries'],1)
                self.assertEqual(m['deduplicated_records'],1)
                self.assertEqual((library/(key+'.json')).read_bytes(),data)
                self.assertEqual(m['records'][key+'.json']['sources'],['first','second'])
                self.assertEqual(library.stat().st_mode & 0o222,0)
                self.assertEqual(len(snapshot(library)),2)
                with self.assertRaises(AssertionError):build(library,dict(first=a),{})
            finally:
                library.chmod(0o755)
                for p in library.iterdir():p.chmod(0o644)

    def test_conflicting_same_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=root/'a';b=root/'b';a.mkdir();b.mkdir()
            key='a'*64
            for i,folder in enumerate([a,b]):
                (folder/(key+'.json')).write_text(json.dumps(dict(schema=1,key=key,payload_sha256=str(i),covers=[])))
            with self.assertRaises(AssertionError):build(root/'library',dict(first=a,second=b),{})


if __name__=='__main__':unittest.main()
