#!/usr/bin/env python3
"""Separate real identity cells, serialization aliases and core sharing."""
import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_buffer_core_validation import buffer_area
from audit_buffer_core import core_fingerprint as subtree_fingerprint
from exact_core_graph import exact_core_fingerprint as core_fingerprint

class Accounting(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name)
        self.lib=self.p/'lib.genlib'
        self.lib.write_text('GATE BUF 0.07 Y=A;\nGATE AND2 0.2 Y=A*B;\n')
    def tearDown(self):self.tmp.cleanup()
    def net(self,name,body):
        p=self.p/name;p.write_text('module top(x0,x1,y0,y1);\ninput x0,x1;\noutput y0,y1;\n'+body+'\nendmodule\n');return p
    def test_physical_buffer_and_output_alias_are_distinct(self):
        before=self.net('before.v','  BUF g0( .A (x0), .Y (y0) );\n  BUF g1( .A (y0), .Y (y1) );')
        after=self.net('after.v','  assign y0 = x0;\n  assign y1 = y0;')
        self.assertAlmostEqual(buffer_area(before,self.lib)[0],.07)
        self.assertEqual(buffer_area(after,self.lib)[0],0)
        self.assertEqual(core_fingerprint(before,self.lib),core_fingerprint(after,self.lib))
    def test_core_fingerprint_preserves_physical_sharing(self):
        duplicate=self.net('duplicate.v','  AND2 g0( .A (x0), .B (x1), .Y (y0) );\n  AND2 g1( .A (x0), .B (x1), .Y (y1) );')
        shared=self.net('shared.v','  AND2 g0( .A (x0), .B (x1), .Y (n0) );\n  assign y0 = n0;\n  assign y1 = n0;')
        a,b=core_fingerprint(duplicate,self.lib),core_fingerprint(shared,self.lib)
        self.assertEqual((a['core_cells'],b['core_cells']),(2,1))
        self.assertNotEqual(a['core_dag_sha256'],b['core_dag_sha256'])
    def test_duplicate_gate_fanout_distribution_is_not_lost(self):
        base='  AND2 g0( .A (x0), .B (x1), .Y (a0) );\n  AND2 g1( .A (x0), .B (x1), .Y (a1) );\n  AND2 g2( .A (a0), .B (x0), .Y (y0) );\n'
        one=self.net('one.v',base+'  AND2 g3( .A (a0), .B (x1), .Y (y1) );\n  assign y2 = a1;')
        two=self.net('two.v',base+'  AND2 g3( .A (a1), .B (x1), .Y (y1) );\n  assign y2 = a1;')
        for p in [one,two]:p.write_text(p.read_text().replace('y0,y1','y0,y1,y2'))
        self.assertEqual(subtree_fingerprint(one,self.lib),subtree_fingerprint(two,self.lib))
        self.assertNotEqual(core_fingerprint(one,self.lib),core_fingerprint(two,self.lib))
if __name__=='__main__':unittest.main()
