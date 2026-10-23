import json
import tempfile
import unittest
from pathlib import Path

from fast_original_cec import verify


class FunctionalModelCECTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.models = self.root / "cells.v"
        self.models.write_text("module MYINV(input A,output Y); assign Y=~A; endmodule\n"
                               "module MYAND(input A,input B,output Y); assign Y=A&B; endmodule\n")
        self.golden = self.root / "original.aig"
        self.golden.write_bytes(b"aig 3 2 0 1 1\n6\n" + bytes([2, 1]) +
                                b"i0 a[0]\ni1 b\no0 y[0]\nc\nfixture\n")
        self.doc = dict(modules=dict(mapped=dict(
            ports={"b": dict(direction="input", bits=[3]),
                   "y[0]": dict(direction="output", bits=[5]),
                   "a[0]": dict(direction="input", bits=[2])},
            cells={"inv": dict(type="MYINV", connections=dict(A=[2], Y=[4])),
                   "and": dict(type="MYAND", connections=dict(A=[4], B=[3], Y=[5]))})))

    def tearDown(self):
        self.tmp.cleanup()

    def check(self):
        path = self.root / "mapped.json"
        path.write_text(json.dumps(self.doc))
        return verify(self.models, self.golden, path, self.root / "proof")

    def test_indexed_names_and_shuffled_port_dictionary_pass(self):
        self.assertTrue(self.check())

    def test_swapping_named_input_connections_fails(self):
        ports = self.doc["modules"]["mapped"]["ports"]
        ports["a[0]"]["bits"], ports["b"]["bits"] = [3], [2]
        self.assertFalse(self.check())

    def test_wrong_output_phase_fails(self):
        self.doc["modules"]["mapped"]["ports"]["y[0]"]["bits"] = [4]
        self.assertFalse(self.check())

    def test_missing_original_port_is_rejected(self):
        self.doc["modules"]["mapped"]["ports"]["wrong"] = self.doc["modules"]["mapped"]["ports"].pop("b")
        with self.assertRaises(ValueError):
            self.check()

    def test_constant_output_passes(self):
        self.golden.write_bytes(b"aig 0 0 0 1 0\n1\no0 y[0]\nc\nconstant\n")
        self.doc = dict(modules=dict(mapped=dict(ports={"y[0]": dict(direction="output", bits=["1"])}, cells={})))
        self.assertTrue(self.check())


if __name__ == "__main__":
    unittest.main()
