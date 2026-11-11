"""Cargo package ordering cannot hide a dependency change in recovery."""
import unittest
from recover_completion import normalized_lock
class LockSemanticTests(unittest.TestCase):
 def test_package_order_is_not_dependency_content(self):
  header=b'version = 3\n';a=b'[[package]]\nname="a"\nversion="1.0.0"\nchecksum="aaa"\n';b=b'[[package]]\nname="b"\nversion="2.0.0"\nchecksum="bbb"\n'
  self.assertEqual(normalized_lock(header+a+b),normalized_lock(header+b+a))
  self.assertNotEqual(normalized_lock(header+a+b),normalized_lock(header+a+b.replace(b'2.0.0',b'2.0.1')))
  self.assertNotEqual(normalized_lock(header+a+b),normalized_lock(header+a+b.replace(b'bbb',b'changed')))
if __name__=='__main__':unittest.main()
