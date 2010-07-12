import unittest
import ast
from codegen import *

class BasicTests(unittest.TestCase):
    def test_num(self):
        k = ast.parse("34")
        cg = CodeGenerator()
        
        self.assertEqual(cg.visit(k), "34")

    def test_name(self):
        k = ast.parse("abc")
        cg = CodeGenerator()
        self.assertEqual(cg.visit(k), "abc")

    def test_basic_binops(self):
        k = ast.parse("34+56")
        cg = CodeGenerator()
        self.assertEqual(cg.visit(k), "34+56")
        
        k = ast.parse("34 * 10 + x")
        self.assertEqual(cg.visit(k), "34*10+x")

    def test_call(self):
        k = ast.parse("abc()")
        cg = CodeGenerator()
        self.assertEqual(cg.visit(k), "abc()")
        k = ast.parse("abc(hi)")
        self.assertEqual(cg.visit(k), "abc(hi)")
        k = ast.parse("abc(hi, bye)")
        self.assertEqual(cg.visit(k), "abc(hi,bye)")


class MoreTests(unittest.TestCase):
    def test_iterator(self):
        k = ast.parse("""
for x in abc.interior_points():
  pass
""")
        cg = CodeGenerator()
        print cg.visit(k)


if __name__ == '__main__':
    unittest.main()
