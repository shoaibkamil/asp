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

class MoreTests(unittest.TestCase):
    def test_funcdef(self):
        k = ast.parse("""
for x in abc.neighbors():
  pass
""")
        cg = CodeGenerator()
        print cg.visit(k)


if __name__ == '__main__':
    unittest.main()
