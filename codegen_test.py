import unittest
import ast
from codegen import *

class BasicTests(unittest.TestCase):
    def test_num(self):
        k = ast.parse("34")
        cg = CodeGenerator()
        
        self.assertEqual(cg.visit(k), "34")

    def test_basic_binops(self):
        k = ast.parse("34+56")
        cg = CodeGenerator()
        self.assertEqual(cg.visit(k), "34+56")
        
        k = ast.parse("34 * 10 + 4")
        self.assertEqual(cg.visit(k), "34*10+4")

if __name__ == '__main__':
    unittest.main()
