import unittest

from codegen import *

class ConversionTests(unittest.TestCase):
    def test_num(self):
        a = ast.Num(4)
        b = ConvertAST().visit(a)
        self.assertEqual(str(b), "4")
    def test_Name(self):
        a = ast.Name("hello", None)
        b = ConvertAST().visit(a)
        self.assertEqual(str(b), "hello")
    def test_BinOp(self):
        a = ast.BinOp(ast.Num(4), ast.Add(), ast.Num(9))
        b = ConvertAST().visit(a)
        self.assertEqual(str(b), "(4 + 9)")
    def test_UnaryOp(self):
        a = ast.UnaryOp(ast.USub(), ast.Name("goober", None))
        b = ConvertAST().visit(a)
        self.assertEqual(str(b), "(-(goober))")
    def test_Subscript(self):
        a = ast.Subscript(ast.Name("hello", None),
                        ast.Index(ast.Num(4)),
                        None)
        b = ConvertAST().visit(a)
        self.assertEqual(str(b), "hello[4]")

if __name__ == '__main__':
    unittest.main()
