import unittest

from asp.codegen.ast_tools import *

class ReplacerTests(unittest.TestCase):
	def test_num(self):
		a = ast.BinOp(ast.Num(4), ast.Add(), ast.Num(9))
		result = ASTNodeReplacer(ast.Num(4), ast.Num(5)).visit(a)
		self.assertEqual(a.left.n, 5)

	def test_Name(self):
		a = ast.BinOp(ast.Num(4), ast.Add(), ast.Name("variable", None))
		result = ASTNodeReplacer(ast.Name("variable", None), ast.Name("my_variable", None)).visit(a)
		self.assertEqual(a.right.id, "my_variable")

		

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

    def test_Assign(self):
        a = ast.Assign([ast.Name("hello", None)], ast.Num(4))
        b = ConvertAST().visit(a)
        self.assertEqual(str(b), "hello = 4;")

    def test_simple_FunctionDef(self):
        a = ast.FunctionDef("hello",
                            ast.arguments([], None, None, []),
                            [ast.BinOp(ast.Num(10), ast.Add(), ast.Num(20))], [])
        b = ConvertAST().visit(a)
        self.assertEqual(str(b), "void hello()\n{\n  (10 + 20);\n}")
    def test_FunctionDef_with_arguments(self):
        a = ast.FunctionDef("hello",
                            ast.arguments([ast.Name("world", None)], None, None, []),
                            [ast.BinOp(ast.Num(10), ast.Add(), ast.Num(20))], [])
        b = ConvertAST().visit(a)
        self.assertEqual(str(b), "void hello(void *world)\n{\n  (10 + 20);\n}")

if __name__ == '__main__':
    unittest.main()
