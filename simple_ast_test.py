from simple_ast import *
import ast
import unittest

class BasicNodeTests(unittest.TestCase):
    def test_BinExp(self):
        abc = ast.parse("3+4")
        out = convert_ast(abc)
        self.assertEqual(out.left.value, 3)
                         
        


if __name__ == '__main__':
    unittest.main()


# abc = ast.parse("1+2")
# goober = ast_tools.CodeGenerator()
# goober2 = ast_tools.ASTPrettyPrinter()

# print goober2.visit(abc)
# print ast.dump(abc)

# print goober.visit(abc)
