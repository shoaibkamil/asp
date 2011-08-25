import unittest

from asp.codegen.ast_tools import *
from asp.codegen.cpp_ast import *
import asp.codegen.python_ast as python_ast

class NodeVisitorTests(unittest.TestCase):
    def test_for_python_nodes(self):
        class Dummy(NodeVisitor):
            def visit_Name(self, node):
                return False
        p = python_ast.Name("hello", False)
        self.assertFalse(Dummy().visit(p))

    def test_for_cpp_nodes(self):
        class Dummy(NodeVisitor):
            def visit_CName(self, node):
                return False
        c = CName("hello")
        self.assertFalse(Dummy().visit(c))

    def test_for_cpp_children(self):
        class Dummy(NodeVisitor):
            def __init__(self):
                self.worked = False
            def visit_CName(self, _):
                self.worked = True

        c = BinOp(CNumber(1), "+", CName("hello"))
        d = Dummy()
        d.visit(c)
        self.assertTrue(d.worked)


class NodeTransformerTests(unittest.TestCase):

    class Dummy(NodeTransformer):
        def visit_Name(self, _):
            return python_ast.Name("hi", False)
        def visit_CName(self, _):
            return CName("hi")

    def test_for_python_nodes(self):
        p = python_ast.Name("hello", False)
        result = self.Dummy().visit(p)
        self.assertEqual(result.id, "hi")

    def test_for_cpp_nodes(self):
        c = CName("hello")
        result = self.Dummy().visit(c)
        self.assertEqual(result.name, "hi")

    def test_for_cpp_children(self):
        c = BinOp(CNumber(1), "+", CName("hello"))
        result = self.Dummy().visit(c)
        self.assertEqual(result.right.name, "hi")


class LoopUnrollerTests(unittest.TestCase):
    def setUp(self):
        # this is "for(int i=0, i<8; i+=1) { a[i] = i; }"
        self.test_ast = For(
            "i",
            CNumber(0),
            CNumber(7),
            CNumber(1),
            Block(contents=[Assign(Subscript(CName("a"), CName("i")),
                   CName("i"))]))

    def test_unrolling_by_2(self):
        result = LoopUnroller().unroll(self.test_ast, 2)
        print result
        wanted_result ='for(int i=0;(i<=(7-1));i=(i+(1*2)))\n {\n a[i]=i;\n a[(i+1)]=(i+1);\n}'
        
        self.assertEqual(str(result).replace(' ',''), str(wanted_result).replace(' ', ''))




    def test_unrolling_by_4(self):
        result = LoopUnroller().unroll(self.test_ast, 4)
        print result
        wanted_result = 'for(inti=0;(i<=(7-3));i=(i+(1*4)))\n{\na[i]=i;\na[(i+1)]=(i+1);\na[(i+2)]=(i+2);\na[(i+3)]=(i+3);\n}'

        self.assertEqual(str(result).replace(' ',''), str(wanted_result).replace(' ', ''))

    def test_imperfect_unrolling (self):
        result = LoopUnroller().unroll(self.test_ast, 3)
        wanted_result = 'for(inti=0;(i<=(7-2));i=(i+(1*3)))\n{\na[i]=i;\na[(i+1)]=(i+1);\na[(i+2)]=(i+2);\n}\nfor(inti=(((((7-0)+1)/3)*3)+0);(i<=7);i=(i+1))\n{\na[i]=i;\n}'

        print str(result)
        self.assertEqual(str(result).replace(' ',''), str(wanted_result).replace(' ', ''))

    def test_with_1_index(self):
        test_ast = For("i",
                       CNumber(1),
                       CNumber(9),
                       CNumber(1),
                       Block(contents=[Assign(Subscript(CName("a"), CName("i")), CName("i"))]))
        result = LoopUnroller().unroll(test_ast, 2)
        print result

class LoopBlockerTests(unittest.TestCase):
    def test_basic_blocking(self):
        # this is "for(int i=0, i<=7; i+=1) { a[i] = i; }"
        test_ast = For(
            "i",
            CNumber(0),
            CNumber(7),
            CNumber(1),
            Block(contents=[Assign(Subscript(CName("a"), CName("i")),
                   CName("i"))]))

        wanted_output = "for(intii=0;(ii<=7);ii=(ii+(1*2)))\n{\nfor(inti=ii;(i<=min((ii+1),7));i=(i+1))\n{\na[i]=i;\n}\n}"
        output = str(LoopBlocker().loop_block(test_ast, 2)).replace(' ', '')
        self.assertEqual(output, wanted_output)


class LoopSwitcherTests(unittest.TestCase):
    def test_basic_switching(self):
        test_ast = For("i",
                       CNumber(0),
                       CNumber(7),
                       CNumber(1),
                       Block(contents=[For("j",
                                       CNumber(0),
                                       CNumber(3),
                                       CNumber(1),
                                       Block(contents=[Assign(CName("v"), CName("i"))]))]))
        wanted_output = "for(intj=0;(j<=3);j=(j+1))\n{\nfor(inti=0;(i<=7);i=(i+1))\n{\nv=i;\n}\n}"
        output = str(LoopSwitcher().switch(test_ast, 0, 1)).replace(' ','')
        self.assertEqual(output, wanted_output)

    def test_more_switching(self):
        test_ast = For("i",
                       CNumber(0),
                       CNumber(7),
                       CNumber(1),
                       Block(contents=[For("j",
                                       CNumber(0),
                                       CNumber(3),
                                       CNumber(1),
                                       Block(contents=[For("k",
                                                           CNumber(0),
                                                           CNumber(4),
                                                           CNumber(1),
                                                           Block(contents=[Assign(CName("v"), CName("i"))]))]))]))
        
        wanted_output = "for(intj=0;(j<=3);j=(j+1))\n{\nfor(inti=0;(i<=7);i=(i+1))\n{\nfor(intk=0;(k<=4);k=(k+1))\n{\nv=i;\n}\n}\n}"
        output = str(LoopSwitcher().switch(test_ast, 0, 1)).replace(' ','')
        self.assertEqual(output, wanted_output)

        test_ast = For("i",
                       CNumber(0),
                       CNumber(7),
                       CNumber(1),
                       Block(contents=[For("j",
                                       CNumber(0),
                                       CNumber(3),
                                       CNumber(1),
                                       Block(contents=[For("k",
                                                           CNumber(0),
                                                           CNumber(4),
                                                           CNumber(1),
                                                           Block(contents=[Assign(CName("v"), CName("i"))]))]))]))

        wanted_output = "for(intk=0;(k<=4);k=(k+1))\n{\nfor(intj=0;(j<=3);j=(j+1))\n{\nfor(inti=0;(i<=7);i=(i+1))\n{\nv=i;\n}\n}\n}"
        output = str(LoopSwitcher().switch(test_ast, 0, 2)).replace(' ','')
        self.assertEqual(output, wanted_output)
        

if __name__ == '__main__':
    unittest.main()
