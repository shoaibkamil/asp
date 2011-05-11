import unittest
from asp.codegen.cpp_ast import *
import xml.etree.ElementTree as ElementTree


class ForTests(unittest.TestCase):
    def test_init(self):
        # For(loopvar, initial, end, increment)
        f = For("i", CNumber(0), CNumber(10), CNumber(1), Block())
        self.assertEqual(str(f), "for (int i = 0; (i <= 10); i = (i + 1))\n{\n}")

class XMLTests(unittest.TestCase):
    def test_BinOp(self):
        t = BinOp(CNumber(5), '+', CName("foo"))
        self.assertEqual(ElementTree.tostring(t.to_xml()),
                         "<BinOp op=\"+\"><left><CNumber num=\"5\" /></left>"+
                         "<right><CName name=\"foo\" /></right></BinOp>")

    def test_UnaryOp(self):
        t = UnaryOp("++", CNumber(5))
        self.assertEqual(ElementTree.tostring(t.to_xml()),
                         "<UnaryOp op=\"++\"><operand><CNumber num=\"5\" /></operand></UnaryOp>")

    def test_Subscript(self):
        t = Subscript(CName("foo"), CNumber("5"))
        self.assertEqual(ElementTree.tostring(t.to_xml()),
                         "<Subscript><value><CName name=\"foo\" /></value>"+
                         "<index><CNumber num=\"5\" /></index></Subscript>")

    def test_Call(self):
        t = Call("foo", [CName("arg")])
        self.assertEqual(ElementTree.tostring(t.to_xml()),
                         "<Call func=\"foo\"><args><CName name=\"arg\" /></args></Call>")

    def test_PostfixUnaryOp(self):
        t = PostfixUnaryOp(CName("foo"), "--");
        self.assertEqual(ElementTree.tostring(t.to_xml()),
                         "<PostfixUnaryOp op=\"--\"><operand><CName name=\"foo\" /></operand></PostfixUnaryOp>")

    def test_ConditionalExpr(self):
        t = ConditionalExpr(CName("foo"), CName("bar"), CName("baz"))
        self.assertEqual(ElementTree.tostring(t.to_xml()),
                         "<ConditionalExpr><test><CName name=\"foo\" /></test><body><CName name=\"bar\" /></body><orelse><CName name=\"baz\" /></orelse></ConditionalExpr>")

    def test_RawFor(self):
        t = RawFor(CName("foo"), CName("bar"), CName("baz"), CName("bin"))
        self.assertEqual(ElementTree.tostring(t.to_xml()),
                         "<For><start><CName name=\"foo\" /></start><condition><CName name=\"bar\" /></condition>"+
                         "<update><CName name=\"baz\" /></update><body><CName name=\"bin\" /></body></For>")

    def test_FunctionBody(self):
        t = FunctionBody(CName("foo"), CName("bar"))
        self.assertEqual(ElementTree.tostring(t.to_xml()),
                         "<FunctionBody><fdecl><CName name=\"foo\" /></fdecl><body><CName name=\"bar\" /></body></FunctionBody>")

    def test_FunctionDeclaration(self):
        t = FunctionDeclaration(CName("foo"), [CName("bar")])
        self.assertEqual(ElementTree.tostring(t.to_xml()),
                         "<FunctionDeclaration><subdecl><CName name=\"foo\" /></subdecl><arg_decls><CName name=\"bar\" /></arg_decls></FunctionDeclaration>")

    def test_Value(self):
        t = Value("int", "foo")
        self.assertEqual(ElementTree.tostring(t.to_xml()),
                         "<Value name=\"foo\" typename=\"int\" />")

    def test_Pointer(self):
        t = Pointer(Value("int", "foo"))
        self.assertEqual(ElementTree.tostring(t.to_xml()),
                         "<Pointer><subdecl><Value name=\"foo\" typename=\"int\" /></subdecl></Pointer>")
                                              

    def test_Block(self):
        t = Block(contents=[CName("foo")])
        self.assertEqual(ElementTree.tostring(t.to_xml()),
                         "<Block><CName name=\"foo\" /></Block>")

    def test_Define(self):
        t = Define("foo", "defined_to")
        self.assertEqual(ElementTree.tostring(t.to_xml()),
                         "<Define symbol=\"foo\" value=\"defined_to\" />")

    def test_Statement(self):
        t = Statement("foo")
        self.assertEqual(ElementTree.tostring(t.to_xml()),
                         "<Statement>foo</Statement>")

    def test_Assign(self):
        t = Assign(CName("foo"), CNumber(5))
        self.assertEqual(ElementTree.tostring(t.to_xml()),
                         "<Assign><lvalue><CName name=\"foo\" /></lvalue><rvalue><CNumber num=\"5\" /></rvalue></Assign>")

    def test_whole_ast(self):
        """ A test using the pickled whole AST from one of the stencil kernel test cases."""
        import pickle
        t = pickle.load(open("tests/pickled_ast"))
        t.to_xml()
        
if __name__ == '__main__':
    unittest.main()
