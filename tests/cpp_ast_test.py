import unittest
from asp.codegen.cpp_ast import *
import xml.etree.ElementTree as ElementTree

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

    def test_For(self):
        t = For(CName("foo"), CName("bar"), CName("baz"), CName("bin"))
        self.assertEqual(ElementTree.tostring(t.to_xml()),
                         "<For><start><CName name=\"foo\" /></start><condition><CName name=\"bar\" /></condition>"+
                         "<update><CName name=\"baz\" /></update><body><CName name=\"bin\" /></body></For>")
        
if __name__ == '__main__':
    unittest.main()
