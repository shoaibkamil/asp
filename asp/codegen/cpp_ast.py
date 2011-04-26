import codepy.cgen
from codepy.cgen import *
import xml.etree.ElementTree as ElementTree

# these are additional classes that, along with codepy's classes, let
# programmers express the C code as a real AST (not the hybrid AST/strings/etc
# that codepy implements.

class CNumber(Generable):
    def __init__(self, num):
        self.num = num

    def __str__(self):
        return str(self.num)

    def to_xml(self):
        return ElementTree.Element("CNumber", attrib={"num":str(self.num)})

class CName(Generable):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return str(self.name)

    def to_xml(self):
        return ElementTree.Element("CName", attrib={"name":str(self.name)})

class Expression(Generable):
    def __str__(self):
        return ""

    def generate(self):
        yield str(self) + ';'

class BinOp(Expression):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def __str__(self):
        return "(%s %s %s)" % (self.left, self.op, self.right)

    def to_xml(self):
        node = ElementTree.Element("BinOp", attrib={"op":str(self.op)})
        left = ElementTree.SubElement(node, "left")
        left.append(self.left.to_xml())
        right = ElementTree.SubElement(node, "right")
        right.append(self.right.to_xml())
        return node

class UnaryOp(Expression):
    def __init__(self, op, operand):
        self.op = op
        self.operand = operand

    def __str__(self):
        return "(%s(%s))" % (self.op, self.operand)

    def to_xml(self):
        node = ElementTree.Element("UnaryOp", attrib={"op":str(self.op)})
        operand = ElementTree.SubElement(node, "operand")
        operand.append(self.operand.to_xml())
        return node

class Subscript(Expression):
    def __init__(self, value, index):
        self.value = value
        self.index = index

    def __str__(self):
        return "%s[%s]" % (self.value, self.index)

    def to_xml(self):
        node = ElementTree.Element("Subscript")
        ElementTree.SubElement(node, "value").append(self.value.to_xml())
        ElementTree.SubElement(node, "index").append(self.index.to_xml())
        return node

class Call(Expression):
    def __init__(self, func, args):
        self.func = func
        self.args = args

    def __str__(self):
        return "%s(%s)" % (self.func, ", ".join(map(str, self.args)))

    def to_xml(self):
        node = ElementTree.Element("Call", attrib={"func":str(self.func)})
        args = ElementTree.SubElement(node, "args")
        for x in self.args:
            args.append(x.to_xml())
        return node


class PostfixUnaryOp(Expression):
    def __init__(self, operand, op):
        self.operand = operand
        self.op = op

    def __str__(self):
        return "((%s)%s)" % (self.operand, self.op)

    def to_xml(self):
        node = ElementTree.Element("PostfixUnaryOp", attrib={"op":str(self.op)})
        operand = ElementTree.SubElement(node, "operand")
        operand.append(self.operand.to_xml())
        return node


class ConditionalExpr(Expression):
    def __init__(self, test, body, orelse):
        self.test = test
        self.body = body
        self.orelse = orelse

    def __str__(self):
        return "(%s ? %s : %s)" % (self.test, self.body, self.orelse)

    def to_xml(self):
        node = ElementTree.Element("ConditionalExpr")
        ElementTree.SubElement(node, "test").append(self.test.to_xml())
        ElementTree.SubElement(node, "body").append(self.body.to_xml())
        ElementTree.SubElement(node, "orelse").append(self.orelse.to_xml())
        return node

class TypeCast(Expression):
    # "type" should be a declaration with an empty variable name
    # e.g. TypeCast(Pointer(Value('int', '')), ...)

    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __str__(self):
        return "((%s)%s)" % (self.type.inline(), self.value)


class For(codepy.cgen.For):
    def to_xml(self):
        node = ElementTree.Element("For")
        if (not isinstance(self.start, str)):
            ElementTree.SubElement(node, "start").append(self.start.to_xml())
        else:
            ElementTree.SubElement(node,"start").text = self.start
            
        if (not isinstance(self.condition, str)):
            ElementTree.SubElement(node, "condition").append(self.condition.to_xml())
        else:
            ElementTree.SubElement(node, "condition").text = self.condition

        if (not isinstance(self.update, str)):
            ElementTree.SubElement(node, "update").append(self.update.to_xml())
        else:
            ElementTree.SubElement(node, "update").text = self.update
            
        ElementTree.SubElement(node, "body").append(self.body.to_xml())
        return node

class FunctionBody(codepy.cgen.FunctionBody):
    def to_xml(self):
        node = ElementTree.Element("FunctionBody")
        ElementTree.SubElement(node, "fdecl").append(self.fdecl.to_xml())
        ElementTree.SubElement(node, "body").append(self.body.to_xml())
        return node

class FunctionDeclaration(codepy.cgen.FunctionDeclaration):
    def to_xml(self):
        node = ElementTree.Element("FunctionDeclaration")
        ElementTree.SubElement(node, "subdecl").append(self.subdecl.to_xml())
        arg_decls = ElementTree.SubElement(node, "arg_decls")
        for x in self.arg_decls:
            arg_decls.append(x.to_xml())
        return node

class Value(codepy.cgen.Value):
    def to_xml(self):
        return ElementTree.Element("Value", attrib={"typename":self.typename, "name":self.name})

class Pointer(codepy.cgen.Pointer):
    def to_xml(self):
        node = ElementTree.Element("Pointer")
        ElementTree.SubElement(node, "subdecl").append(self.subdecl.to_xml())
        return node

class Block(codepy.cgen.Block):
    def to_xml(self):
        node = ElementTree.Element("Block")
        for x in self.contents:
            node.append(x.to_xml())
        return node

class Define(codepy.cgen.Define):
    def to_xml(self):
        return ElementTree.Element("Define", attrib={"symbol":self.symbol, "value":self.value})

class Statement(codepy.cgen.Statement):
    def to_xml(self):
        node = ElementTree.Element("Statement")
        node.text = self.text
        return node

class Assign(codepy.cgen.Assign):
    def to_xml(self):
        node = ElementTree.Element("Assign")
        ElementTree.SubElement(node, "lvalue").append(self.lvalue.to_xml())
        ElementTree.SubElement(node, "rvalue").append(self.rvalue.to_xml())
        return node
        
