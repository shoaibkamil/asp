import codepy.cgen
from codepy.cgen import *
import xml.etree.ElementTree as ElementTree

# these are additional classes that, along with codepy's classes, let
# programmers express the C code as a real AST (not the hybrid AST/strings/etc
# that codepy implements.

#TODO: add all of CodePy's classes we want to support

class CNumber(Generable):
    def __init__(self, num):
        self.num = num
        self._fields = []

    def __str__(self):
        return str(self.num)

    def to_xml(self):
        return ElementTree.Element("CNumber", attrib={"num":str(self.num)})

class CName(Generable):
    def __init__(self, name):
        self.name = name
        self._fields = []

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
        self._fields = ['left', 'right']

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
        self._fields = ['operand']

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
        self._fields = ['value', 'index']

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
        self._fields = ['func', 'args']

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
        self._fields = ['op', 'operand']

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
        self._fields = ['test', 'body', 'orelse']

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
        self._fields = ['type', 'value']

    def __str__(self):
        return "((%s)%s)" % (self.type.inline(), self.value)

class ForInitializer(codepy.cgen.Initializer):
    def __str__(self):
        return super(ForInitializer, self).__str__()[0:-1]


class RawFor(codepy.cgen.For):
    def __init__(self, start, condition, update, body):
        super(RawFor, self).__init__(start, condition, update, body)
        self._fields = ['start', 'condition', 'update', 'body']

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

class For(RawFor):
    #TODO: setting initial,end,etc should update the field in the shadow
    #TODO: should loopvar be a string or a CName?
    def __init__(self, loopvar, initial, end, increment, body):
        self.loopvar = loopvar
        self.initial = initial
        self.end = end
        self.increment = increment
        self._fields = ['start', 'condition', 'update', 'body']
        super(For, self).__init__(
            ForInitializer(Value("int", self.loopvar), self.initial),
            BinOp(CName(self.loopvar), "<=", self.end),
            Assign(CName(self.loopvar), BinOp(CName(self.loopvar), "+", increment)),
            body)

    def intro_line(self):
        return "for (%s; %s; %s)" % (self.start, self.condition, str(self.update)[0:-1])


class FunctionBody(codepy.cgen.FunctionBody):
    def __init__(self, fdecl, body):
        super(FunctionBody, self).__init__(fdecl, body)
        self._fields = ['fdecl', 'body']
        
    def to_xml(self):
        node = ElementTree.Element("FunctionBody")
        ElementTree.SubElement(node, "fdecl").append(self.fdecl.to_xml())
        ElementTree.SubElement(node, "body").append(self.body.to_xml())
        return node

class FunctionDeclaration(codepy.cgen.FunctionDeclaration):
    def __init__(self, subdecl, arg_decls):
        super(FunctionDeclaration, self).__init__(subdecl, arg_decls)
        self._fields = ['subdecl', 'arg_decls']

    def to_xml(self):
        node = ElementTree.Element("FunctionDeclaration")
        ElementTree.SubElement(node, "subdecl").append(self.subdecl.to_xml())
        arg_decls = ElementTree.SubElement(node, "arg_decls")
        for x in self.arg_decls:
            arg_decls.append(x.to_xml())
        return node

class Value(codepy.cgen.Value):
    def __init__(self, typename, name):
        super(Value, self).__init__(typename, name)
        self._fields = []
        
    def to_xml(self):
        return ElementTree.Element("Value", attrib={"typename":self.typename, "name":self.name})

class Pointer(codepy.cgen.Pointer):
    def __init__(self, subdecl):
        super(Pointer, self).__init__(subdecl)
        self._fields = ['subdecl']
        
    def to_xml(self):
        node = ElementTree.Element("Pointer")
        ElementTree.SubElement(node, "subdecl").append(self.subdecl.to_xml())
        return node

class Block(codepy.cgen.Block):
    def __init__(self, contents=[]):
        super(Block, self).__init__(contents)
        self._fields = ['contents']
        
    def to_xml(self):
        node = ElementTree.Element("Block")
        for x in self.contents:
            node.append(x.to_xml())
        return node

class Define(codepy.cgen.Define):
    def __init__(self, symbol, value):
        super(Define, self).__init__(symbol, value)
        self._fields = ['symbol', 'value']
        
    def to_xml(self):
        return ElementTree.Element("Define", attrib={"symbol":self.symbol, "value":self.value})

class Statement(codepy.cgen.Statement):
    def __init__(self, text):
        super(Statement, self).__init__(text)
        self._fields = []
        
    def to_xml(self):
        node = ElementTree.Element("Statement")
        node.text = self.text
        return node

class Assign(codepy.cgen.Assign):
    def __init__(self, lvalue, rvalue):
        super(Assign, self).__init__(lvalue, rvalue)
        self._fields = ['lvalue', 'rvalue']
        
    def to_xml(self):
        node = ElementTree.Element("Assign")
        ElementTree.SubElement(node, "lvalue").append(self.lvalue.to_xml())
        ElementTree.SubElement(node, "rvalue").append(self.rvalue.to_xml())
        return node
        
