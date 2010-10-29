
from codepy.cgen import *
import python_ast as ast

# class to replace python AST nodes
class ASTNodeReplacer(ast.NodeTransformer):
	def __init__(self, original, replacement):
		self.original = original
		self.replacement = replacement

	def visit(self, node):
		eql = False
		if node.__class__ == self.original.__class__:
			eql = True
			for (field, value) in ast.iter_fields(self.original):
				if field != 'ctx' and node.__getattribute__(field) != value:
					print str(node.__getattribute__(field)) + " != " + str(value)
					eql = False
			
		if eql:
			import copy
			print "Found something to replace!!!!"
			return copy.deepcopy(self.replacement)
		else:
			return self.generic_visit(node)
	

# class to convert from python AST to C++ AST
class ConvertAST(ast.NodeTransformer):
    def visit_Num(self, node):
        return CNumber(node.n)

    def visit_Name(self, node):
        return CName(node.id)

    def visit_BinOp(self, node):
        return BinOp(self.visit(node.left),
                self.visit(node.op),
                self.visit(node.right))

    def visit_Add(self, node):
        return "+"
    def visit_Sub(self, node):
        return "-"
    def visit_Mult(self, node):
        return "*"
    def visit_Div(self, node):
        return "/"

    def visit_UnaryOp(self, node):
        return UnaryOp(self.visit(node.op),
                        self.visit(node.operand))

    def visit_Invert(self, node):
        return "-"
    def visit_USub(self, node):
        return "-"
    def visit_UAdd(self, node):
        return "+"
    def visit_Not(self, node):
        return "!"

    def visit_Subscript(self, node):
        return Subscript(self.visit(node.value),
                self.visit(node.slice))

    def visit_Index(self, node):
        return self.visit(node.value)

    
    def visit_Pass(self, node):
        return Expression()
    
    # by default, only do first statement in a module
    def visit_Module(self, node):
        return self.visit(node.body[0])

    def visit_Expr(self, node):
        return Expression(self.visit(node.value))

    # only single targets supported
    def visit_Assign(self, node):
        return Assign(self.visit(node.targets[0]),
                self.visit(node.value))

    def visit_FunctionDef(self, node):
        print("In FunctionDef:")
        print ast.dump(node)
        print("----")
        return FunctionBody(FunctionDeclaration(Value("void",
                                                      node.name),
                                                self.visit(node.args)),
                            Block([self.visit(x) for x in node.body]))

    # only do the basic case: everything is void*,  no named args, no default values
    def visit_arguments(self, node):
        return [Pointer(Value("void",self.visit(x))) for x in node.args]
        

# classes to express everything in C++ AST

class CNumber(Generable):
    def __init__(self, num):
        self.num = num

    def __str__(self):
        return str(self.num)

class CName(Generable):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return str(self.name)


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

class UnaryOp(Expression):
    def __init__(self, op, operand):
        self.op = op
        self.operand = operand

    def __str__(self):
        return "(%s(%s))" % (self.op, self.operand)

class Subscript(Expression):
    def __init__(self, value, index):
        self.value = value
        self.index = index

    def __str__(self):
        return "%s[%s]" % (self.value, self.index)

class Call(Expression):
    def __init__(self, func, args):
        self.func = func
        self.args = args

    def __str__(self):
        return "%s(%s)" % (self.func, ", ".join(map(str, self.args)))

# expression types not used in this example:

class PostfixUnaryOp(Expression):
    def __init__(self, operand, op):
        self.operand = operand
        self.op = op

    def __str__(self):
        return "((%s)%s)" % (self.operand, self.op)

class ConditionalExpr(Expression):
    def __init__(self, test, body, orelse):
        self.test = test
        self.body = body
        self.orelse = orelse

    def __str__(self):
        return "(%s ? %s : %s)" % (self.test, self.body, self.orelse)

class TypeCast(Expression):
    # "type" should be a declaration with an empty variable name
    # e.g. TypeCast(Pointer(Value('int', '')), ...)

    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __str__(self):
        return "((%s)%s)" % (self.type.inline(), self.value)

