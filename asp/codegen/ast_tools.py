
from cpp_ast import *
import python_ast as ast
from asp.util import *

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
					debug_print( str(node.__getattribute__(field)) + " != " + str(value) )
					eql = False
			
		if eql:
			import copy
			debug_print( "Found something to replace!!!!" )
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
        debug_print("In FunctionDef:")
        debug_print(ast.dump(node))
        debug_print("----")
        return FunctionBody(FunctionDeclaration(Value("void",
                                                      node.name),
                                                self.visit(node.args)),
                            Block([self.visit(x) for x in node.body]))

    # only do the basic case: everything is void*,  no named args, no default values
    def visit_arguments(self, node):
        return [Pointer(Value("void",self.visit(x))) for x in node.args]
        


