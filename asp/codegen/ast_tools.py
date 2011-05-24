
from cpp_ast import *
import python_ast as ast
from asp.util import *


# unified class for visiting python and c++ AST nodes
class NodeVisitor(ast.NodeVisitor):
    # adapted from Python source
    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST) or isinstance(value, Generable):
                self.visit(value)
	

# unified class for *transforming* python and c++ AST nodes
class NodeTransformer(ast.NodeTransformer):
    # adapted from Python source
     def generic_visit(self, node):
	        for field, old_value in ast.iter_fields(node):
	            old_value = getattr(node, field, None)
	            if isinstance(old_value, list):
	                new_values = []
	                for value in old_value:
	                    if isinstance(value, ast.AST) or isinstance(value, Generable):
	                        value = self.visit(value)
	                        if value is None:
	                            continue
	                        elif not (isinstance(value, ast.AST) or isinstance(value, Generable)):
	                            new_values.extend(value)
	                            continue
	                    new_values.append(value)
	                old_value[:] = new_values
	            elif isinstance(old_value, ast.AST) or isinstance(old_value, Generable):
	                new_node = self.visit(old_value)
	                if new_node is None:
	                    delattr(node, field)
	                else:
	                    setattr(node, field, new_node)
	        return node


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
        

class LoopUnroller(object):
    class UnrollReplacer(NodeTransformer):
        def __init__(self, loopvar, increment):
            self.loopvar = loopvar
            self.increment = increment
            self.in_new_scope = False
            self.inside_for = False
            super(LoopUnroller.UnrollReplacer, self).__init__()

        def visit_CName(self, node):
#            print "node.name is ", node.name
            if node.name == self.loopvar:
                return BinOp(CName(self.loopvar), "+", CNumber(self.increment))
            else:
                return node
        
        def visit_Block(self, node):
#            print "visiting Block...."
            if self.inside_for:
                old_scope = self.in_new_scope
                self.in_new_scope = True
#            print "visiting block in ", node
                contents = [self.visit(x) for x in node.contents]
                retnode = Block(contents=[x for x in contents if x != None])
                self.in_new_scope = old_scope
            else:
                self.inside_for = True
                contents = [self.visit(x) for x in node.contents]
                retnode = Block(contents=[x for x in contents if x != None])

            return retnode

        # assigns take care of stuff like "int blah = foo"
        def visit_Value(self, node):
            if not self.in_new_scope:
                return None
            else:
                return node
            
        def visit_Pointer(self, node):
            if not self.in_new_scope:
                return None
            else:
                return node

        # ignore typecast declarators
        def visit_TypeCast(self, node):
            return TypeCast(node.tp, self.visit(node.value))

        # make lvalue not a declaration
        def visit_Assign(self, node):
            if not self.in_new_scope:
                if isinstance(node.lvalue, NestedDeclarator):
                    tp, new_lvalue = node.lvalue.subdecl.get_decl_pair()
                    rvalue = self.visit(node.rvalue)
                    return Assign(CName(new_lvalue), rvalue)
                if isinstance(node.lvalue, Declarator):
                    tp, new_lvalue = node.lvalue.get_decl_pair()
                    rvalue = self.visit(node.rvalue)
                    return Assign(CName(new_lvalue), rvalue)
            return Assign(self.visit(node.lvalue), self.visit(node.rvalue))

    def unroll(self, node, factor, perfect=True):
        import copy

#        print "Called with %s", node.loopvar

        new_increment = BinOp(node.increment, "*", CNumber(factor))

        new_block = Block(contents=node.body.contents)
        for x in xrange(1, factor):
            new_extension = copy.deepcopy(node.body)
            new_extension = LoopUnroller.UnrollReplacer(node.loopvar, x).visit(new_extension)
            new_block.extend(new_extension.contents)

        return For(node.loopvar, node.initial, node.end, new_increment, new_block)
        

