
from cpp_ast import *
import python_ast as ast
from asp.util import *



class NodeVisitor(ast.NodeVisitor):
    """Unified class for visiting Python and C++ AST nodes, adapted from Python source."""
    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST) or isinstance(value, Generable):
                self.visit(value)


class NodeTransformer(ast.NodeTransformer):
    """Unified class for *transforming* Python and C++ AST nodes, adapted from Python source"""
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



class ASTNodeReplacer(ast.NodeTransformer):
    """Class to replace Python AST nodes."""
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



class ConvertAST(ast.NodeTransformer):
    """Class to convert from Python AST to C++ AST"""
    def visit_Num(self, node):
        return CNumber(node.n)

    def visit_Str(self, node):
        return String(node.s)

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
    def visit_Mod(self, node):
        return "%"

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


    def visit_Pass(self, _):
        return Expression()

    # by default, only do first statement in a module
    def visit_Module(self, node):
        return self.visit(node.body[0])

    def visit_Expr(self, node):
        return self.visit(node.value)

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


    def visit_arguments(self, node):
        """Only return the basic case: everything is void*,  no named args, no default values"""
        return [Pointer(Value("void",self.visit(x))) for x in node.args]

    def visit_Call(self, node):
        """We only handle calls that are casts; everything else (eventually) will be
           translated into callbacks into Python."""
        if isinstance(node.func, ast.Name):
            if node.func.id == "int":
                return TypeCast(Value('int', ''), self.visit(node.args[0]))
            if node.func.id == "abs":
                return Call(CName("abs"), [self.visit(x) for x in node.args])

    def visit_Print(self, node):
        if len(node.values) > 0:
            text = '<< ' + str(self.visit(node.values[0]))
        else:
            text = ''
        for fragment in node.values[1:]:
            text += ' << \" \" << ' + str(self.visit(fragment))
        return Print(text, node.nl)


class LoopUnroller(object):
    class UnrollReplacer(NodeTransformer):
        def __init__(self, loopvar, increment):
            self.loopvar = loopvar
            self.increment = increment
            self.in_new_scope = False
            self.inside_for = False
            super(LoopUnroller.UnrollReplacer, self).__init__()

        def visit_CName(self, node):
            #print "node.name is ", node.name
            if node.name == self.loopvar:
                return BinOp(CName(self.loopvar), "+", CNumber(self.increment))
            else:
                return node

        def visit_Block(self, node):
            #print "visiting Block...."
            if self.inside_for:
                old_scope = self.in_new_scope
                self.in_new_scope = True
                #print "visiting block in ", node
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

    def unroll(self, node, factor):
        """Given a For node, unrolls the loop with a given factor.

        If the number of iterations in the given loop is not a multiple of
        the unroll factor, a 'leftover' loop will be generated to run the
        remaining iterations.

        """

        import copy

        #Number of iterations in the initial loop
        num_iterations = node.end.num - node.initial.num + 1

        #Integer division provides number of iterations
        # in the unrolled loop
        num_unrolls = num_iterations / factor

        #Iterations left over after unrolled loop
        leftover = num_iterations % factor

        #End of unrolled loop, add one to get beginning of leftover loop
        loop_end = CNumber(node.end.num - leftover)
        leftover_begin = CNumber(node.end.num - leftover + 1)

        debug_print("Loop unroller called with ", node.loopvar)
        debug_print("Number of iterations: ", num_iterations)
        debug_print("Number of unrolls: ", num_unrolls)
        debug_print("Leftover iterations: ", leftover)

        new_increment = BinOp(node.increment, "*", CNumber(factor))

        new_block = Block(contents=node.body.contents)
        for x in xrange(1, factor):
            new_extension = copy.deepcopy(node.body)
            new_extension = LoopUnroller.UnrollReplacer(node.loopvar, x).visit(new_extension)
            new_block.extend(new_extension.contents)

        return_block = UnbracedBlock()

        unrolled_for_node = For(
            node.loopvar,
            node.initial,
            loop_end,
            new_increment,
            new_block)

        leftover_for_node = For(
            node.loopvar,
            leftover_begin,
            node.end,
            node.increment,
            node.body)

        return_block.append(unrolled_for_node)

        if leftover != 0:
            return_block.append(leftover_for_node)

        return return_block

