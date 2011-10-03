
from cpp_ast import *
import cpp_ast
import python_ast as ast
import python_ast
from asp.util import *

def is_cpp_node(x):
    return isinstance(x, Generable)    

class NodeVisitorCustomNodes(ast.NodeVisitor):
    # Based on NodeTransformer.generic_visit(), but visits all sub-nodes
    # matching is_node(), not just those derived from ast.AST. By default
    # behaves just like ast.NodeTransformer, but is_node() can be overridden.
    def generic_visit(self, node):
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if self.is_node(item):
                        self.visit(item)
            elif self.is_node(value):
                self.visit(value)

    def is_node(self, x):
        return isinstance(x, ast.AST)

class NodeVisitor(NodeVisitorCustomNodes):
    def is_node(self, x):
        return isinstance(x, ast.AST) or is_cpp_node(x)

class NodeTransformerCustomNodes(ast.NodeTransformer):
    # Based on NodeTransformer.generic_visit(), but visits all sub-nodes
    # matching is_node(), not just those derived from ast.AST. By default
    # behaves just like ast.NodeTransformer, but is_node() can be overridden.
    def generic_visit(self, node):
        for field in node._fields:
            old_value = getattr(node, field, None)
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if self.is_node(value):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not self.is_node(value):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif self.is_node(old_value):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node

    def is_node(self, x):
        return isinstance(x, ast.AST)

class NodeTransformerCustomNodesExtended(NodeTransformerCustomNodes):
    """Extended version of NodeTransformerCustomNodes that also tracks line numbers"""
    def visit(self, node):
        result = super(NodeTransformerCustomNodesExtended, self).visit(node)
        return self.transfer_lineno(node, result)

    def transfer_lineno(self, node_from, node_to):
        if hasattr(node_from, 'lineno') and hasattr(node_to, 'lineno'):
            node_to.lineno = node_from.lineno
        if hasattr(node_from, 'col_offset') and hasattr(node_to, 'col_offset'):
            node_to.col_offset = node_from.col_offset
        return node_to

class NodeTransformer(NodeTransformerCustomNodesExtended):
    """Unified class for *transforming* Python and C++ AST nodes"""
    def is_node(self, x):
        return isinstance(x, ast.AST) or is_cpp_node(x)

class ASTNodeReplacer(NodeTransformer):
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
                    break

        if eql:
            import copy
            debug_print( "Found something to replace!!!!" )
            return copy.deepcopy(self.replacement)
        else:
            return self.generic_visit(node)

class ASTNodeReplacerCpp(ASTNodeReplacer):
    def is_node(self, x):
        return is_cpp_node(x)

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
        if isinstance(node, python_ast.Assign):
            return Assign(self.visit(node.targets[0]),
                          self.visit(node.value))
        elif isinstance(node, cpp_ast.Assign):
            return Assign(self.visit(node.lvalue),
                          self.visit(node.rvalue))

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

    def visit_Compare(self, node):
        # only handles 1 thing on right side for now (1st op and comparator)
        # also currently not handling: Is, IsNot, In, NotIn
        ops = {'Eq':'==','NotEq':'!=','Lt':'<','LtE':'<=','Gt':'>','GtE':'>='}
        op = ops[node.ops[0].__class__.__name__]
        return Compare(self.visit(node.left), op, self.visit(node.comparators[0]))

    def visit_If(self, node):
        test = self.visit(node.test)
        body = Block([self.visit(x) for x in node.body])
        if node.orelse == []:
            orelse = None
        else:
            orelse = Block([self.visit(x) for x in node.orelse])
        return IfConv(test, body, orelse)

    def visit_Return(self, node):
        return ReturnStatement(self.visit(node.value))

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

        # we can't precalculate the number of leftover iterations in the case that
        # the number of iterations are not known a priori, so we build an Expression
        # and let the compiler deal with it
        #leftover_begin = BinOp(CNumber(factor),
        #                       "*", 
        #                       BinOp(BinOp(node.end, "+", 1), "/", CNumber(factor)))


        # we begin leftover iterations at factor*( (end-initial+1) / factor ) + initial
        # note that this works due to integer division
        leftover_begin = BinOp(BinOp(BinOp(BinOp(BinOp(node.end, "-", node.initial),
                                                 "+",
                                                    CNumber(1)),
                                           "/",
                                           CNumber(factor)),
                                     "*",
                                     CNumber(factor)),
                               "+",
                               node.initial)

        new_limit = BinOp(node.end, "-", CNumber(factor-1))
        
#        debug_print("Loop unroller called with ", node.loopvar)
#        debug_print("Number of iterations: ", num_iterations)
#        debug_print("Number of unrolls: ", num_unrolls)
#        debug_print("Leftover iterations: ", leftover)

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
            new_limit,
            #node.end,
            new_increment,
            new_block)

        leftover_for_node = For(
            node.loopvar,
            leftover_begin,
            node.end,
            node.increment,
            node.body)


        return_block.append(unrolled_for_node)

        # if we *know* this loop has no leftover iterations, then
        # we return without the leftover loop
        if not (isinstance(node.initial, CNumber) and isinstance(node.end, CNumber) and
           ((node.end.num - node.initial.num + 1) % factor == 0)):
            return_block.append(leftover_for_node)

        return return_block


class LoopBlocker(object):
    def loop_block(self, node, block_size):
        outer_incr_name = CName(node.loopvar + node.loopvar)

        new_inner_for = For(
            node.loopvar,
            outer_incr_name,
            FunctionCall("min", [BinOp(outer_incr_name, 
                                       "+", 
                                       CNumber(block_size-1)), 
                                 node.end]),
            CNumber(1),
            node.body)

        new_outer_for = For(
            node.loopvar + node.loopvar,
            node.initial,
            node.end,
            BinOp(node.increment, "*", CNumber(block_size)),
            Block(contents=[new_inner_for]))
        debug_print(new_outer_for)
        return new_outer_for

class LoopSwitcher(NodeTransformer):
    """
    Class that switches two loops.  The user is responsible for making sure the switching
    is valid (i.e. that the code can still compile/run).  Given two integers i,j this
    class switches the ith and jth loops encountered.
    """

    
    def __init__(self):
        self.current_loop = -1
        self.saved_first_loop = None
        self.saved_second_loop = None
        super(LoopSwitcher, self).__init__()

    def switch(self, tree, i, j):
        """Switch the i'th and j'th loops in tree."""
        self.first_target = min(i,j)
        self.second_target = max(i,j)

        self.original_ast = tree
        
        return self.visit(tree)

    def visit_For(self, node):
        self.current_loop += 1

        debug_print("At loop %d, targets are %d and %d" % (self.current_loop, self.first_target, self.second_target))

        if self.current_loop == self.first_target:
            # save the loop
            debug_print("Saving loop")
            self.saved_first_loop = node
            new_body = self.visit(node.body)
            assert self.second_target < self.current_loop + 1, 'Tried to switch loops %d and %d but only %d loops available' % (self.first_target, self.second_target, self.current_loop + 1)
            # replace with the second loop (which has now been saved)
            return For(self.saved_second_loop.loopvar,
                       self.saved_second_loop.initial,
                       self.saved_second_loop.end,
                       self.saved_second_loop.increment,
                       new_body)


        if self.current_loop == self.second_target:
            # save this
            self.saved_second_loop = node
            # replace this
            debug_print("replacing loop")
            return For(self.saved_first_loop.loopvar,
                       self.saved_first_loop.initial,
                       self.saved_first_loop.end,
                       self.saved_first_loop.increment,
                       node.body)


        return For(node.loopvar,
                   node.initial,
                   node.end,
                   node.increment,
                   self.visit(node.body))
