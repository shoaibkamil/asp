import cpp_ast as cpp
import python_ast as ast
import scala_ast as scala 
import inspect

try:
    from asp.util import *
except Exception,e:
    pass    

def is_python_node(x):
    return isinstance(x, ast.AST)    

def is_cpp_node(x):
    return isinstance(x, cpp.Generable)    

def is_scala_node(x):
    return isinstance(x, scala.Generable)

def parse_method(method):
    src = inspect.getsource(method)
    return ast.parse(src.lstrip())

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
        return isinstance(x, ast.AST) or is_cpp_node(x) or is_scala_node(x)

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
        return isinstance(x, ast.AST) or is_cpp_node(x) or is_scala_node(x)

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
        return cpp.CNumber(node.n)

    def visit_Str(self, node):
        return cpp.String(node.s)

    def visit_Name(self, node):
        return cpp.CName(node.id)

    def visit_BinOp(self, node):
        return cpp.BinOp(self.visit(node.left),
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
        return cpp.UnaryOp(self.visit(node.op),
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
        return cpp.Subscript(self.visit(node.value),
                             self.visit(node.slice))

    def visit_Index(self, node):
        return self.visit(node.value)

    def visit_Pass(self, _):
        return cpp.Expression()

    # by default, only do first statement in a module
    def visit_Module(self, node):
        return self.visit(node.body[0])

    def visit_Expr(self, node):
        return self.visit(node.value)

    # only single targets supported
    def visit_Assign(self, node):
        if is_python_node(node):
            return cpp.Assign(self.visit(node.targets[0]),
                              self.visit(node.value))
        elif is_cpp_node(node):
            return cpp.Assign(self.visit(node.lvalue),
                              self.visit(node.rvalue))
        else:
            raise Exception ("Unknown Assign node type")

    def visit_FunctionDef(self, node):
        debug_print("In FunctionDef:")
        debug_print(ast.dump(node))
        debug_print("----")
        return cpp.FunctionBody(cpp.FunctionDeclaration(cpp.Value("void",
                                                                  node.name),
                                                        self.visit(node.args)),
                                cpp.Block([self.visit(x) for x in node.body]))


    def visit_arguments(self, node):
        """Only return the basic case: everything is void*,  no named args, no default values"""
        return [cpp.Pointer(cpp.Value("void",self.visit(x))) for x in node.args]

    def visit_Call(self, node):
        """We only handle calls that are casts; everything else (eventually) will be
           translated into callbacks into Python."""
        if isinstance(node.func, ast.Name):
            if node.func.id == "int":
                return cpp.TypeCast(cpp.Value('int', ''), self.visit(node.args[0]))
            if node.func.id == "abs":
                return cpp.Call(cpp.CName("abs"), [self.visit(x) for x in node.args])

    def visit_Print(self, node):
        if len(node.values) > 0:
            text = '<< ' + str(self.visit(node.values[0]))
        else:
            text = ''
        for fragment in node.values[1:]:
            text += ' << \" \" << ' + str(self.visit(fragment))
        return cpp.Print(text, node.nl)

    def visit_Compare(self, node):
        # only handles 1 thing on right side for now (1st op and comparator)
        # also currently not handling: Is, IsNot, In, NotIn
        ops = {'Eq':'==','NotEq':'!=','Lt':'<','LtE':'<=','Gt':'>','GtE':'>='}
        op = ops[node.ops[0].__class__.__name__]
        return cpp.Compare(self.visit(node.left), op, self.visit(node.comparators[0]))

    def visit_If(self, node):
        test = self.visit(node.test)
        body = cpp.Block([self.visit(x) for x in node.body])
        if node.orelse == []:
            orelse = None
        else:
            orelse = cpp.Block([self.visit(x) for x in node.orelse])
        return cpp.IfConv(test, body, orelse)

    def visit_Return(self, node):
        return cpp.ReturnStatement(self.visit(node.value))
    
    
class ConvertPyAST_ScalaAST(ast.NodeTransformer):
    """Class to convert from Python AST to Scala AST"""    
    def visit_Num(self,node):
    	return scala.Number(node.n)
   
    def visit_Str(self,node):
	       return scala.String(node.s)

    def visit_Name(self,node):
	       return scala.Name(node.id)

    def visit_Add(self,node):
	       return "+"

    def visit_Sub(self,node):
	       return "-" 
    
    def visit_Mult(self,node):
	       return "*"

    def visit_Div(self,node):
	       return "/"

    def visit_Mod(self,node):
	       return "%"
    
    def visit_ClassDef(self,node):
        pass
    
    def visit_FunctionDef(self,node):
        return scala.Function(scala.FunctionDeclaration(node.name, self.visit(node.args)),
                            [self.visit(x) for x in node.body])
        
    def visit_Call(self,node):

        args = []
        for a in node.args:
            args.append(self.visit(a))
        return scala.Call(self.visit(node.func), args)
    
    def visit_arguments(self,node):  
        args = []
        for a in node.args:
            args.append(self.visit(a))
        return scala.Arguments(args)
        
    def visit_Return(self,node):
        return scala.ReturnStatement(self.visit(node.value))
        
    # only single targets supported
    def visit_Assign(self, node):
        if is_python_node(node):
            return scala.Assign(self.visit(node.targets[0]),
                          self.visit(node.value))
        #below happen ever?
        elif is_scala_node(node):
            return scala.Assign(self.visit(node.lvalue),
                          self.visit(node.rvalue))
        
    def visit_AugAssign(self,node):
        return scala.AugAssign(self.visit(node.target), self.visit(node.op), self.visit(node.value))
    
    def visit_Print(self,node):
        text = []
        if len(node.values) > 0:
            text.append(self.visit(node.values[0]))
        else:
            text = ''
        for fragment in node.values[1:]:
            text.append(self.visit(fragment))
        return scala.Print(text, node.nl, node.dest)
        
    def visit_If(self,node, inner_if = False):  
        test = self.visit(node.test)
        body = [self.visit(x) for x in node.body]
        
        if node.orelse == []:
            orelse = None
        else:
            if isinstance(node.orelse[0], ast.If):
                orelse = [self.visit_If(node.orelse[0], True)]
            else:
                orelse = [self.visit(x) for x in node.orelse]

        if inner_if:
            return scala.IfConv(test,body, orelse, True)
        else:
            return scala.IfConv(test, body, orelse)
    
    def visit_Subscript(self,node):
        context= ''
        if type(node.ctx) == ast.Store:
            context ='store'
        elif type(node.ctx) == ast.Load:
            context = 'load'
        else:
            raise Exception ("Unknown Subscript Context")
        return scala.Subscript(self.visit(node.value),self.visit(node.slice), context)
    
    def visit_List(self,node):
        elements = []
        for e in node.elts:
            elements.append(self.visit(e))
        return scala.List(elements)
    
    def visit_Tuple(self,node):        
        if node.elts:
            first = node.elts[0]
            if type(first) == ast.Str and first.s == 'TYPE_DECS':
                return scala.func_types(node.elts[1:])     
            else: 
                elements =[]
                for e in node.elts:
                    elements.append(self.visit(e))
                return scala.List(elements)
        else:
            return scala.List([])
            
    """"
    only for loops of type below work:
        for item in list:
    cannot use ranges yet..        
    """        
    def visit_For(self,node):
        body = [self.visit(x) for x in node.body]
        return scala.For(self.visit(node.target), self.visit(node.iter), body)
    
    def visit_While(self,node):
        newbody = []
        for stmt in node.body:
            newbody.append(self.visit(stmt))
        return scala.While(self.visit(node.test), newbody)

    def visit_Expr(self,node):
        return self.visit(node.value)
   
    def visit_Attribute(self,node):
        return scala.Attribute(self.visit(node.value), node.attr)
        
    def visit_Compare(self, node):
        # only handles 1 thing on right side for now (1st op and comparator)
        # also currently not handling: Is, IsNot, In, NotIn
        ops = {'Eq':'==','NotEq':'!=','Lt':'<','LtE':'<=','Gt':'>','GtE':'>='}
        op = ops[node.ops[0].__class__.__name__]
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        return scala.Compare(left, op, right)
        
    def visit_BinOp(self,node):
        return scala.BinOp(self.visit(node.left), self.visit(node.op),self.visit(node.right))

    def visit_BoolOp(self,node):
        values = []
        for v in node.values:
            values.append(self.visit(v))
        return scala.BoolOp(self.visit(node.op), values)
    
    def visit_UnaryOp(self,node):
	       return scala.UnaryOp(self.visit(node.op), self.visit(node.operand))
  

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
                return cpp.BinOp(cpp.CName(self.loopvar), "+", cpp.CNumber(self.increment))
            else:
                return node

        def visit_Block(self, node):
            #print "visiting Block...."
            if self.inside_for:
                old_scope = self.in_new_scope
                self.in_new_scope = True
                #print "visiting block in ", node
                contents = [self.visit(x) for x in node.contents]
                retnode = cpp.Block(contents=[x for x in contents if x != None])
                self.in_new_scope = old_scope
            else:
                self.inside_for = True
                contents = [self.visit(x) for x in node.contents]
                retnode = cpp.Block(contents=[x for x in contents if x != None])

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
            return cpp.TypeCast(node.tp, self.visit(node.value))

        # make lvalue not a declaration
        def visit_Assign(self, node):
            if not self.in_new_scope:
                if isinstance(node.lvalue, cpp.NestedDeclarator):
                    tp, new_lvalue = node.lvalue.subdecl.get_decl_pair()
                    rvalue = self.visit(node.rvalue)
                    return cpp.Assign(cpp.CName(new_lvalue), rvalue)

                if isinstance(node.lvalue, cpp.Declarator):
                    tp, new_lvalue = node.lvalue.get_decl_pair()
                    rvalue = self.visit(node.rvalue)
                    return cpp.Assign(cpp.CName(new_lvalue), rvalue)

            return cpp.Assign(self.visit(node.lvalue), self.visit(node.rvalue))

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
        #leftover_begin = cpp.BinOp(cpp.CNumber(factor),
        #                           "*", 
        #                           cpp.BinOp(cpp.BinOp(node.end, "+", 1), "/", cpp.CNumber(factor)))


        # we begin leftover iterations at factor*( (end-initial+1) / factor ) + initial
        # note that this works due to integer division
        leftover_begin = cpp.BinOp(cpp.BinOp(cpp.BinOp(cpp.BinOp(cpp.BinOp(node.end, "-", node.initial),
                                                 "+",
                                                    cpp.CNumber(1)),
                                           "/",
                                           cpp.CNumber(factor)),
                                     "*",
                                     cpp.CNumber(factor)),
                               "+",
                               node.initial)

        new_limit = cpp.BinOp(node.end, "-", cpp.CNumber(factor-1))
        
#        debug_print("Loop unroller called with ", node.loopvar)
#        debug_print("Number of iterations: ", num_iterations)
#        debug_print("Number of unrolls: ", num_unrolls)
#        debug_print("Leftover iterations: ", leftover)

        new_increment = cpp.BinOp(node.increment, "*", cpp.CNumber(factor))

        new_block = cpp.Block(contents=node.body.contents)
        for x in xrange(1, factor):
            new_extension = copy.deepcopy(node.body)
            new_extension = LoopUnroller.UnrollReplacer(node.loopvar, x).visit(new_extension)
            new_block.extend(new_extension.contents)

        return_block = cpp.UnbracedBlock()

        unrolled_for_node = cpp.For(
            node.loopvar,
            node.initial,
            new_limit,
            #node.end,
            new_increment,
            new_block)

        leftover_for_node = cpp.For(
            node.loopvar,
            leftover_begin,
            node.end,
            node.increment,
            node.body)


        return_block.append(unrolled_for_node)

        # if we *know* this loop has no leftover iterations, then
        # we return without the leftover loop
        if not (isinstance(node.initial, cpp.CNumber) and isinstance(node.end, cpp.CNumber) and
           ((node.end.num - node.initial.num + 1) % factor == 0)):
            return_block.append(leftover_for_node)

        return return_block


class LoopBlocker(object):
    def loop_block(self, node, block_size):
        outer_incr_name = cpp.CName(node.loopvar + node.loopvar)

        new_inner_for = cpp.For(
            node.loopvar,
            outer_incr_name,
            cpp.FunctionCall("min", [cpp.BinOp(outer_incr_name, 
                                               "+", 
                                               cpp.CNumber(block_size-1)), 
                                     node.end]),
            cpp.CNumber(1),
            node.body)

        new_outer_for = cpp.For(
            node.loopvar + node.loopvar,
            node.initial,
            node.end,
            cpp.BinOp(node.increment, "*", cpp.CNumber(block_size)),
            cpp.Block(contents=[new_inner_for]))
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
            return cpp.For(self.saved_second_loop.loopvar,
                           self.saved_second_loop.initial,
                           self.saved_second_loop.end,
                           self.saved_second_loop.increment,
                           new_body)


        if self.current_loop == self.second_target:
            # save this
            self.saved_second_loop = node
            # replace this
            debug_print("replacing loop")
            return cpp.For(self.saved_first_loop.loopvar,
                           self.saved_first_loop.initial,
                           self.saved_first_loop.end,
                           self.saved_first_loop.increment,
                           node.body)


        return cpp.For(node.loopvar,
                       node.initial,
                       node.end,
                       node.increment,
                       self.visit(node.body))
