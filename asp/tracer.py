import asp.codegen.ast_tools as ast_tools
import asp.codegen.python_ast as p_ast

class TracedFunc(object):
    """
    Class that encapsulate a method to trace.  Given a method definition, it transforms it into
    a method that keeps track of types, then overrides execution to make it look 'invisible-ish'.
    To use, after the call, query the .types dict to get all the recorded type information.
    """
    

    def __init__(self, tree):
        self.tree = tree
        # this class needs to be passed a Module with a single FunctionBody instance
        self.fname = tree.body[0].name
        print p_ast.dump(tree)
        self.types = {}
        transformer = TraceTransformer()
        self.new_tree = p_ast.fix_missing_locations(transformer.visit(tree))
        self.with_return = transformer.with_return
        print p_ast.dump(self.new_tree)
        
    def __call__(self, *args, **kwargs):
        print self.fname, " called!"
        # compile & execute the definition.
        exec compile(self.new_tree, "NONE", "exec")

        # now evaluate.  we may need to eventually encapsulate locals at the callsite
        # in order to support things like self, but not sure.
        if self.with_return:
            self.types, retval = eval(self.fname).__call__(*args, **kwargs)
            return retval
        else:
            self.types = eval(self.fname).__call__(*args, **kwargs)


class TraceTransformer(ast_tools.NodeTransformer):
    """
    Class that transforms a method definition into one that memoizes the types as they are
    assigned.  This basically adds nodes that keep track of the classes of whatever is being
    assigned to LHS's in the method.
    
    On instantiation, this needs to be passed a python_ast.Module with a single python_ast.FunctionBody
    as the body of the Module.
    """
    def __init__(self):
        super(TraceTransformer, self).__init__()
        self.with_return = False
            
    def visit_Assign(self, node):
        # no plans to support more than just a single target on the LHS
        #FIXME: there has to be a better way to do this
        #FIXME: support all possible LHS: Attribute, Name, Subscript.  What about: List, Tuple?
        newnode = p_ast.parse("if True:\n _ts['%s'] = type(%s)\n" % (node.targets[0].id, node.targets[0].id)).body[0]
        newnode.lineno, newnode.col_offset = (1,1)
        
        newnode.body = [node] + newnode.body
        return newnode

    def visit_Return(self, node):
        #FIXME: what if some code paths return something, and others don't?  Dumb but people do program
        # in this way
        
        self.with_return = True
        newnode = p_ast.parse("return (_ts,)").body[0]
        newnode.value.elts.append(node.value)
        return newnode

    def process_args(self, args):
        #FIXME: only supports "simple" arguments right now
        
        return_nodes = []
        all_args = args.args
        if args.kwarg:
            all_args += args.kwarg
            
        for arg in all_args:
            newnode = p_ast.parse("_ts['%s'] = type(%s)\n" % (arg.id, arg.id)).body[0]
            return_nodes.append(newnode)

        return return_nodes
        
    def visit_FunctionDef(self, node):
        initializer = p_ast.parse("_ts = {}").body
        if not self.with_return:
            return_node = p_ast.parse("return _ts").body
        else:
            return_node = []
        new_body = [self.visit(x) for x in node.body]
        new_body = initializer + self.process_args(node.args) + new_body + return_node
        return p_ast.FunctionDef(node.name, node.args, new_body, node.decorator_list)
        
