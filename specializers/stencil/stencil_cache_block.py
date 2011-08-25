import asp.codegen.cpp_ast as cpp_ast
import asp.codegen.ast_tools as ast_tools
from stencil_convert import *
from asp.util import *

class StencilConvertASTBlocked(StencilConvertAST):
    """
    Converter class for cache blocked stencils.
    """
    
    class FindInnerMostLoop(ast_tools.NodeVisitor):
        """
        Helper class that returns the innermost loop of perfectly nested loops.
        """
        def __init__(self):
            self.inner_most = None

        def find(self, node):
            self.visit(node)
            return self.inner_most
        
        def visit_For(self, node):
            self.inner_most = node
            self.visit(node.body)
    
    def __init__(self, model, input_grids, output_grid, unroll_factor=None, block_factor=None):    
        self.block_factor = block_factor
    	super(StencilConvertASTBlocked, self).__init__(model, input_grids, output_grid, unroll_factor=unroll_factor)
        
    def gen_loops(self, node):
        inner, unblocked = super(StencilConvertASTBlocked, self).gen_loops(node)

        if not self.block_factor:
            return [inner, unblocked]

        #factors = [self.block_factor for x in self.output_grid.shape]
        #factors[len(self.output_grid.shape)-1] = 1
        
        # use the helper class below to do the actual blocking.
        blocked = StencilCacheBlocker().block(unblocked, self.block_factor)

        # need to update inner to point to the innermost in the new blocked version
        inner = StencilConvertASTBlocked.FindInnerMostLoop().find(blocked)

        assert(inner != None)
        return [inner, blocked]

    def visit_StencilModel(self, node):
        ret = super(StencilConvertASTBlocked, self).visit_StencilModel(node)
        debug_print("in VISIT" + str(ret))

        # need to add the min macro, which is used by blocking
        macro = cpp_ast.Define("min(_a,_b)", "(_a < _b ?  _a : _b)")
        ret.body.contents.insert(0, macro)

        # modify the function name to reflect both blocking and unrolling
        if self.block_factor and self.unroll_factor:
            debug_print("CHANGING FUNCTION NAME")
            func_name = "kernel_block_%s_unroll_%s" % ('_'.join([str(x) for x in self.block_factor]), self.unroll_factor)
            ret = cpp_ast.FunctionBody(cpp_ast.FunctionDeclaration(cpp_ast.Value("void", func_name), ret.fdecl.arg_decls),
                                       ret.body)

        return ret

class StencilCacheBlocker(object):
    """
    Class that takes a tree of perfectly-nested For loops (as in a stencil) and performs standard cache blocking
    on them.  Usage: StencilCacheBlocker().block(tree, factors) where factors is a tuple, one for each loop nest
    in the original tree.
    """
    class StripMineLoopByIndex(ast_tools.NodeTransformer):
        """Helper class that strip mines a loop of a particular index in the nest."""
        def __init__(self, index, factor):
            self.current_idx = -1
            self.target_idx = index
            self.factor = factor
            super(StencilCacheBlocker.StripMineLoopByIndex, self).__init__()
            
        def visit_For(self, node):
            self.current_idx += 1

            debug_print("Searching for loop %d, currently at %d" % (self.target_idx, self.current_idx))

            if self.current_idx == self.target_idx:
                debug_print("Before blocking:")
                debug_print(node)
                
                return ast_tools.LoopBlocker().loop_block(node, self.factor)
            else:
                return cpp_ast.For(node.loopvar,
                           node.initial,
                           node.end,
                           node.increment,
                           self.visit(node.body))
            
    def block(self, tree, factors):
        """Main method in StencilCacheBlocker.  Used to block the loops in the tree."""
        # first we apply strip mining to the loops given in factors
        for x in xrange(len(factors)):
            debug_print("Doing loop %d by %d" % (x*2, factors[x]))

            # we may want to not block a particular loop, e.g. when doing Rivera/Tseng blocking
            if factors[x] > 1:
                tree = StencilCacheBlocker.StripMineLoopByIndex(x*2, factors[x]).visit(tree)
            debug_print(tree)

        # now we move all the outer strip-mined loops to be outermost
        for x in xrange(1,len(factors)):
            if factors[x] > 1:
                tree = self.bubble(tree, 2*x, x)
    
        return tree
        
    def bubble(self, tree, index, new_index):
        """
        Helper function to 'bubble up' a loop at index to be at new_index (new_index < index)
        while preserving the ordering of the loops between index and new_index.
        """
        for x in xrange(index-new_index):
            debug_print("In bubble, switching %d and %d" % (index-x-1, index-x))
            tree = ast_tools.LoopSwitcher().switch(tree, index-x-1, index-x)
        return tree
