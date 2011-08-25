from asp.codegen.cpp_ast import *
import asp.codegen.ast_tools as ast_tools
from asp.util import *

class StencilOptimizeCpp(ast_tools.ConvertAST):
    """
    Does unrolling and cache blocking on the C++ AST representation.
    """
    
    def __init__(self, model, output_grid_shape, unroll_factor, block_factor=None):    
        self.model = model
        self.output_grid_shape = output_grid_shape
        self.unroll_factor = unroll_factor
        self.block_factor = block_factor
        super(StencilOptimizeCpp, self).__init__()

    def run(self):
        self.model = self.visit(self.model)
        return self.model

    def visit_FunctionDeclaration(self, node):
        if self.block_factor:
            node.subdecl.name = "kernel_block_%s_unroll_%s" % ('_'.join([str(x) for x in self.block_factor]), self.unroll_factor)
        else:
            node.subdecl.name = "kernel_unroll_%s" % self.unroll_factor
        return node

    def visit_FunctionBody(self, node):
        # need to add the min macro, which is used by blocking
        macro = Define("min(_a,_b)", "(_a < _b ?  _a : _b)")
        node.body.contents.insert(0, macro)
        self.visit(node.fdecl)
        for i in range(0, len(node.body.contents)):
            node.body.contents[i] = self.visit(node.body.contents[i])
        return node

    def visit_For(self, node):
        inner = FindInnerMostLoop().find(node)
        if self.block_factor:
            (inner, node) = self.block_loops(inner=inner, unblocked=node)
        new_inner = ast_tools.LoopUnroller().unroll(inner, self.unroll_factor)
        node = ast_tools.ASTNodeReplacerCpp(inner, new_inner).visit(node)
        return node

    def block_loops(self, inner, unblocked):
        #factors = [self.block_factor for x in self.output_grid_shape]
        #factors[len(self.output_grid_shape)-1] = 1

        
        # use the helper class below to do the actual blocking.
        blocked = StencilCacheBlocker().block(unblocked, self.block_factor)

        # need to update inner to point to the innermost in the new blocked version
        inner = FindInnerMostLoop().find(blocked)

        assert(inner != None)
        return [inner, blocked]

  

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
                return For(node.loopvar,
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
