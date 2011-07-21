import asp.codegen.cpp_ast as cpp_ast
import asp.codegen.ast_tools as ast_tools


class StencilCacheBlocker(object):
    class StripMineLoopByIndex(ast_tools.NodeTransformer):
        def __init__(self, index, factor):
            self.current_idx = -1
            self.target_idx = index
            self.factor = factor
            super(StencilCacheBlocker.StripMineLoopByIndex, self).__init__()
            
        def visit_For(self, node):
            self.current_idx += 1

            print "Searching for loop %d, currently at %d" % (self.target_idx, self.current_idx)

            if self.current_idx == self.target_idx:
                print "Before blocking:"
                print node
                
                return ast_tools.LoopBlocker().loop_block(node, self.factor)
            else:
                return cpp_ast.For(node.loopvar,
                           node.initial,
                           node.end,
                           node.increment,
                           self.visit(node.body))
            
    def block(self, tree, factors):
        # first we apply strip mining to the loops given in factors
        for x in xrange(len(factors)):
            print "Doing loop %d by %d" % (x*2, factors[x])
            tree = StencilCacheBlocker.StripMineLoopByIndex(x*2, factors[x]).visit(tree)
            print tree

        # now we move all the outer strip-mined loops to be outermost

        for x in xrange(1,len(factors)):
            self.bubble(tree, 2*x, x)
    
        return tree
        
    def bubble(self, tree, index, new_index):
        for x in xrange(index-new_index):
            print "In bubble, switching %d and %d" % (index-x-1, index-x)
            ast_tools.LoopSwitcher().switch(tree, index-x-1, index-x)
        return tree
