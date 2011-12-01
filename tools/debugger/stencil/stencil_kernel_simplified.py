class StencilKernel(object):
    def __init__(self, with_cilk=False):
        self.kernel_ast = ast.parse(inspect.getsource(self.kernel))
        self.model = StencilPythonFrontEnd().parse(self.kernel_ast)
        self.pure_python_kernel = self.kernel
        self.kernel = self.shadow_kernel

    def shadow_kernel(self, *args):
        model = StencilUnrollNeighborIter(model, args[0:-1], args[-1]).run()
        func = StencilConvertAST(model, args[0:-1], args[-1]).run()
        func = StencilOptimizeCpp(func, args[-1].shape, unroll_factor=4, block_factor=16).run()
        variants = [variant]; variant_names = ["kernel"]

        mod = ASPModule()
        mod.add_function("kernel", func)
        mod.kernel(*[y.data for y in args])

class StencilPythonFrontEnd(ast_tools.NodeTransformer):
    # ...
    def visit_BinOp(self, node):
        return ScalarBinOp(self.visit(node.left), node.op, self.visit(node.right))

    def visit_Num(self, node):
        return Constant(node.n)

    def visit_Call(self, node):
        assert isinstance(node.func, ast.Name), 'Cannot call expression'
        if node.func.id == 'distance' and len(node.args) == 2:
            if ((node.args[0].id == self.neighbor_target.name and node.args[1].id == self.kernel_target.name) or \
                (node.args[0].id == self.kernel_target.name and node.args[1].id == self.neighbor_target.name)):
                return NeighborDistance()
            elif ((node.args[0].id == self.neighbor_target.name and node.args[1].id == self.neighbor_target.name) or \
                  (node.args[0].id == self.kernel_target.name and node.args[1].id == self.kernel_target.name)):
                return Constant(0)
            else:
                assert False, 'Unexpected arguments to distance (expected previously defined grid point)'
        else:
            return MathFunction(node.func.id, map(self.visit, node.args))
    # ...

class StencilConvertAST(ast_tools.ConvertAST):
    # ...
    def visit_InputElementExprIndex(self, node):
        return cpp_ast.Subscript("_my_" + node.grid.name, self.visit(node.index))

    def visit_ScalarBinOp(self, node):
        return super(StencilConvertAST, self).visit_BinOp(ast.BinOp(node.left, node.op, node.right))

    def visit_MathFunction(self, node):
        return cpp_ast.FunctionCall(cpp_ast.CName(node.name), params=map(self.visit, node.args))
    # ...
