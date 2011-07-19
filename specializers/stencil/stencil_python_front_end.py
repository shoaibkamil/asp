"""Takes a Python AST and converts it to a corresponding StencilModel.

Throws an exception if the input does not represent a valid stencil
kernel program. This is the first stage of processing and is done only
once when a stencil class is initialized.
"""

from stencil_model import *
from assert_utils import *
import ast
from asp.util import *

# class to convert from Python AST to StencilModel
class StencilPythonFrontEnd(ast.NodeTransformer):
    def __init__(self):
        super(StencilPythonFrontEnd, self).__init__()

    def parse(self, ast):
        return self.visit(ast)

    def visit_Module(self, node):
        body = map(self.visit, node.body)
        assert len(body) == 1
        assert_has_type(body[0], StencilModel)
        return body[0]

    def visit_FunctionDef(self, node):
        assert len(node.decorator_list) == 0
        arg_ids = self.visit(node.args)
        assert arg_ids[0] == 'self'
        self.output_arg_id = arg_ids[-1]
        self.input_arg_ids = arg_ids[1:-1]
        kernels = map(self.visit, node.body)
        interior_kernels = map(lambda x: x['kernel'], filter(lambda x: x['kernel_type'] == 'interior_points', kernels))
        border_kernels = map(lambda x: x['kernel'], filter(lambda x: x['kernel_type'] == 'border_points', kernels))
        assert len(interior_kernels) <= 1, 'Can only have one loop over interior points'
        assert len(border_kernels) <= 1, 'Can only have one loop over border points'
        return StencilModel(map(lambda x: Identifier(x), self.input_arg_ids),
                            interior_kernels[0] if len(interior_kernels) > 0 else Kernel([]),
                            border_kernels[0] if len(border_kernels) > 0 else Kernel([]))

    def visit_arguments(self, node):
        assert node.vararg == None, 'kernel function may not take variable argument list'
        assert node.kwarg == None, 'kernel function may not take variable argument list'
        return map (self.visit, node.args)

    def visit_Name(self, node):
        return node.id

    def visit_For(self, node):
        # check if this is the right kind of For loop
        if (type(node.iter) is ast.Call and
            type(node.iter.func) is ast.Attribute):

            if (node.iter.func.attr == "interior_points" or
                node.iter.func.attr == "border_points"):
                assert node.iter.args == [] and node.iter.starargs == None and node.iter.kwargs == None, 'Invalid argument list for %s()' % node.iter.func.attr
                grid_id = self.visit(node.iter.func.value)
                assert grid_id == self.output_arg_id, 'Can only iterate over %s of output grid "%s" but "%s" was given' % (node.iter.func.attr, self.output_arg_id, grid_id)
                self.kernel_target = self.visit(node.target)
                body = map(self.visit, node.body)
                self.kernel_target = None
                return {'kernel_type': node.iter.func.attr, 'kernel': Kernel(body)}

            elif node.iter.func.attr == "neighbors":
                assert len(node.iter.args) == 2 and node.iter.starargs == None and node.iter.kwargs == None, 'Invalid argument list for neighbors()'
                self.neighbor_grid_id = self.visit(node.iter.func.value)
                assert self.neighbor_grid_id in self.input_arg_ids, 'Can only iterate over neighbors in an input grid but "%s" was given' % grid_id
                neighbors_of_grid_id = self.visit(node.iter.args[0])
                assert neighbors_of_grid_id == self.kernel_target, 'Can only iterate over neighbors of an output grid point but "%s" was given' % neighbors_of_grid_id
                self.neighbor_target = self.visit(node.target)
                body = map(self.visit, node.body)
                self.neighbor_target = None
                self.neigbor_grid_id = None
                neighbors_id = self.visit(node.iter.args[1])
                return StencilNeighborIter(Identifier(self.neighbor_grid_id), neighbors_id, body)
            else:
                assert False, 'Invalid call in For loop argument \'%s\', can only iterate over interior_points, boder_points, or neighbor_points of a grid' % node.iter.func.attr
        else:
            assert False, 'Unexpected For loop \'%s\', can only iterate over interior_points, boder_points, or neighbor_points of a grid' % node

    def visit_AugAssign(self, node):
        target = self.visit(node.target)
        assert type(target) is OutputElement, 'Only assignments to current output element permitted'
        return OutputAssignment(ScalarBinOp(OutputElement(), node.op, self.visit(node.value)))

    def visit_Assign(self, node):
        targets = map (self.visit, node.targets)
        assert len(targets) == 1 and type(targets[0]) is OutputElement, 'Only assignments to current output element permitted'
        return OutputAssignment(self.visit(node.value))

    def visit_Subscript(self, node):
        if type(node.slice) is ast.Index:
            grid_id = self.visit(node.value)
            target = self.visit(node.slice.value)
            if grid_id == self.output_arg_id and target == self.kernel_target:
                return OutputElement()
            elif target == self.kernel_target:
                return InputElementZeroOffset(Identifier(grid_id))
            elif grid_id == self.neighbor_grid_id and target == self.neighbor_target:
                return Neighbor()
            elif isinstance(target, Expr):
                return InputElementExprIndex(Identifier(grid_id), target)
            else:
                assert False, 'Unexpected subscript index \'%s\' on grid \'%s\'' % (target, grid_id)
        else:
            assert False, 'Unsupported subscript object \'%s\' on grid \'%s\'' % (node.slice, grid_id)

    def visit_BinOp(self, node):
        return ScalarBinOp(self.visit(node.left), node.op, self.visit(node.right))

    def visit_Num(self, node):
        return Constant(node.n)

    def visit_Call(self, node):
        assert isinstance(node.func, ast.Name), 'Cannot call expression'
        return MathFunction(node.func.id, map(self.visit, node.args))
