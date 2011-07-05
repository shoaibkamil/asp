'''
Tree grammar for stencil semantic model, based on language specification and other feedback:

StencilModel(input_grids=Identifier*, interior_kernel=Kernel, border_kernel=Kernel)

Identifier(name=string)

Kernel(body=(StencilNeighborIter | OutputAssignment)*)

StencilNeighborIter(grid=Identifier, distance=Constant, body=OutputAssignment*)

# Assigns Expr to current output element
OutputAssignment(value=Expr)

Expr = Constant      # int, long, or float
     | Neighbor      # Refers to current neighbor inside a StencilNeighborIter
     | OutputElement
     | InputElement
     | ScalarBinOp

# Offsets are relative to current output element location, given as a list of integers,
# one per dimension.
InputElement(grid=Identifier, offset_list=Integer*)

ScalarBinOp(left=Expr, op=('+'|'-'|'*'|'/'), right=Expr)

'''

from types import *
from assert_utils import *
import ast

class StencilNode(ast.AST):
    def __init__(self):
        pass

class StencilModel(StencilNode):
    def __init__(self, input_grids, interior_kernel, border_kernel):
        self._fields = ('input_grids', 'interior_kernel', 'border_kernel')
        super(StencilModel, self).__init__()
        assert_is_list_of(input_grids, Identifier)
        assert_has_type(interior_kernel, Kernel)
        assert_has_type(border_kernel, Kernel)
        self.input_grids = input_grids
        self.interior_kernel = interior_kernel
        self.border_kernel = border_kernel
        self.verify()

    def verify(self):
        StencilModelVerifier(self).verify()

class Identifier(StencilNode):
    def __init__(self, name):
        self._fields = ('name',)
        super(Identifier, self).__init__()
        self.name = name

class Kernel(StencilNode):
    def __init__(self, body):
        self._fields = ('body',)
        super(Kernel, self).__init__()
        assert_is_list_of(body, [StencilNeighborIter, OutputAssignment])
        self.body = body

class StencilNeighborIter(StencilNode):
    def __init__(self, grid, distance, body):
        self._fields = ('grid', 'distance', 'body')
        super(StencilNeighborIter, self).__init__()
        assert_has_type(grid, Identifier)
        assert_has_type(distance, Constant)
        assert distance.value >= 0, "distance must be nonnegative but was: %d" % distance.value
        assert_is_list_of(body, OutputAssignment)
        self.grid = grid
        self.distance = distance
        self.body = body

class Expr(StencilNode):
    def __init__(self):
        super(Expr, self).__init__()

# Assigns value to current output element
class OutputAssignment(StencilNode):
    def __init__(self, value):
        self._fields = ('value',)
        super(OutputAssignment, self).__init__()
        assert_has_type(value, Expr)
        self.value = value

class Constant(Expr):
    def __init__(self, value):
        self._fields = ('value',)
        super(Constant, self).__init__()
        assert_has_type(value, [IntType, LongType, FloatType])
        self.value = value

class Neighbor(Expr):
    def __init__(self):
        super(Neighbor, self).__init__()

class OutputElement(Expr):
    def __init__(self):
        super(OutputElement, self).__init__()

# Offsets are relative to current output element location, given
# as a list of integers, one per dimension.
class InputElement(Expr):
    def __init__(self, grid, offset_list):
        self._fields = ('grid', 'offset_list')
        super(InputElement, self).__init__()
        assert_has_type(grid, Identifier)
        assert_is_list_of(offset_list, [IntType, LongType])
        self.grid = grid
        self.offset_list = offset_list

class ScalarBinOp(Expr):
    def __init__(self, left, op, right):
        self._fields = ('left', 'op', 'right')
        super(ScalarBinOp, self).__init__()
        assert_has_type(left, Expr)
        assert_has_type(right, Expr)
        assert(type(op) is ast.Add or
               type(op) is ast.Sub or
               type(op) is ast.Mult or
               type(op) is ast.Div or
               type(op) is ast.FloorDiv)
        self.left = left
        self.op = op
        self.right = right

class StencilModelVerifier(ast.NodeVisitor):
    def __init__(self, stencil_model):
        assert_has_type(stencil_model, StencilModel)
        self.model = stencil_model
        super(StencilModelVerifier, self).__init__()

    def verify(self):
        self.visit(self.model)

    def visit_StencilModel(self, node):
        self.input_grid_names = map(lambda x: x.name, node.input_grids)
        self.generic_visit(node)

    def visit_Identifier(self, node):
        assert node.name in self.input_grid_names, 'Identifier %s not listed among input grid identifiers' % node.name

    def visit_StencilNeighborIter(self, node):
        self.in_stencil_neighbor_iter = True
        self.generic_visit(node)
        self.in_stencil_neighbor_iter = False

    def visit_Neighbor(self, node):
        assert self.in_stencil_neighbor_iter, 'Neighbor node allowed only inside StencilNeighborIter'
