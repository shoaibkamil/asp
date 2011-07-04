'''
Tree grammar for stencil semantic model, based on language specification and other feedback:

StencilModel(inputGrids=Identifier*, interiorKernel=Kernel, boundaryKernel=Kernel)

Identifier(name=string)

Kernel(body=(StencilNeighborIter | OutputAssignment)*)

StencilNeighborIter(grid=Identifier, distance=integer, body=OutputAssignment*)

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
    def __init__(self, inputGrids, interiorKernel, boundaryKernel):
        self._fields = ('inputGrids', 'interiorKernel', 'boundaryKernel')
        super(StencilModel, self).__init__()
        assert_is_list_of(inputGrids, Identifier)
        assert_has_type(interiorKernel, Kernel)
        assert_has_type(boundaryKernel, Kernel)
        self.inputGrids = inputGrids
        self.interiorKernel = interiorKernel
        self.boundaryKernel = boundaryKernel
        self.perform_checks()

    def perform_checks(self):
        # TODO: Check that all input grid references in tree refer to
        # identifiers in self.inputGrids

        # TODO: Check that Neighbor only used inside StencilNeighborIter
        pass

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
        assert_has_type(distance, IntType)
        assert distance >= 0, "distance must be nonnegative but was: %d" % distance
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
