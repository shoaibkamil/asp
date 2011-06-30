'''
Tree grammar for stencil semantic model, based on language specification and other feedback:

StencilModel(inputGrids=Identifier*, interiorKernel=Kernel, boundaryKernel=Kernel)

Identifier(name=string)

Kernel(body=(StencilNeighborIter | OutputAssignment)*)

StencilNeighborIter(grid=Identifier, distance=integer, body=OutputAssignment*)

# Assigns Expr to current output element
OutputAssignment(value=Expr)

Expr( value=( Constant      # int, long, or float
            | Neighbor      # Refers to current neighbor inside a StencilNeighborIter
            | OutputElement
            | InputElement
            | ScalarBinOp) )

# Offsets are relative to current output element location, given as a list of integers,
# one per dimension.
InputElement(grid=Identifier, offset_list=Integer*)

ScalarBinOp(left=Expr, op=('+'|'-'|'*'|'/'), right=Expr)

'''

from types import *
import ast

def assert_has_type(x, t, x_name='obj'):
    if type(t) is ListType:
        type_found = False
        for t2 in t:
            if isinstance(x, t2):
                type_found = True
        assert type_found, "%s is not one of the types %s: %s" % (x_name, t, `x`)
    else:
        assert isinstance(x, t), "%s is not %s: %s" % (x_name, t, `x`)

def assert_is_list_of(lst, t, lst_name='list'):
    assert_has_type(lst, ListType, lst_name)
    for x in lst:
        assert_has_type(x, t, "%s element" % lst_name)

class StencilNode(object):
    def __init__(self):
        pass

class StencilModel(StencilNode):
    def __init__(self, inputGrids, interiorKernel, boundaryKernel):
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
        super(Identifier, self).__init__()
        self.name = name

class Kernel(StencilNode):
    def __init__(self, body):
        super(Kernel, self).__init__()
        assert_is_list_of(body, [StencilNeighborIter, OutputAssignment])
        self.body = body

class StencilNeighborIter(StencilNode):
    def __init__(self, grid, distance, body):
        super(StencilNeighborIter, self).__init__()
        assert_has_type(grid, Identifier)
        assert_has_type(distance, IntType)
        assert distance >= 0, "distance must be nonnegative but was: %d" % distance
        assert_is_list_of(body, OutputAssignment)
        self.grid = grid
        self.distance = distance
        self.body = body

# Assigns value to current output element
class OutputAssignment(StencilNode):
    def __init__(self, value):
        super(OutputAssignment, self).__init__()
        assert_has_type(value, Expr)
        self.value = value

class Expr(StencilNode):
    def __init__(self, value):
        super(Expr, self).__init__()
        assert_has_type(value, [Constant,
                                Neighbor,
                                OutputElement,
                                InputElement,
                                ScalarBinOp])
        self.value = value

class Constant(StencilNode):
    def __init__(self, value):
        super(Constant, self).__init__()
        assert_has_type(value, [IntType, LongType, FloatType])
        self.value = value

class Neighbor(StencilNode):
    def __init__(self):
        super(Neighbor, self).__init__()

class OutputElement(StencilNode):
    def __init__(self):
        super(OutputElement, self).__init__()

# Offsets are relative to current output element location, given
# as a list of integers, one per dimension.
class InputElement(StencilNode):
    def __init__(self, grid, offset_list):
        super(InputElement, self).__init__()
        assert_has_type(grid, Identifier)
        assert_is_list_of(offset_list, [IntType, LongType])
        self.grid = grid
        self.offset_list = offset_list

class ScalarBinOp(StencilNode):
    def __init__(self, left, op, right):
        super(ScalarBinOp, self).__init__()
        assert_has_type(left, Expr)
        assert_has_type(right, Expr)
        assert(op in [ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv])
        self.left = left
        self.op = op
        self.right = right
