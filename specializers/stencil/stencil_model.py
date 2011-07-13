import types
import ast
from assert_utils import *

from asp.tree_grammar import *
parse('''
# Tree grammar for stencil semantic model, based on language specification and other feedback

StencilModel(input_grids=Identifier*, interior_kernel=Kernel, border_kernel=Kernel)
    check StencilModelStructuralConstraintsVerifier(self).verify()

Identifier(name)

Kernel(body=(StencilNeighborIter | OutputAssignment)*)

StencilNeighborIter(grid=Identifier, distance=Constant, body=OutputAssignment*)
    check assert self.distance.value >= 0, "distance must be nonnegative but was: %d" % self.distance.value

# Assigns Expr to current output element
OutputAssignment(value=Expr)

Expr = Constant
     | Neighbor      # Refers to current neighbor inside a StencilNeighborIter
     | OutputElement # Refers to current output element
     | InputElement
     | ScalarBinOp

Constant(value = types.IntType | types.LongType | types.FloatType)

# Offsets are relative to current output element location, given as a list of integers,
# one per dimension.
InputElement(grid=Identifier, offset_list=types.IntType*)

ScalarBinOp(left=Expr, op=(ast.Add|ast.Sub|ast.Mult|ast.Div|ast.FloorDiv), right=Expr)
''', globals(), checker='StencilModelChecker')

# Verifies a few structural constraints (semantic properties) of the tree
class StencilModelStructuralConstraintsVerifier(ast.NodeVisitor):
    def __init__(self, stencil_model):
        assert_has_type(stencil_model, StencilModel)
        self.model = stencil_model
        super(StencilModelStructuralConstraintsVerifier, self).__init__()

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
