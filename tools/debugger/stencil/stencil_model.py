"""Defines the semantic model, a tree data structure representing a valid stencil kernel program.

The semantic model is specified using Asp's tree_grammar DSL.  The
stencil_model classes have generated assertions and additional manual
structural checks to prevent the construction of a tree not
corresponding to a valid stencil kernel program.
"""

import types
import ast
from assert_utils import *

from asp.tree_grammar import *
parse('''
# Tree grammar for stencil semantic model, based on language specification and other feedback

StencilModel(input_grids=Identifier*, interior_kernel=Kernel, border_kernel=Kernel)
    check StencilModelStructuralConstraintsVerifier(self).verify()
    check assert len(set([x.name for x in self.input_grids]))==len(self.input_grids), 'Input grids must have distinct names'

Identifier(name)

Kernel(body=(StencilNeighborIter | OutputAssignment)*)

StencilNeighborIter(grid=Identifier, neighbors_id=Constant, body=OutputAssignment*)
    check assert self.neighbors_id.value >= 0, "neighbors_id must be nonnegative but was: %d" % self.neighbors_id.value

# Assigns Expr to current output element
OutputAssignment(value=Expr)

Expr = Constant
     | Neighbor      # Refers to current neighbor inside a StencilNeighborIter
     | OutputElement # Refers to current output element
     | InputElement
     | InputElementZeroOffset
     | InputElementExprIndex
     | ScalarBinOp
     | MathFunction
     | NeighborDistance

Constant(value = types.IntType | types.LongType | types.FloatType)

# Offsets are relative to current output element location, given as a list of integers,
# one per dimension.
InputElement(grid=Identifier, offset_list=types.IntType*)

# Input element at same position as current output element
InputElementZeroOffset(grid=Identifier)

# Input element at an index given by an expression (must be 1D grid)
InputElementExprIndex(grid=Identifier, index=Expr)

# Use a built-in pure math function
MathFunction(name, args=Expr*)
    check assert self.name in math_functions.keys(), "Tried to use function \'%s\' not in math_functions list" % self.name
    check assert len(self.args) == math_functions[self.name], "Expected %d arguments to math function \'%s\' but received %d arguments" % (math_functions[self.name], self.name, len(self.args))

ScalarBinOp(left=Expr, op=(ast.Add|ast.Sub|ast.Mult|ast.Div|ast.FloorDiv|ast.Mod), right=Expr)
''', globals(), checker='StencilModelChecker')

# Gives number of arguments for each math function
math_functions = {'int':1, 'abs':1}

# Verifies a few structural constraints (semantic properties) of the tree
class StencilModelStructuralConstraintsVerifier(ast.NodeVisitor):
    def __init__(self, stencil_model):
        assert_has_type(stencil_model, StencilModel)
        self.model = stencil_model
        self.in_stencil_neighbor_iter = False
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
