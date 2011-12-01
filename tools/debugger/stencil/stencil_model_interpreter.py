"""Takes a StencilModel and interprets it (slowly) in Python.

Facilitates isolation of bugs between stages in the specializer.
"""

from stencil_model import *
from stencil_grid import distance
import ast
from assert_utils import *
import math

class StencilModelInterpreter(ast.NodeVisitor):
    def __init__(self, stencil_model, input_grids, output_grid):
        assert_has_type(stencil_model, StencilModel)
        assert len(input_grids) == len(stencil_model.input_grids), 'Incorrect number of input grids'
        self.model = stencil_model
        self.input_grids = input_grids
        self.output_grid = output_grid
        super(StencilModelInterpreter, self).__init__()

    def run(self):
        self.visit(self.model)

    def visit_StencilModel(self, node):
        self.input_dict = dict()
        for i in range(len(node.input_grids)):
            self.input_dict[node.input_grids[i].name] = self.input_grids[i]
            
        for x in self.output_grid.interior_points():
            self.current_output_point = x
            self.visit(node.interior_kernel)
        for x in self.output_grid.border_points():
            self.current_output_point = x
            self.visit(node.border_kernel)

    def visit_Identifier(self, node):
        return self.input_dict[node.name]

    def visit_StencilNeighborIter(self, node):
        grid = self.visit(node.grid)
        neighbors_id = self.visit(node.neighbors_id)
        self.current_neighbor_grid = grid
        for x in grid.neighbors(self.current_output_point, neighbors_id):
            self.current_neighbor_point = x
            for statement in node.body:
                self.visit(statement)

    def visit_OutputAssignment(self, node):
        self.output_grid[self.current_output_point] = self.visit(node.value)
        
    def visit_Constant(self, node):
        return node.value

    def visit_Neighbor(self, node):
        return self.current_neighbor_grid[self.current_neighbor_point]

    def visit_OutputElement(self, node):
        return self.output_grid[self.current_output_point]

    def visit_InputElement(self, node):
        grid = self.visit(node.grid)
        x = tuple(map(lambda a,b: a+b, list(self.current_output_point), node.offset_list))
        return grid[x]

    def visit_InputElementZeroOffset(self, node):
        grid = self.visit(node.grid)
        return grid[self.current_output_point]

    def visit_InputElementExprIndex(self, node):
        grid = self.visit(node.grid)
        return grid[self.visit(node.index)]

    def visit_ScalarBinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if type(node.op) is ast.Add:
            return left + right
        elif type(node.op) is ast.Sub:
            return left - right
        elif type(node.op) is ast.Mult:
            return left * right
        elif type(node.op) is ast.Div:
            return left / right
        elif type(node.op) is ast.FloorDiv:
            return left // right

    math_func_to_python_func = {'abs': abs, 'int': int}

    def visit_MathFunction(self, node):
        func = self.math_func_to_python_func[node.name]
        args = map(self.visit, node.args)
        return apply(func, args)

    def visit_NeighborDistance(self, node):
        return distance(self.current_neighbor_point, self.current_output_point)
