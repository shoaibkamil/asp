import unittest2 as unittest
from stencil_python_front_end import *
from assert_utils import *
import ast

class BasicTests(unittest.TestCase):
    def test_parse_interior(self):
        python_ast = ast.parse(
'''
def kernel(self, in_grid, out_grid):
    for x in out_grid.interior_points():
        out_grid[x] = in_grid[x]
'''
                              )
        stencil_model = StencilPythonFrontEnd().parse(python_ast)
        assert_has_type(stencil_model, StencilModel)
        assert len(stencil_model.interior_kernel.body) > 0 and len(stencil_model.border_kernel.body) == 0
        assert_contains_node_type(stencil_model, InputElementZeroOffset)

    def test_parse_neighbor(self):
        python_ast = ast.parse(
'''
def kernel(self, in_grid, out_grid):
    for x in out_grid.interior_points():
        for y in in_grid.neighbors(x, 1):
            out_grid[x] = out_grid[x] + in_grid[y]
'''
                              )
        stencil_model = StencilPythonFrontEnd().parse(python_ast)
        assert_has_type(stencil_model, StencilModel)
        assert len(stencil_model.interior_kernel.body) > 0 and len(stencil_model.border_kernel.body) == 0
        assert_contains_node_type(stencil_model, ScalarBinOp)
        assert_contains_node_type(stencil_model, ast.Add)

    def test_parse_border(self):
        python_ast = ast.parse(
'''
def kernel(self, in_grid, out_grid):
    for x in out_grid.border_points():
        for y in in_grid.neighbors(x, 1):
            out_grid[x] = out_grid[x] + in_grid[y]
'''
                              )
        stencil_model = StencilPythonFrontEnd().parse(python_ast)
        assert_has_type(stencil_model, StencilModel)
        assert len(stencil_model.interior_kernel.body) == 0 and len(stencil_model.border_kernel.body) > 0
        assert_contains_node_type(stencil_model, ScalarBinOp)
        assert_contains_node_type(stencil_model, ast.Add)

    def test_parse_both(self):
        python_ast = ast.parse(
'''
def kernel(self, in_grid, out_grid):
    for x in out_grid.interior_points():
        for y in in_grid.neighbors(x, 1):
            out_grid[x] = out_grid[x] + in_grid[y]
    for x in out_grid.border_points():
        for y in in_grid.neighbors(x, 1):
            out_grid[x] = out_grid[x] + in_grid[y]
'''
                              )
        stencil_model = StencilPythonFrontEnd().parse(python_ast)
        assert_has_type(stencil_model, StencilModel)
        assert len(stencil_model.interior_kernel.body) > 0 and len(stencil_model.border_kernel.body) > 0
        assert_contains_node_type(stencil_model, ScalarBinOp)
        assert_contains_node_type(stencil_model, ast.Add)

    def test_parse_constants(self):
        python_ast = ast.parse(
'''
def kernel(self, in_grid, out_grid):
    for x in out_grid.interior_points():
        for y in in_grid.neighbors(x, 1):
            out_grid[x] = out_grid[x] - 1/4*in_grid[y] + 1.0/4*in_grid[y]
'''
                              )
        stencil_model = StencilPythonFrontEnd().parse(python_ast)
        assert_has_type(stencil_model, StencilModel)
        assert_contains_node_type(stencil_model, Constant)
        assert_contains_node_type(stencil_model, ast.Sub)
        assert_contains_node_type(stencil_model, ast.Mult)
        assert_contains_node_type(stencil_model, ast.Div)

    def test_parse_function(self):
        python_ast = ast.parse(
'''
def kernel(self, in_grid, out_grid):
    for x in out_grid.interior_points():
        for y in in_grid.neighbors(x, 1):
            out_grid[x] = abs(-2)
'''
                              )
        stencil_model = StencilPythonFrontEnd().parse(python_ast)
        assert_has_type(stencil_model, StencilModel)
        assert_contains_node_type(stencil_model, MathFunction)

    def test_parse_distance(self):
        python_ast = ast.parse(
'''
def kernel(self, in_grid, out_grid):
    for x in out_grid.interior_points():
        for y in in_grid.neighbors(x, 1):
            out_grid[x] += distance(x,y)
'''
                              )
        stencil_model = StencilPythonFrontEnd().parse(python_ast)
        assert_has_type(stencil_model, StencilModel)
        assert_contains_node_type(stencil_model, NeighborDistance)

    def test_parse_distance_zero(self):
        python_ast = ast.parse(
'''
def kernel(self, in_grid, out_grid):
    for x in out_grid.interior_points():
        for y in in_grid.neighbors(x, 1):
            out_grid[x] += distance(x,x)
            out_grid[x] += distance(y,y)
'''
                              )
        stencil_model = StencilPythonFrontEnd().parse(python_ast)
        assert_has_type(stencil_model, StencilModel)
        assert not(ContainsNodeTypeVisitor().contains(stencil_model, NeighborDistance)), "Expected tree not to contain node of type NeighorDistance for self-distance"

    def test_parse_bad_function(self):
        python_ast = ast.parse(
'''
def kernel(self, in_grid, out_grid):
    for x in out_grid.interior_points():
        out_grid[x] = foo()
'''
                              )
        with self.assertRaises(AssertionError):
            stencil_model = StencilPythonFrontEnd().parse(python_ast)

    def test_parse_bad_num_args(self):
        python_ast = ast.parse(
'''
def kernel(self, in_grid, out_grid):
    for x in out_grid.interior_points():
        out_grid[x] = abs(1, 2)
'''
                              )
        with self.assertRaises(AssertionError):
            stencil_model = StencilPythonFrontEnd().parse(python_ast)

    def test_parse_bad_for(self):
        python_ast = ast.parse(
'''
def kernel(self, in_grid, out_grid):
    for x in [1,2,3]:
        out_grid[x] = 1
'''
                              )
        with self.assertRaises(AssertionError):
            stencil_model = StencilPythonFrontEnd().parse(python_ast)

def assert_contains_node_type(root_node, node_type):
    assert ContainsNodeTypeVisitor().contains(root_node, node_type), "Expected tree to contain node of type %s" % node_type

class ContainsNodeTypeVisitor(ast.NodeVisitor):
    def contains(self, root_node, node_type):
        self.node_type = node_type
        self.result = False
        self.visit(root_node)
        return self.result

    def visit(self, node):
        if isinstance(node, self.node_type):
            self.result = True
        self.generic_visit(node)

if __name__ == '__main__':
    unittest.main()

