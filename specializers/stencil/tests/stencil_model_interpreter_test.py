import unittest2 as unittest
from stencil_model_interpreter import *
from stencil_grid import *
import numpy
import ast

class BasicTests(unittest.TestCase):
    def setUp(self):
        self.in_grid = StencilGrid([5,5])
        self.in_grid.data = numpy.ones([5,5])
        self.out_grid = StencilGrid([5,5])

    def test_identity(self):
        model = StencilModel([Identifier('in_grid')],
                             Kernel([StencilNeighborIter(Identifier('in_grid'),
                                                         Constant(1),
                                                         [OutputAssignment(InputElementZeroOffset(Identifier('in_grid')))])]),
                             Kernel([]))

        StencilModelInterpreter(model, [self.in_grid], self.out_grid).run()
        for x in self.out_grid.interior_points():
            self.assertEqual(self.out_grid[x], 1)
        for x in self.out_grid.border_points():
            self.assertEqual(self.out_grid[x], 0)

    def test_add_neighbors(self):
        model = StencilModel([Identifier('in_grid')],
                             Kernel([StencilNeighborIter(Identifier('in_grid'),
                                                         Constant(1),
                                                         [OutputAssignment(ScalarBinOp(OutputElement(), ast.Add(), Neighbor()))])]),
                             Kernel([]))

        StencilModelInterpreter(model, [self.in_grid], self.out_grid).run()
        for x in self.out_grid.interior_points():
            self.assertEqual(self.out_grid[x], 4)
        for x in self.out_grid.border_points():
            self.assertEqual(self.out_grid[x], 0)

    def test_constant(self):
        model = StencilModel([Identifier('in_grid')],
                             Kernel([OutputAssignment(Constant(7))]),
                             Kernel([]))

        StencilModelInterpreter(model, [self.in_grid], self.out_grid).run()
        for x in self.out_grid.interior_points():
            self.assertEqual(self.out_grid[x], 7)
        for x in self.out_grid.border_points():
            self.assertEqual(self.out_grid[x], 0)

    def test_function_expr_index(self):
        self.helper_grid = StencilGrid([5])
        self.helper_grid.data[1] = 7  # All elements of in_grid are set to 1
        model = StencilModel([Identifier('in_grid'), Identifier('helper_grid')],
                             Kernel([OutputAssignment(InputElementExprIndex(Identifier('helper_grid'),
                                                                            InputElementZeroOffset(Identifier('in_grid'))))]),
                             Kernel([]))

        StencilModelInterpreter(model, [self.in_grid, self.helper_grid], self.out_grid).run()
        for x in self.out_grid.interior_points():
            self.assertEqual(self.out_grid[x], 7)
        for x in self.out_grid.border_points():
            self.assertEqual(self.out_grid[x], 0)

    def test_function_abs(self):
        model = StencilModel([Identifier('in_grid')],
                             Kernel([OutputAssignment(MathFunction('abs', [Constant(-2)]))]),
                             Kernel([]))

        StencilModelInterpreter(model, [self.in_grid], self.out_grid).run()
        for x in self.out_grid.interior_points():
            self.assertEqual(self.out_grid[x], 2)
        for x in self.out_grid.border_points():
            self.assertEqual(self.out_grid[x], 0)

    def test_function_int(self):
        model = StencilModel([Identifier('in_grid')],
                             Kernel([OutputAssignment(MathFunction('int', [Constant(2.5)]))]),
                             Kernel([]))

        StencilModelInterpreter(model, [self.in_grid], self.out_grid).run()
        for x in self.out_grid.interior_points():
            self.assertEqual(self.out_grid[x], 2)
        for x in self.out_grid.border_points():
            self.assertEqual(self.out_grid[x], 0)

    def test_parse_distance(self):
        model = StencilModel([Identifier('in_grid')],
                             Kernel([StencilNeighborIter(Identifier('in_grid'),
                                                         Constant(1),
                                                         [OutputAssignment(ScalarBinOp(OutputElement(), ast.Add(), NeighborDistance()))])]),
                             Kernel([]))
        
        self.in_grid.data[(1,1)] = 5  # Make it not just ones
        StencilModelInterpreter(model, [self.in_grid], self.out_grid).run()
        
        for x in self.out_grid.interior_points():
            self.assertEqual(self.out_grid[x], 4)
        for x in self.out_grid.border_points():
            self.assertEqual(self.out_grid[x], 0)

if __name__ == '__main__':
    unittest.main()
