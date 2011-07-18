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

if __name__ == '__main__':
    unittest.main()
