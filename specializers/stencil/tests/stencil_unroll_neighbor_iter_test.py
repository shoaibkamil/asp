import unittest2 as unittest
from stencil_unroll_neighbor_iter import *
from stencil_model_interpreter import *
from stencil_grid import *
import numpy
import ast

class BasicTests(unittest.TestCase):
    def test_identity(self):
        in_grid = StencilGrid([5,5])
        in_grid.data = numpy.ones([5,5])
        out_grid = StencilGrid([5,5])
        model = StencilModel([Identifier('in_grid')],
                             Kernel([OutputAssignment(InputElementZeroOffset(Identifier('in_grid')))]),
                             Kernel([]))

        StencilUnrollNeighborIter(model, [in_grid], out_grid).run()
        StencilModelInterpreter(model, [in_grid], out_grid).run()
        for x in out_grid.interior_points():
            assert out_grid[x] == 1
        for x in out_grid.border_points():
            assert out_grid[x] == 0

    def test_add_neighbors(self):
        in_grid = StencilGrid([5,5])
        in_grid.data = numpy.ones([5,5])
        out_grid = StencilGrid([5,5])
        model = StencilModel([Identifier('in_grid')],
                             Kernel([StencilNeighborIter(Identifier('in_grid'),
                                                         Constant(1),
                                                         [OutputAssignment(ScalarBinOp(OutputElement(), ast.Add(), Neighbor()))])]),
                             Kernel([]))

        StencilUnrollNeighborIter(model, [in_grid], out_grid).run()
        StencilModelInterpreter(model, [in_grid], out_grid).run()
        for x in out_grid.interior_points():
            assert out_grid[x] == 4
        for x in out_grid.border_points():
            assert out_grid[x] == 0

if __name__ == '__main__':
    unittest.main()
