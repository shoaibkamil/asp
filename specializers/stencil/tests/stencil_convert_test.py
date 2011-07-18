import unittest2 as unittest
from stencil_grid import *
from stencil_model import *
from stencil_convert import *
import numpy
import ast

class AddNeighborsBasicTest(unittest.TestCase):
    def setUp(self):
        self.in_grid = StencilGrid([10,10])
        self.in_grid.data = numpy.ones([10,10])
        self.out_grid = StencilGrid([10,10])
        self.model_identity =
            StencilModel([Identifier('in_grid')],
                          Kernel([OutputAssignment(InputElementZeroOffset(Identifier('in_grid')))]),
                          Kernel([]))
        self.model_add_neighbors =
            StencilModel([Identifier('in_grid')],
                          Kernel([OutputAssignment(ScalarBinOp(OutputElement(), ast.Add(), InputElement(Identifier('in_grid'),  [1, 0]))),
                                  OutputAssignment(ScalarBinOp(OutputElement(), ast.Add(), InputElement(Identifier('in_grid'), [-1, 0]))),
                                  OutputAssignment(ScalarBinOp(OutputElement(), ast.Add(), InputElement(Identifier('in_grid'),  [0, 1]))),
                                  OutputAssignment(ScalarBinOp(OutputElement(), ast.Add(), InputElement(Identifier('in_grid'),  [0,-1])))]),
                          Kernel([]))

    def test_single_threaded(self):
        for unroll_count in range(1, 10):
            StencilConvertAST(self.model_add_neighbors, [self.in_grid], self.out_grid, unroll_count).run()

    def test_cilk(self):
        for unroll_count in range(1, 10):
            StencilConvertASTCilk(self.model_add_neighbors, [self.in_grid], self.out_grid, unroll_count).run()

    def test_convert_InputElementZeroOffset(self):
        for unroll_count in [1, 7]: # Just 1 and 7
            StencilConvertAST(self.model_identity, [self.in_grid], self.out_grid, 1).run()

if __name__ == '__main__':
    unittest.main()
