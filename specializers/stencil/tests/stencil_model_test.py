import unittest2 as unittest
import ast
from stencil_model import *
from assert_utils import *

class BasicTests(unittest.TestCase):
    def setUp(self):
        self.in_grid = Identifier('in_grid')
        self.in_grid_2 = Identifier('in_grid_2')
        self.neighbor = Neighbor()
        self.output_element = OutputElement()
        self.scalar_bin_op = ScalarBinOp(self.output_element, ast.Add(), self.neighbor)
        self.output_assignment = OutputAssignment(self.scalar_bin_op)
        self.neighbor_iter = StencilNeighborIter(self.in_grid, Constant(1), [self.output_assignment])
        self.kernel = Kernel([self.neighbor_iter])
        self.empty_kernel = Kernel([])
        self.model = StencilModel([self.in_grid], self.kernel, self.empty_kernel)

        self.constant = Constant(2)
        self.constant_output_assignment = OutputAssignment(self.constant)
        self.constant_kernel = Kernel([self.constant_output_assignment])

        self.input_element = InputElement(self.in_grid, [1, -1])

    def test_identifer_attributes(self):
        self.assertEqual(self.in_grid.name, 'in_grid')

    def test_stencil_model_init(self):
        StencilModel([self.in_grid], self.kernel, self.empty_kernel)
        StencilModel([self.in_grid], self.empty_kernel, self.kernel)
        StencilModel([self.in_grid, self.in_grid_2], self.kernel, self.empty_kernel)
        # Assigns a constant to all elements of output grid, requires no input grid
        StencilModel([], self.constant_kernel, self.constant_kernel)
        # Empty model, does nothing
        StencilModel([], self.empty_kernel, self.empty_kernel)

    def test_stencil_model_bad_neighbor(self):
        with self.assertRaises(AssertionError):
            StencilModel([self.in_grid], Kernel([OutputAssignment(Neighbor())]), self.empty_kernel)

    def test_kernel_init(self):
        Kernel([self.neighbor_iter])
        Kernel([self.constant_output_assignment])
        Kernel([self.constant_output_assignment, self.constant_output_assignment])
        Kernel([self.neighbor_iter, self.neighbor_iter])
        Kernel([self.neighbor_iter, self.constant_output_assignment])

    def test_stencil_neighbor_init(self):
        for distance in range(0,10):
            StencilNeighborIter(self.in_grid, Constant(distance), [self.output_assignment])
        with self.assertRaises(AssertionError):
            StencilNeighborIter(self.in_grid, Constant(-10), [self.output_assignment])
        # Empty neighbor loop allowed
        StencilNeighborIter(self.in_grid, Constant(1), [])

    def test_Expr(self):
        assert_has_type(self.constant, Expr)
        assert_has_type(self.neighbor, Expr)
        assert_has_type(self.output_element, Expr)
        assert_has_type(self.input_element, Expr)
        assert_has_type(self.scalar_bin_op, Expr)
        with self.assertRaises(AssertionError):
            assert_has_type(2, Expr)

    def test_constant_init(self):
        Constant(0)
        Constant(2)
        Constant(-2)
        Constant(2L)
        Constant(2.0)
        Constant(float('inf'))
        Constant(float('-inf'))
        Constant(float('NaN'))

    def test_input_element_init(self):
        # Note, zero length allowed for zero-dimensional (one point) grids
        for count in range(10):
            InputElement(self.in_grid, [1 for x in range(count)])
        InputElement(self.in_grid, [1, -1])
        InputElement(self.in_grid, [-1000000, 1000000])

    def test_scalar_bin_op_init(self):
        ScalarBinOp(self.constant, ast.Add(), self.constant)
        ScalarBinOp(self.output_element, ast.Add(), self.neighbor)
        ScalarBinOp(self.output_element, ast.Sub(), self.neighbor)
        ScalarBinOp(self.output_element, ast.Mult(), self.neighbor)
        ScalarBinOp(self.output_element, ast.Div(), self.neighbor)
        ScalarBinOp(self.output_element, ast.FloorDiv(), self.neighbor)
        ScalarBinOp(self.output_element, ast.Mod(), self.neighbor)
        for op in [ast.Mod, ast.Pow, ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd]:
            with self.assertRaises(AssertionError):
                ScalarBinOp(self.output_element, op, self.neighbor)

if __name__ == '__main__':
    unittest.main()
