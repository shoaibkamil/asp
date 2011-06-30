import unittest2 as unittest
import ast
from stencil_model import *

class AssertTypeTests(unittest.TestCase):
    def test_assert_has_type_right_type(self):
        assert_has_type(2, IntType)
        assert_has_type(2, object)

    def test_assert_has_type_wrong_type(self):
        with self.assertRaises(AssertionError):
            assert_has_type(2, StringType)

    def test_assert_has_type_list_right_type(self):
        assert_has_type(2, [IntType, StringType])
        assert_has_type('hello', [IntType, StringType])
        assert_has_type([2], [IntType, StringType, object])

    def test_assert_has_type_list_wrong_type(self):
        with self.assertRaises(AssertionError):
            assert_has_type([2], [IntType, StringType])

    def test_assert_is_list_of_not_list(self):
        with self.assertRaises(AssertionError):
            assert_is_list_of(2, object)

    def test_assert_is_list_of_right_type(self):
        assert_is_list_of([2, 3, 4], IntType)

    def test_assert_is_list_of_wrong_type(self):
        with self.assertRaises(AssertionError):
            assert_has_type([2, 'string', 4], IntType)

    def test_assert_is_list_of_list_right_type(self):
        assert_is_list_of([2, 'string', 4], [IntType, StringType])
        assert_is_list_of([2, 'string', [2], 4], [IntType, StringType, object])

    def test_assert_is_list_of_list_wrong_type(self):
        with self.assertRaises(AssertionError):
            assert_has_type([2, 'string', [2], 4], [IntType, StringType])

class BasicTests(unittest.TestCase):
    def setUp(self):
        self.in_grid = Identifier('in_grid')
        self.in_grid_2 = Identifier('in_grid_2')
        self.neighbor = Neighbor()
        self.expr_neighbor = Expr(self.neighbor)
        self.output_element = OutputElement()
        self.expr_output_element = Expr(self.output_element)
        self.scalar_bin_op = ScalarBinOp(self.expr_output_element, ast.Add, self.expr_neighbor)
        self.expr_scalar_bin_op = Expr(self.scalar_bin_op)
        self.output_assignment = OutputAssignment(self.expr_scalar_bin_op)
        self.neighbor_iter = StencilNeighborIter(self.in_grid, 1, [self.output_assignment])
        self.kernel = Kernel([self.neighbor_iter])
        self.empty_kernel = Kernel([])
        self.model = StencilModel([self.in_grid], self.kernel, self.empty_kernel)

        self.constant = Constant(2)
        self.expr_constant = Expr(self.constant)
        self.constant_output_assignment = OutputAssignment(self.expr_constant)
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

    def test_kernel_init(self):
        Kernel([self.neighbor_iter])
        Kernel([self.constant_output_assignment])
        Kernel([self.constant_output_assignment, self.constant_output_assignment])
        Kernel([self.neighbor_iter, self.neighbor_iter])
        Kernel([self.neighbor_iter, self.constant_output_assignment])

    def test_stencil_neighbor_init(self):
        for distance in range(0,10):
            StencilNeighborIter(self.in_grid, distance, [self.output_assignment])
        with self.assertRaises(AssertionError):
            StencilNeighborIter(self.in_grid, -10, [self.output_assignment])
        # Empty neighbor loop allowed
        StencilNeighborIter(self.in_grid, distance, [])

    def test_expr_init(self):
        Expr(self.constant)
        Expr(self.neighbor)
        Expr(self.output_element)
        Expr(self.input_element)
        Expr(self.scalar_bin_op)
        with self.assertRaises(AssertionError):
            Expr(2)

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
        ScalarBinOp(self.expr_constant, ast.Add, self.expr_constant)
        ScalarBinOp(self.expr_output_element, ast.Add, self.expr_neighbor)
        ScalarBinOp(self.expr_output_element, ast.Sub, self.expr_neighbor)
        ScalarBinOp(self.expr_output_element, ast.Mult, self.expr_neighbor)
        ScalarBinOp(self.expr_output_element, ast.Div, self.expr_neighbor)
        ScalarBinOp(self.expr_output_element, ast.FloorDiv, self.expr_neighbor)
        for op in [ast.Mod, ast.Pow, ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd]:
            with self.assertRaises(AssertionError):
                ScalarBinOp(self.expr_output_element, op, self.expr_neighbor)

if __name__ == '__main__':
    unittest.main()
