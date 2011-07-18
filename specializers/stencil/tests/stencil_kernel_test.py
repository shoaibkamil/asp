import unittest2 as unittest
import ast
from stencil_kernel import *
from stencil_python_front_end import *
from stencil_unroll_neighbor_iter import *
from stencil_convert import *
from asp.util import *

class BasicTests(unittest.TestCase):
    def test_init(self):
        # if no kernel method is defined, it should fail
        self.failUnlessRaises((Exception), StencilKernel)
    
    def test_pure_python(self):
        class MyKernel(StencilKernel):
            def kernel(self, in_grid, out_grid):
                for x in out_grid.interior_points():
                    out_grid[x] = in_grid[x]

        kernel = MyKernel()
        in_grid = StencilGrid([10,10])
        out_grid = StencilGrid([10,10])
        kernel.pure_python = True
        kernel.kernel(in_grid, out_grid)
        self.failIf(in_grid[3,3] != out_grid[3,3])

class StencilConvertASTTests(unittest.TestCase):
    def setUp(self):
        class MyKernel(StencilKernel):
            def kernel(self, in_grid, out_grid):
                for x in out_grid.interior_points():
                    for y in in_grid.neighbors(x, 1):
                        out_grid[x] = out_grid[x] + in_grid[y]

        self.kernel = MyKernel()
        self.in_grid = StencilGrid([10,10])
        self.in_grids = [self.in_grid]
        self.out_grid = StencilGrid([10,10])
        self.model = python_func_to_unrolled_model(MyKernel.kernel, self.in_grids, self.out_grid)

    def test_StencilConvertAST_array_macro_use(self):
        import asp.codegen.cpp_ast as cpp_ast
        result = StencilConvertAST(self.model, self.in_grids, self.out_grid).gen_array_macro('in_grid',
                                                                                             [cpp_ast.CNumber(3),
                                                                                              cpp_ast.CNumber(4)])
        self.assertEqual(str(result), "_in_grid_array_macro(3, 4)")

    def test_whole_thing(self):
        import numpy
        self.in_grid.data = numpy.ones([10,10])
        self.kernel.kernel(self.in_grid, self.out_grid)
        self.assertEqual(self.out_grid[5,5],4.0)


class Stencil1dAnd3dTests(unittest.TestCase):
    def setUp(self):
        class My1DKernel(StencilKernel):
            def kernel(self, in_grid_1d, out_grid_1d):
                for x in out_grid_1d.interior_points():
                    for y in in_grid_1d.neighbors(x, 1):
                        out_grid_1d[x] = out_grid_1d[x] + in_grid_1d[y]


        self.kernel = My1DKernel()
        self.in_grid = StencilGrid([10])
        self.in_grids = [self.in_grid]
        self.out_grid = StencilGrid([10])
        self.model = python_func_to_unrolled_model(My1DKernel.kernel, self.in_grids, self.out_grid)
        
    def test_whole_thing(self):
        import numpy
        import numpy
        self.in_grid.data = numpy.ones([10])
        self.kernel.kernel(self.in_grid, self.out_grid)
        self.assertEqual(self.out_grid[4], 2.0)


class VariantTests(unittest.TestCase):
    def test_no_regeneration_if_same_sizes(self):
        class My1DKernel(StencilKernel):
            def kernel(self, in_grid_1d, out_grid_1d):
                for x in out_grid_1d.interior_points():
                    for y in in_grid_1d.neighbors(x, 1):
                        out_grid_1d[x] = out_grid_1d[x] + in_grid_1d[y]


        kernel = My1DKernel()
        in_grid = StencilGrid([10])
        out_grid = StencilGrid([10])

        kernel.kernel(in_grid, out_grid)
        saved = kernel.mod
        kernel.kernel(in_grid, out_grid)
        self.assertEqual(saved, kernel.mod)
        

class StencilConvertASTCilkTests(unittest.TestCase):
    def setUp(self):
        class MyKernel(StencilKernel):
            def kernel(self, in_grid, out_grid):
                for x in out_grid.interior_points():
                    for y in in_grid.neighbors(x, 1):
                        out_grid[x] = out_grid[x] + in_grid[y]


        self.kernel = MyKernel()
        self.in_grid = StencilGrid([10,10])
        self.out_grid = StencilGrid([10,10])
        self.argdict = argdict = {'in_grid': self.in_grid, 'out_grid': self.out_grid}

def python_func_to_unrolled_model(func, in_grids, out_grid):
    python_ast = ast.parse(inspect.getsource(func).lstrip())
    model = StencilPythonFrontEnd().parse(python_ast)
    return StencilUnrollNeighborIter(model, in_grids, out_grid).run()

if __name__ == '__main__':
    unittest.main()
