import unittest2 as unittest
import ast
import math
import itertools
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
        class IdentityKernel(StencilKernel):
            def kernel(self, in_grid, out_grid):
                for x in out_grid.interior_points():
                    out_grid[x] = in_grid[x]

        kernel = IdentityKernel()
        in_grid = StencilGrid([10,10])
        out_grid = StencilGrid([10,10])
        kernel.pure_python = True
        kernel.kernel(in_grid, out_grid)
        self.failIf(in_grid[3,3] != out_grid[3,3])
class StencilConvertASTTests(unittest.TestCase):
    def setUp(self):
        class IdentityKernel(StencilKernel):
            def kernel(self, in_grid, out_grid):
                for x in out_grid.interior_points():
                    for y in in_grid.neighbors(x, 1):
                        out_grid[x] = out_grid[x] + in_grid[y]

        self.kernel = IdentityKernel()
        self.in_grid = StencilGrid([10,10])
        self.in_grids = [self.in_grid]
        self.out_grid = StencilGrid([10,10])
        self.model = python_func_to_unrolled_model(IdentityKernel.kernel, self.in_grids, self.out_grid)

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
        class IdentityKernel(StencilKernel):
            def kernel(self, in_grid, out_grid):
                for x in out_grid.interior_points():
                    for y in in_grid.neighbors(x, 1):
                        out_grid[x] = out_grid[x] + in_grid[y]


        self.kernel = IdentityKernel()
        self.in_grid = StencilGrid([10,10])
        self.out_grid = StencilGrid([10,10])
        self.argdict = argdict = {'in_grid': self.in_grid, 'out_grid': self.out_grid}

class StencilConvert1DDeriativeTests(unittest.TestCase):
    def setUp(self):
        self.h = 0.01
        self.points = 100
        class DerivativeKernel(StencilKernel):
            def kernel(self, in_grid, out_grid):
                for x in out_grid.interior_points():
                    for y in in_grid.neighbors(x, 0):
                        out_grid[x] += in_grid[y]
                    for y in in_grid.neighbors(x, 1):
                        out_grid[x] -= 8*in_grid[y]
                    for y in in_grid.neighbors(x, 2):
                        out_grid[x] += 8*in_grid[y]
                    for y in in_grid.neighbors(x, 3):
                        out_grid[x] -= in_grid[y]
                    out_grid[x] /= 12 * 0.01

        self.kernel = DerivativeKernel()
        self.in_grid = StencilGrid([self.points])
        self.in_grid.ghost_depth = 2
        self.in_grid.neighbor_definition = [ [(-2,)], [(-1,)], [(1,)], [(2,)] ]
        self.out_grid = StencilGrid([self.points])
        self.out_grid.ghost_depth = 2
        self.expected_out_grid = StencilGrid([self.points])

    def test_whole_thing(self):
        import numpy
        for xi in range(0,self.points):
            x = xi * self.h
            self.in_grid.data[(xi,)] = math.sin(x)
            self.expected_out_grid[(xi,)] = math.cos(x) # Symbolic derivative

        self.kernel.kernel(self.in_grid, self.out_grid)

        for x in self.out_grid.interior_points():
            self.assertAlmostEqual(self.out_grid[x], self.expected_out_grid[x])

class StencilConvert2DLaplacianTests(unittest.TestCase):
    def setUp(self):
        self.h = 0.01
        self.points = 10
        class LaplacianKernel(StencilKernel):
            def kernel(self, in_grid, out_grid):
                for x in out_grid.interior_points():
                    for y in in_grid.neighbors(x, 1):
                        out_grid[x] += in_grid[y]
                    out_grid[x] -= 4*in_grid[x]
                    out_grid[x] /= 0.01 * 0.01

        self.kernel = LaplacianKernel()
        self.in_grid = StencilGrid([self.points,self.points])
        self.out_grid = StencilGrid([self.points,self.points])
        self.expected_out_grid = StencilGrid([self.points,self.points])

    def test_whole_thing(self):
        import numpy
        for xi in range(0,self.points):
            for yi in range(0,self.points):
                x = xi * self.h
                y = yi * self.h
                self.in_grid.data[(xi, yi)] = x**3 + y**3
                self.expected_out_grid[(xi, yi)] = 6*x + 6*y # Symbolic Laplacian

        self.kernel.kernel(self.in_grid, self.out_grid)

        for x in self.out_grid.interior_points():
            self.assertAlmostEqual(self.out_grid[x], self.expected_out_grid[x])

class StencilConvert3DBilateralTests(unittest.TestCase):
    def setUp(self):
        self.points = 10
        class BilateralKernel(StencilKernel):
           def kernel(self, in_img, filter, out_img):
               for x in out_img.interior_points():
                   for y in in_img.neighbors(x, 1):
                       out_img[x] = out_img[x] + filter[abs(int(in_img[x]-in_img[y]))%255] * in_img[y]

        self.filter = StencilGrid([256])
        stdev = 40.0
        mean = 0.0
        scale = 1.0/(stdev*math.sqrt(2.0*math.pi))
        divisor = 1.0 / (2.0 * stdev * stdev)
        for x in xrange(256):
           self.filter[x] = scale * math.exp( -1.0 * (float(x)-mean) * (float(x)-mean) * divisor)

        self.kernel = BilateralKernel()
        # because of the large number of neighbors, unrolling breaks gcc
        self.kernel.should_unroll = False
        self.out_grid = StencilGrid([self.points,self.points,self.points])
        self.out_grid.ghost_depth = 3
        self.expected_out_grid = StencilGrid([self.points,self.points,self.points])
        self.expected_out_grid.ghost_depth = 3

        self.in_grid = StencilGrid([self.points,self.points,self.points])
        self.in_grid.ghost_depth = 3
        # set neighbors to be everything within -3 to 3 of each 3D point in each direction
        self.in_grid.neighbor_definition[1] = list(
            set([x for x in itertools.permutations([-1,-1,-1,-2,-2,-2,-3,-3,-3,0,0,0,1,1,1,2,2,2,3,3,3],3)]))

    def test_whole_thing(self):
        import numpy
        for x in range(0,self.points):
            for y in range(0,self.points):
                for z in range(0,self.points):
                    self.in_grid.data[(x, y, z)] = (x + y + z) % 256

        self.kernel.kernel(self.in_grid, self.filter, self.out_grid)
        self.kernel.pure_python = True
        self.kernel.kernel(self.in_grid, self.filter, self.expected_out_grid)

        for x in self.out_grid.interior_points():
            self.assertAlmostEqual(self.out_grid[x], self.expected_out_grid[x])

class StencilDistanceTests(unittest.TestCase):
    def setUp(self):
        class DistanceKernel(StencilKernel):
            def kernel(self, in_grid, out_grid):
                for x in out_grid.interior_points():
                    for y in in_grid.neighbors(x, 1):
                        out_grid[x] += distance(x,y)
                        out_grid[x] += distance(y,x)
                        out_grid[x] += distance(x,x)
                        out_grid[x] += distance(y,y)

        self.kernel = DistanceKernel()
        self.in_grid = StencilGrid([10,10])
        self.in_grid.neighbor_definition[1] = [(0,1), (1,0), (1,1), (0,2), (1,2)]
        self.in_grids = [self.in_grid]
        self.out_grid = StencilGrid([10,10])
        self.out_grid.ghost_depth = 2
        self.model = python_func_to_unrolled_model(DistanceKernel.kernel, self.in_grids, self.out_grid)
        
    def test_whole_thing(self):
        import numpy
        self.in_grid.data = numpy.ones([10,10])
        self.kernel.kernel(self.in_grid, self.out_grid)
        for x in self.out_grid.interior_points():
            self.assertAlmostEqual(self.out_grid[x], 2 * (1 + 1 + math.sqrt(2) + 2 + math.sqrt(5)))

def python_func_to_unrolled_model(func, in_grids, out_grid):
    python_ast = ast.parse(inspect.getsource(func).lstrip())
    model = StencilPythonFrontEnd().parse(python_ast)
    return StencilUnrollNeighborIter(model, in_grids, out_grid).run()

if __name__ == '__main__':
    unittest.main()
