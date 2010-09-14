import unittest
from stencil_kernel import *

class BasicTests(unittest.TestCase):
	def test_init(self):
		# if no kernel method is defined, it should fail
		self.failUnlessRaises((Exception), StencilKernel)
	
	def test_pure_python(self):
		class MyKernel(StencilKernel):
			def kernel(self, in_grid, out_grid):
				print "Running kernel...\n"
				for x in out_grid.interior_points():
					out_grid[x] = in_grid[x]


		kernel = MyKernel()
		in_grid = StencilGrid([10,10])
		out_grid = StencilGrid([10,10])
		kernel.pure_python = True
		kernel.kernel(in_grid, out_grid)
		self.failIf(in_grid[3,3] != out_grid[3,3])


class IntrospectionTests(unittest.TestCase):
	def setUp(self):
		class MyKernel(StencilKernel):
			def kernel(self, in_grid, out_grid):
				for x in out_grid.interior_points():
					for y in in_grid.neighbors(x, 1):
						out_grid[x] += in_grid[y]


		self.kernel = MyKernel()
		self.in_grid = StencilGrid([10,10])
		self.out_grid = StencilGrid([10,10])
	
	def test_get_kernel_body(self):
		self.failIfEqual(self.kernel.kernel_ast, None)

	def test_simple_kernel(self):
		
		self.kernel.kernel(self.in_grid, self.out_grid)

if __name__ == '__main__':
	unittest.main()
