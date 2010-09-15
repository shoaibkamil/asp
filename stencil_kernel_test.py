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





class StencilProcessASTTests(unittest.TestCase):
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

		
	def test__StencilInteriorIter_and_StencilNeighborIter(self):
		import re
		argdict = {'in_grid': self.in_grid, 'out_grid': self.out_grid}
		output_as_string = ast.dump(StencilKernel.StencilProcessAST(argdict).visit(self.kernel.kernel_ast))
		self.assertTrue(re.search("StencilInteriorIter", output_as_string))
		self.assertTrue(re.search("StencilNeighborIter", output_as_string))


class StencilConvertASTTests(unittest.TestCase):
	def setUp(self):
		class MyKernel(StencilKernel):
			def kernel(self, in_grid, out_grid):
				for x in out_grid.interior_points():
					for y in in_grid.neighbors(x, 1):
						out_grid[x] += in_grid[y]


		self.kernel = MyKernel()
		self.in_grid = StencilGrid([10,10])
		self.out_grid = StencilGrid([10,10])
		self.argdict = argdict = {'in_grid': self.in_grid, 'out_grid': self.out_grid}


	def test_StencilConvertAST_array_macro(self):
		import re
		
		result = StencilKernel.StencilConvertAST(self.argdict).gen_array_macro('in_grid')
		print str(result)
		self.assertTrue(re.search("array_macro", str(result)))
		self.assertTrue(re.search("#define", str(result)))

	def test_StencilConvertAST_interior_points(self):
		import ast, re
		
		n = StencilKernel.StencilInteriorIter("in_grid",
						      [ast.Pass()],
						      ast.Name("targ", None))
		result = StencilKernel.StencilConvertAST(self.argdict).visit(n)
		self.assertTrue(re.search("For", str(type(result))))
			       
		


if __name__ == '__main__':
	unittest.main()
