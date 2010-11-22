import unittest
from stencil_kernel import *
from asp.util import *

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


		
	def test_StencilInteriorIter_and_StencilNeighborIter(self):
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
						out_grid[x] = out_grid[x] + in_grid[y]


		self.kernel = MyKernel()
		self.in_grid = StencilGrid([10,10])
		self.out_grid = StencilGrid([10,10])
		self.argdict = argdict = {'in_grid': self.in_grid, 'out_grid': self.out_grid}


	def test_StencilConvertAST_array_macro(self):
		import re
		
		result = StencilKernel.StencilConvertAST(self.argdict).gen_array_macro_definition('in_grid')

		self.assertTrue(re.search("array_macro", str(result)))
		self.assertTrue(re.search("#define", str(result)))

	def test_StencilConvertAST_array_macro_use(self):
		result = StencilKernel.StencilConvertAST(self.argdict).gen_array_macro('in_grid', [3,4])
		self.assertEqual(result, "_in_grid_array_macro(3,4)")

	def test_StencilConvertAST_array_replacement(self):
		import asp.codegen.python_ast as ast
		return True
		n = ast.Subscript(ast.Name("grid", None), ast.Index(ast.Num(1)), None)
		result = StencilKernel.StencilConvertAST(self.argdict).visit(n)
		self.assertEqual(str(result), "_my_grid[1]")


	def test_StencilConvertAST_array_unpack_to_double(self):
		result = StencilKernel.StencilConvertAST(self.argdict).gen_array_unpack()
		self.assertEqual(result, "double* _my_out_grid = (double *) PyArray_DATA(out_grid);\n" +
			"double* _my_in_grid = (double *) PyArray_DATA(in_grid);")

	def test_visit_StencilInteriorIter(self):
		import asp.codegen.python_ast as ast, re
		
		n = StencilKernel.StencilInteriorIter("in_grid",
						      [ast.Pass()],
						      ast.Name("targ", None))
		result = StencilKernel.StencilConvertAST(self.argdict).visit(n)
		debug_print(str(result))
		self.assertTrue(re.search("For", str(type(result))))
	
	def test_visit_StencilNeighborIter(self):
		import asp.codegen.python_ast as ast, re
		n = StencilKernel.StencilNeighborIter("in_grid",
						      [ast.parse("in_grid[x] = in_grid[x] + out_grid[y]").body[0]],
						      ast.Name("y", None),
						      1)
		converter = StencilKernel.StencilConvertAST(self.argdict)
		# visit_StencilNeighborIter expects to have dim vars defined already
		converter.gen_dim_var()
		converter.gen_dim_var()
		result = converter.visit(n)
		self.assertTrue(re.search("array_macro", str(result)))

	def test_whole_thing(self):

		import numpy
		self.in_grid.data = numpy.ones([10,10])

		print self.in_grid.data
		
		self.kernel.kernel(self.in_grid, self.out_grid)
		
		print self.out_grid.data
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
		self.out_grid = StencilGrid([10])
		self.argdict =  {'in_grid_1d': self.in_grid, 'out_grid_1d': self.out_grid}
		
	def test_1d_gen_array_macro_definition(self):
		result = StencilKernel.StencilConvertAST(self.argdict).gen_array_macro_definition('in_grid_1d')
		self.assertEqual(result.__str__(), "#define _in_grid_1d_array_macro(_d0) (_d0)")


	def test_1d_visit_StencilInteriorIter(self):

		import asp.codegen.python_ast as ast, re
		n = StencilKernel.StencilInteriorIter("in_grid_1d",
						      [ast.Pass()],
						      ast.Name("targ", None))
		result = StencilKernel.StencilConvertAST(self.argdict).visit(n)

		self.assertTrue(re.search("For", str(type(result))))

	def test_whole_thing(self):
		import numpy
		import numpy
		self.in_grid.data = numpy.ones([10])
		self.kernel.kernel(self.in_grid, self.out_grid)
		print self.out_grid.data
		self.assertEqual(self.out_grid[4], 2.0)


if __name__ == '__main__':
	unittest.main()
