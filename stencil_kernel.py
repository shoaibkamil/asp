import numpy
import inspect
from stencil_grid import *
import simple_ast
import ast

# Overall flow: StencilProcessAST (now has stencil node) ---> StencilToCAST (now is a C++ AST) ---> codegen


# may want to make this inherit from something else...
class StencilKernel(object):
	def __init__(self):
		# we want to raise an exception if there is no kernel()
		# method defined.
		try:
			dir(self).index("kernel")
		except ValueError:
			raise Exception("No kernel method defined.")

		# if the method is defined, let us introspect and find
		# its AST
		import ast 
		self.kernel_src = inspect.getsource(self.kernel)
		self.kernel_ast = ast.parse(self.remove_indentation(self.kernel_src))

		self.pure_python = False
		self.pure_python_kernel = self.kernel

		# replace kernel with shadow version
		self.kernel = self.shadow_kernel
		

	def remove_indentation(self, src):
		return src.lstrip()

	def shadow_kernel(self, *args):
		if self.pure_python:
			return self.pure_python_kernel(*args)


		#FIXME: need to somehow match arg names to args
		argnames = map(lambda x: str(x.id), self.kernel_ast.body[0].args.args)
		argdict = dict(zip(argnames[1:], args))
		print argdict
		#cg = self.StencilCodegen(argdict)
		phase2 = StencilKernel.StencilProcessAST(argdict).visit(self.kernel_ast)
		#print ast.dump(StencilKernel.StencilProcessAST(argdict).visit(self.kernel_ast))
		#print ast.dump(StencilKernel.)
		print ast.dump(phase2)
		phase3 = StencilKernel.StencilConvertAST(argdict).visit(phase2)

	# the actual Stencil AST Node
	class StencilInteriorIter(ast.AST):
		def __init__(self, grid, body, target):
		  self.grid = grid
		  self.body = body
		  self.target = target
		  self._fields = ('grid', 'body', 'target')

		  super(StencilKernel.StencilInteriorIter, self).__init__()
			
	class StencilNeighborIter(ast.AST):
		def __init__(self, grid, body, target, dist):
			self.grid = grid
			self.body = body
			self.target = target
			self.dist = dist
			self._fields = ('grid', 'body', 'target', 'dist')
			super (StencilKernel.StencilNeighborIter, self).__init__()


	# class to convert from Python AST to an AST with special Stencil node
	class StencilProcessAST(ast.NodeTransformer):
		def __init__(self, argdict):
			self.argdict = argdict
			super(StencilKernel.StencilProcessAST, self).__init__()

		
		def visit_For(self, node):
			print "visiting a For...\n"
			# check if this is the right kind of For loop
			if (node.iter.__class__.__name__ == "Call" and
				node.iter.func.__class__.__name__ == "Attribute"):
				
				print "Found something to change...\n"

				if (node.iter.func.attr == "interior_points"):
					grid = self.visit(node.iter.func.value)	   # do we need the name of the grid, or the obj itself?
					target = self.visit(node.target)
					body = map(self.visit, node.body)
					newnode = StencilKernel.StencilInteriorIter(grid, body, target)
					return newnode

				elif (node.iter.func.attr == "neighbors"):
					print ast.dump(node) + "\n"
					target = self.visit(node.target)
					body = map(self.visit, node.body)
					grid = self.visit(node.iter.func.value)
					dist = self.visit(node.iter.args[0])
					newnode = StencilKernel.StencilNeighborIter(grid, body, target, dist)
					return newnode

				else:
					return node
			else:
				return node

	import codegen
	class StencilConvertAST(codegen.ConvertAST):

		def __init__(self, argdict):
			self.argdict = argdict
			super(StencilKernel.StencilConvertAST, self).__init__()

		def gen_array_macro(self, arg):
			import codepy
			try:
				array = self.argdict[arg]
				if  array.dim == 2:
					return codepy.cgen.Define("_"+arg+"_array_macro(_a,_b)", 
								  "(_b+(_a*" + str(array.shape[0]) +
								  "))")
			except KeyError:
				return codepy.cgen.Comment("Not found argument: " + arg)

		def visit_StencilInteriorIter(self, node):
			import codepy, codegen
			# should catch KeyError here
			array = self.argdict[node.grid]
			dim = len(array.shape)
			if dim == 2:
				start1 = codepy.cgen.Assign(codepy.cgen.Value("int", "i"),
							    codegen.CNumber(array.ghost_depth))
				condition1 = "i < " + str(array.shape[0]-array.ghost_depth)
				update1 = "i++"
				start2 = codepy.cgen.Assign(codepy.cgen.Value("int", "j"),
							    codegen.CNumber(array.ghost_depth))
				condition2 = "j < " + str(array.shape[1]-array.ghost_depth)
				update2 = "j++"

				body = codepy.cgen.Block([self.visit(x) for x in node.body])

				return codepy.cgen.For(start1, condition1, update1,
						       codepy.cgen.For(start2, condition2, update2,
								       body))

		def visit_StencilNeighborIter(self, node):
			import codegen

			return codegen.Expression()



#         def visit_For(self, node):
            
#             if (node.iter.__class__.__name__ == "Call" and
#                 node.iter.func.__class__.__name__ == "Attribute"):

#                 if (node.iter.func.attr == "interior_points"):
#                     grid_shape =  eval(self.visit(node.iter.func.value) + ".shape", self.argdict)
#                     grid_dim = len(grid_shape)
#                     target = self.visit(node.target)
                    
#                     if grid_dim == 2:

#                         i1 = self.gensym()
#                         i2 = self.gensym()
#                         self.grid_vars = [i1,i2]
#                         str = "\nfor (int %s=1; %s < %d; %s++){\n " % (i1,i1,grid_shape[0],i1)
#                         str += "for (int %s=1; %s < %d; %s++){\n " % (i2,i2,grid_shape[0],i2)
#                         # now let iterator var = proper index
#                         str += "int " + target + " = _INDEX(" + ','.join(self.grid_vars) + ");\n"
#                         str += ';'.join(map(self.visit, node.body))
#                         str += ";} }"
                        
#                         return str
#                 if (node.iter.func.attr == "neighbors"):
#                     # read the neighbors out
#                     return super(StencilKernel.StencilCodegen, self).vist_For(node)

#             return super(StencilKernel.StencilCodegen, self).visit_For(node)






