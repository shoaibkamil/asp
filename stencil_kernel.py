import numpy
import inspect
from stencil_grid import *
import ast_tools

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
        print self.kernel_src
        self.kernel_ast = ast.parse(self.remove_indentation(self.kernel_src))
        #print ast.dump(self.kernel_ast)
        ast_tools.ASTPrettyPrinter().visit(self.kernel_ast)

    def remove_indentation(self,src):
        return src.lstrip()

        







