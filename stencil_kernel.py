import numpy
import inspect
from stencil_grid import *
import simple_ast

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
        #print ast.dump(self.kernel_ast)
        #ast_tools.ASTPrettyPrinter().visit(self.kernel_ast)

        # replace kernel with shadow version
        self.kernel = self.shadow_kernel

    def remove_indentation(self, src):
        return src.lstrip()

    def shadow_kernel(self, *args):
        #FIXME: need to somehow match arg names to args
        argnames = map(lambda x: str(x.id), self.kernel_ast.body[0].args.args)
        argdict = dict(zip(argnames[1:], args))
        cg = self.StencilCodegen(argdict)
        print cg.visit(self.kernel_ast)



    # class to codegen stencils
    import codegen
    class StencilCodegen(codegen.CodeGenerator):
        
        def __init__(self, argdict):
            #FIXME: should support multiple input arrays
            #FIXME: should reverse order?
            self.argdict = argdict
            super(StencilKernel.StencilCodegen, self).__init__()

        
        def gensym(self):
            """ Generates random strings for unique identifiers. """
            import random, string
            return "_i" + "".join(random.sample(string.letters+string.digits, 8))

        def visit_For(self, node):
            
            if (node.iter.__class__.__name__ == "Call" and
                node.iter.func.__class__.__name__ == "Attribute"):

                if (node.iter.func.attr == "interior_points"):
                    grid_shape =  eval(self.visit(node.iter.func.value) + ".shape", self.argdict)
                    grid_dim = len(grid_shape)
                    target = self.visit(node.target)
                    
                    if grid_dim == 2:

                        i1 = self.gensym()
                        i2 = self.gensym()
                        self.grid_vars = [i1,i2]
                        str = "\nfor (int %s=1; %s < %d; %s++){\n " % (i1,i1,grid_shape[0],i1)
                        str += "for (int %s=1; %s < %d; %s++){\n " % (i2,i2,grid_shape[0],i2)
                        # now let iterator var = proper index
                        str += "int " + target + " = _INDEX(" + ','.join(self.grid_vars) + ");\n"
                        str += ';'.join(map(self.visit, node.body))
                        str += ";} }"
                        
                        return str
                if (node.iter.func.attr == "neighbors"):
                    # read the neighbors out
                    return super(StencilKernel.StencilCodegen, self).vist_For(node)

            return super(StencilKernel.StencilCodegen, self).visit_For(node)






