import numpy
import inspect
from stencil_grid import *
import asp.codegen.python_ast as ast
import asp.codegen.cpp_ast as cpp_ast
import asp.codegen.ast_tools as ast_tools
from asp.util import *

# may want to make this inherit from something else...
class StencilKernel(object):
    def __init__(self, with_cilk=False):
        # we want to raise an exception if there is no kernel()
        # method defined.
        try:
            dir(self).index("kernel")
        except ValueError:
            raise Exception("No kernel method defined.")

        # if the method is defined, let us introspect and find
        # its AST
        self.kernel_src = inspect.getsource(self.kernel)
        self.kernel_ast = ast.parse(self.remove_indentation(self.kernel_src))

        self.pure_python = False
        self.pure_python_kernel = self.kernel

        # replace kernel with shadow version
        self.kernel = self.shadow_kernel

        self.specialized_sizes = None
        self.with_cilk = with_cilk

    def remove_indentation(self, src):
        return src.lstrip()

    def add_libraries(self, mod):
        # these are necessary includes, includedirs, and init statements to use the numpy library
        mod.add_library("numpy",[numpy.get_include()+"/numpy"])
        mod.add_header("arrayobject.h")
        mod.add_to_init([cpp_ast.Statement("import_array();")])
        if self.with_cilk:
            mod.module.add_to_preamble([cpp_ast.Include("cilk/cilk.h", True)])
        

    def shadow_kernel(self, *args):
        if self.pure_python:
            return self.pure_python_kernel(*args)

        #FIXME: instead of doing this short-circuit, we should use the Asp infrastructure to
        # do it, by passing in a lambda that does this check
        # if already specialized to these sizes, just run
        if self.specialized_sizes and self.specialized_sizes == [y.shape for y in args]:
            print "match!"
            self.mod.kernel(*[y.data for y in args])
            return

        # otherwise, do the first-run flow

        # check if we can specialize for this data
        #FIXME: impelement.

        # ask asp infrastructure for machine and platform info, including if cilk+ is available
        #FIXME: impelement.  set self.with_cilk=true if cilk is available
        
        

        #FIXME: need to somehow match arg names to args
        argnames = map(lambda x: str(x.id), self.kernel_ast.body[0].args.args)
        argdict = dict(zip(argnames[1:], args))
        debug_print(argdict)

        phase2 = StencilKernel.StencilProcessAST(argdict).visit(self.kernel_ast)
        debug_print(ast.dump(phase2))

        
        # depending on whether cilk is available, we choose which converter to use
        if not self.with_cilk:
            Converter = StencilKernel.StencilConvertAST
        else:
            Converter = StencilKernel.StencilConvertASTCilk

        # generate variant with no unrolling, then generate variants for various unrollings
        variants = [Converter(argdict).visit(phase2)]
        variant_names = ["kernel_unroll_1"]
        for x in [2,4,8,16,32,64]:
            check_valid = max(map(
                lambda y: (y.shape[-1]-2*y.ghost_depth) % x,
                args))
            if check_valid == 0:
                to_append = Converter(argdict, unroll_factor=x).visit(phase2)
                variants.append(Converter(argdict, unroll_factor=x).visit(phase2))
                variant_names.append("kernel_unroll_%s" % x)

        from asp.jit import asp_module

        mod = self.mod = asp_module.ASPModule()
        self.add_libraries(mod)
        if self.with_cilk:
            mod.toolchain.cc = "icc"
            mod.toolchain.cflags += ["-intel-extensions", "-fast"]
            mod.toolchain.cflags += ["-I/usr/include/x86_64-linux-gnu"]
            mod.toolchain.cflags.remove('-fwrapv')
        else:
            mod.toolchain.cflags += ["-fopenmp", "-O3", "-msse3"]
#        print mod.toolchain.cflags
        if mod.toolchain.cflags.count('-Os') > 0:
            mod.toolchain.cflags.remove('-Os')
        if mod.toolchain.cflags.count('-O2') > 0:
            mod.toolchain.cflags.remove('-O2')
        debug_print("toolchain" + str(mod.toolchain.cflags))
        mod.add_function_with_variants(variants, "kernel", variant_names)

        # package arguments
        myargs = [y.data for y in args]
        print myargs
        # and do the call 
        mod.kernel(*myargs)

        # save parameter sizes for next run
        self.specialized_sizes = [x.shape for x in args]

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


    # separate files for different architectures
    # class to convert from Python AST to an AST with special Stencil node
    class StencilProcessAST(ast.NodeTransformer):
        def __init__(self, argdict):
            self.argdict = argdict
            super(StencilKernel.StencilProcessAST, self).__init__()

        
        def visit_For(self, node):
            debug_print("visiting a For...\n")
            # check if this is the right kind of For loop
            if (node.iter.__class__.__name__ == "Call" and
                node.iter.func.__class__.__name__ == "Attribute"):
                
                debug_print("Found something to change...\n")

                if (node.iter.func.attr == "interior_points"):
                    grid = self.visit(node.iter.func.value).id     # do we need the name of the grid, or the obj itself?
                    target = self.visit(node.target)
                    body = map(self.visit, node.body)
                    newnode = StencilKernel.StencilInteriorIter(grid, body, target)
                    return newnode

                elif (node.iter.func.attr == "neighbors"):
                    debug_print(ast.dump(node) + "\n")
                    target = self.visit(node.target)
                    body = map(self.visit, node.body)
                    grid = self.visit(node.iter.func.value).id
                    dist = self.visit(node.iter.args[1]).n
                    newnode = StencilKernel.StencilNeighborIter(grid, body, target, dist)
                    return newnode

                else:
                    return node
            else:
                return node

    class StencilConvertAST(ast_tools.ConvertAST):
        
        def __init__(self, argdict, unroll_factor=None):
            self.argdict = argdict
            self.dim_vars = []
            self.unroll_factor = unroll_factor
            super(StencilKernel.StencilConvertAST, self).__init__()

        def gen_array_macro_definition(self, arg):
            array = self.argdict[arg]
            defname = "_"+arg+"_array_macro"
            params = "(" + ','.join(["_d"+str(x) for x in xrange(array.dim)]) + ")"
            calc = "(_d%d" % (array.dim-1)
            for x in range(0,array.dim-1):
                calc += "+(_d%s * %s)" % (str(x), str(array.data.strides[x]/array.data.itemsize))
            calc += ")"
            return cpp_ast.Define(defname+params, calc)


        def gen_array_macro(self, arg, point):
            name = "_%s_array_macro" % arg
            #param = [cpp_ast.CNumber(x) for x in point]
            return cpp_ast.Call(cpp_ast.CName(name), point)



        def gen_dim_var(self):
            import random
            import string
            var = "_" + ''.join(random.choice(string.letters) for i in xrange(8))
            self.dim_vars.append(var)
            return var

        def gen_array_unpack(self):
            ret =  [cpp_ast.Assign(cpp_ast.Pointer(cpp_ast.Value("npy_double", "_my_"+x)), 
                    cpp_ast.TypeCast(cpp_ast.Pointer(cpp_ast.Value("npy_double", "")), cpp_ast.FunctionCall(cpp_ast.CName("PyArray_DATA"), params=[cpp_ast.CName(x)])))
                    for x in self.argdict.keys()]

            return ret
        
        # all arguments are PyObjects
        def visit_arguments(self, node):
            return [cpp_ast.Pointer(cpp_ast.Value("PyObject", self.visit(x))) for x in node.args[1:]]

        # we want to rename the function based on the optimization parameters
        def visit_FunctionDef(self, node):
            
            if not self.unroll_factor:
                new_name = "%s_unroll_1" % node.name
            else:
                new_name = "%s_unroll_%s" % (node.name, self.unroll_factor)
            new_node = ast.FunctionDef(new_name, node.args, node.body, node.decorator_list)
            return super(StencilKernel.StencilConvertAST, self).visit_FunctionDef(new_node)

        def gen_loops(self, node):
            # should catch KeyError here
            array = self.argdict[node.grid]
            dim = len(array.shape)

            ret_node = None
            cur_node = None

            for d in xrange(dim):
                dim_var = self.gen_dim_var()

                initial = cpp_ast.CNumber(array.ghost_depth)
                end = cpp_ast.CNumber(array.shape[d]-array.ghost_depth-1)
                increment = cpp_ast.CNumber(1)
                if d == 0:
                    ret_node = cpp_ast.For(dim_var, initial, end, increment, cpp_ast.Block())
                    cur_node = ret_node
                elif d == dim-2:
                    pragma = cpp_ast.Pragma("omp parallel for")
                    for_node = cpp_ast.For(dim_var, initial, end, increment, cpp_ast.Block())
                    cur_node.body = cpp_ast.Block(contents=[pragma, for_node])
                    cur_node = for_node
                else:
                    cur_node.body = cpp_ast.For(dim_var, initial, end, increment, cpp_ast.Block())
                    cur_node = cur_node.body

            return (cur_node, ret_node)

        def visit_StencilInteriorIter(self, node):

            cur_node, ret_node = self.gen_loops(node)

            body = cpp_ast.Block()
            body.extend([self.gen_array_macro_definition(x) for x in self.argdict])


            body.extend(self.gen_array_unpack())
            
            body.append(cpp_ast.Value("int", self.visit(node.target)))
            body.append(cpp_ast.Assign(self.visit(node.target),
                                       self.gen_array_macro(
                                           node.grid, [cpp_ast.CName(x) for x in self.dim_vars])))


            

            replaced_body = None
            for gridname in self.argdict.keys():
                replaced_body = [ast_tools.ASTNodeReplacer(
                                ast.Name(gridname, None), ast.Name("_my_"+gridname, None)).visit(x) for x in node.body]
            body.extend([self.visit(x) for x in replaced_body])

            
            cur_node.body = body

            # unroll
            if self.unroll_factor:
                replacement = ast_tools.LoopUnroller().unroll(cur_node, self.unroll_factor)
                ret_node = ast_tools.ASTNodeReplacer(cur_node, replacement).visit(ret_node)
            
            return ret_node

        def visit_StencilNeighborIter(self, node):

            block = cpp_ast.Block()
            target = self.visit(node.target)
            block.append(cpp_ast.Value("int", target))
                     
            grid = self.argdict[node.grid]
            debug_print(node.dist)
            for n in grid.neighbor_definition[node.dist]:
                block.append(cpp_ast.Assign(
                    target,
                    self.gen_array_macro(node.grid,
                                         map(lambda x,y: cpp_ast.BinOp(cpp_ast.CName(x), "+", cpp_ast.CNumber(y)),
                                             self.dim_vars,
                                             n))))

                block.extend( [self.visit(z) for z in node.body] )
                

            debug_print(block)
            return block

    class StencilConvertASTCilk(StencilConvertAST):
        class CilkFor(cpp_ast.For):
            def intro_line(self):
                return "cilk_for (%s; %s; %s += %s)" % (self.start, self.condition, self.loopvar, self.increment)

        def gen_loops(self, node):
            # should catch KeyError here
            array = self.argdict[node.grid]
            dim = len(array.shape)

            ret_node = None
            cur_node = None

            for d in xrange(dim):
                dim_var = self.gen_dim_var()

                initial = cpp_ast.CNumber(array.ghost_depth)
                end = cpp_ast.CNumber(array.shape[d]-array.ghost_depth-1)
                increment = cpp_ast.CNumber(1)
                if d == 0:
                    ret_node = cpp_ast.For(dim_var, initial, end, increment, cpp_ast.Block())
                    cur_node = ret_node
                elif d == dim-2:
                    cur_node.body = StencilKernel.StencilConvertASTCilk.CilkFor(dim_var, initial, end, increment, cpp_ast.Block())
                    cur_node = cur_node.body
                else:
                    cur_node.body = cpp_ast.For(dim_var, initial, end, increment, cpp_ast.Block())
                    cur_node = cur_node.body

            return (cur_node, ret_node)








