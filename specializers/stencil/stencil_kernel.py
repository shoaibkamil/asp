"""The main driver, intercepts the kernel() call and invokes the other components.

Stencil kernel classes are subclassed from the StencilKernel class
defined here. At initialization time, the text of the kernel() method
is parsed into a Python AST, then converted into a StencilModel by
stencil_python_front_end. The kernel() function is replaced by
shadow_kernel(), which intercepts future calls to kernel().

During each call to kernel(), stencil_unroll_neighbor_iter is called
to unroll neighbor loops, stencil_convert is invoked to convert the
model to C++, and an external compiler tool is invoked to generate a
binary which then efficiently completes executing the call. The binary
is cached for future calls.
"""

import numpy
import inspect
from stencil_grid import *
from stencil_python_front_end import *
from stencil_unroll_neighbor_iter import *
from stencil_convert import *
import asp.codegen.python_ast as ast
import asp.codegen.cpp_ast as cpp_ast
import asp.codegen.ast_tools as ast_tools
from asp.util import *
import copy

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
        self.model = StencilPythonFrontEnd().parse(self.kernel_ast)

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
            debug_print("match!")
            self.mod.kernel(*[y.data for y in args])
            return

        # otherwise, do the first-run flow

        # check if we can specialize for this data
        #FIXME: impelement.

        # ask asp infrastructure for machine and platform info, including if cilk+ is available
        #FIXME: impelement.  set self.with_cilk=true if cilk is available
        
        input_grids = args[0:-1]
        output_grid = args[-1]
        model = copy.deepcopy(self.model)
        model = StencilUnrollNeighborIter(model, input_grids, output_grid).run()

        # depending on whether cilk is available, we choose which converter to use
        if not self.with_cilk:
            Converter = StencilConvertAST
        else:
            Converter = StencilConvertASTCilk

        # generate variant with no unrolling, then generate variants for various unrollings
        variants = [Converter(model, input_grids, output_grid).run()]
        variant_names = ["kernel_unroll_1"]
        for x in [2,4,8,16,32,64]:
            check_valid = max(map(
                lambda y: (y.shape[-1]-2*y.ghost_depth) % x,
                args))
            if check_valid == 0:
                variants.append(Converter(model, input_grids, output_grid, unroll_factor=x).run())
                variant_names.append("kernel_unroll_%s" % x)

        from asp.jit import asp_module

        mod = self.mod = asp_module.ASPModule()
        self.add_libraries(mod)
        if self.with_cilk:
            mod.backends["c++"].toolchain.cc = "icc"
            mod.backends["c++"].toolchain.cflags += ["-intel-extensions", "-fast"]
            mod.backends["c++"].toolchain.cflags += ["-I/usr/include/x86_64-linux-gnu"]
            mod.backends["c++"].toolchain.cflags.remove('-fwrapv')
        else:
            mod.backends["c++"].toolchain.cflags += ["-fopenmp", "-O3", "-msse3"]
#        print mod.toolchain.cflags
        if mod.backends["c++"].toolchain.cflags.count('-Os') > 0:
            mod.backends["c++"].toolchain.cflags.remove('-Os')
        if mod.backends["c++"].toolchain.cflags.count('-O2') > 0:
            mod.backends["c++"].toolchain.cflags.remove('-O2')
        debug_print("toolchain" + str(mod.backends["c++"].toolchain.cflags))
        mod.add_function("kernel", variants, variant_names)

        # package arguments and do the call 
        myargs = [y.data for y in args]
        mod.kernel(*myargs)

        # save parameter sizes for next run
        self.specialized_sizes = [x.shape for x in args]
