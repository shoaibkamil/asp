"""Takes an unrolled StencilModel and converts it to a C++ AST.

The third stage in processing. Input must be processed with
StencilUnrollNeighborIter first to remove neighbor loops and
InputElementZeroOffset nodes. Done once per call.
"""

import ast
import asp.codegen.cpp_ast as cpp_ast
import asp.codegen.ast_tools as ast_tools
import stencil_model
from assert_utils import *

class StencilConvertAST(ast_tools.ConvertAST):
    def __init__(self, model, input_grids, output_grid, unroll_factor=None):
        assert_has_type(model, stencil_model.StencilModel)
        assert len(input_grids) == len(model.input_grids), 'Incorrect number of input grids'
        self.model = model
        self.input_grids = input_grids
        self.output_grid = output_grid
        self.output_grid_name = 'out_grid'
        self.dim_vars = []
        self.unroll_factor = unroll_factor
        self.var_names = [self.output_grid_name]
        self.next_fresh_var = 0
        super(StencilConvertAST, self).__init__()

    def run(self):
        self.model = self.visit(self.model)
        return self.model

    # Visitors
    
    def visit_StencilModel(self, node):
        self.argdict = dict()
        for i in range(len(node.input_grids)):
            self.var_names.append(node.input_grids[i].name)
            self.argdict[node.input_grids[i].name] = self.input_grids[i]
        self.argdict[self.output_grid_name] = self.output_grid

        assert node.border_kernel.body == [], 'Border kernels not yet implemented'

        if not self.unroll_factor:
            func_name = "kernel_unroll_1"
        else:
            func_name = "kernel_unroll_%s" % self.unroll_factor
        arg_names = [x.name for x in node.input_grids] + [self.output_grid_name]
        args = [cpp_ast.Pointer(cpp_ast.Value("PyObject", x)) for x in arg_names]
        body = self.visit_interior_kernel(node.interior_kernel)
        return cpp_ast.FunctionBody(cpp_ast.FunctionDeclaration(cpp_ast.Value("void", func_name), args),
                                    cpp_ast.Block([body]))

    def visit_interior_kernel(self, node):
        cur_node, ret_node = self.gen_loops(node)

        body = cpp_ast.Block()
        body.extend([self.gen_array_macro_definition(x) for x in self.argdict])
        body.extend(self.gen_array_unpack())

        self.output_index_var = cpp_ast.CName(self.gen_fresh_var())
        body.append(cpp_ast.Value("int", self.output_index_var))
        body.append(cpp_ast.Assign(self.output_index_var,
                                   self.gen_array_macro(
                                       self.output_grid_name, [cpp_ast.CName(x) for x in self.dim_vars])))

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

    def visit_OutputAssignment(self, node):
        return cpp_ast.Assign(self.visit(stencil_model.OutputElement()), self.visit(node.value))

    def visit_ScalarBinOp(self, node):
        return super(StencilConvertAST, self).visit_BinOp(ast.BinOp(node.left, node.op, node.right))

    def visit_OutputElement(self, node):
        return cpp_ast.Subscript("_my_" + self.output_grid_name, self.output_index_var)

    def visit_InputElement(self, node):
        index = self.gen_array_macro(node.grid.name,
                                     map(lambda x,y: cpp_ast.BinOp(cpp_ast.CName(x), "+", cpp_ast.CNumber(y)),
                                         self.dim_vars,
                                         node.offset_list))
        return cpp_ast.Subscript("_my_" + node.grid.name, index)

    # Helper functions
    
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
        return cpp_ast.Call(cpp_ast.CName(name), point)

    def gen_array_unpack(self):
        ret =  [cpp_ast.Assign(cpp_ast.Pointer(cpp_ast.Value("npy_double", "_my_"+x)), 
                cpp_ast.TypeCast(cpp_ast.Pointer(cpp_ast.Value("npy_double", "")), cpp_ast.FunctionCall(cpp_ast.CName("PyArray_DATA"), params=[cpp_ast.CName(x)])))
                for x in self.argdict.keys()]

        return ret

    def gen_loops(self, node):
        dim = len(self.output_grid.shape)

        ret_node = None
        cur_node = None

        for d in xrange(dim):
            dim_var = self.gen_fresh_var()
            self.dim_vars.append(dim_var)

            initial = cpp_ast.CNumber(self.output_grid.ghost_depth)
            end = cpp_ast.CNumber(self.output_grid.shape[d]-self.output_grid.ghost_depth-1)
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

    def gen_fresh_var(self):
        while True:
            self.next_fresh_var += 1
            var = "x%d" % self.next_fresh_var
            if var not in self.var_names:
                return var

class StencilConvertASTCilk(StencilConvertAST):
    class CilkFor(cpp_ast.For):
        def intro_line(self):
            return "cilk_for (%s; %s; %s += %s)" % (self.start, self.condition, self.loopvar, self.increment)

    def gen_loops(self, node):
        dim = len(self.output_grid.shape)

        ret_node = None
        cur_node = None

        for d in xrange(dim):
            dim_var = self.gen_fresh_var()
            self.dim_vars.append(dim_var)

            initial = cpp_ast.CNumber(self.output_grid.ghost_depth)
            end = cpp_ast.CNumber(self.output_grid.shape[d]-self.output_grid.ghost_depth-1)
            increment = cpp_ast.CNumber(1)
            if d == 0:
                ret_node = cpp_ast.For(dim_var, initial, end, increment, cpp_ast.Block())
                cur_node = ret_node
            elif d == dim-2:
                cur_node.body = StencilConvertASTCilk.CilkFor(dim_var, initial, end, increment, cpp_ast.Block())
                cur_node = cur_node.body
            else:
                cur_node.body = cpp_ast.For(dim_var, initial, end, increment, cpp_ast.Block())
                cur_node = cur_node.body

        return (cur_node, ret_node)
