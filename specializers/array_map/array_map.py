# really dumb example of using tree transformations w/asp

import asp.codegen.ast_tools as ast_tools
import asp.codegen.python_ast as ast
import asp.codegen.cpp_ast as cpp
#import asp.codegen.ast_explorer as ast_explorer

class Converter(ast_tools.ConvertAST):
    pass

class ArrayMap(object):

    def __init__(self):
        self.pure_python = True

    def map_using_trees(self, arr):
        operation_ast = ast_tools.parse_method(self.operation)
        expr_ast = operation_ast.body[0].body[0].value
        converter = Converter()
        expr_cpp = converter.visit(expr_ast)

        import asp.codegen.templating.template as template
        mytemplate = template.Template(filename="templates/map_template.mako", disable_unicode=True)
        rendered = mytemplate.render(num_items=len(arr), expr=expr_cpp)

        import asp.jit.asp_module as asp_module
        mod = asp_module.ASPModule()
        mod.add_function("map_in_c", rendered)
        return mod.map_in_c(arr)

    def map(self, arr):
        for i in range(0, len(arr)):
            arr[i] = self.operation(arr[i])
