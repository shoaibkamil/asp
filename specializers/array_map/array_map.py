# really dumb example of using tree transformations w/asp

class ArrayMap(object):

    def __init__(self):
        self.pure_python = True

    def map_using_trees(self, arr):
        import asp.codegen.templating.template as template
        import inspect
        import asp.codegen.python_ast as ast
        import asp.codegen.ast_tools as ast_tools

        src = inspect.getsource(self.operation)
        operation_ast = ast.parse(src.lstrip())
        return_ast = operation_ast.body[0]
        expr_ast = return_ast.body[0].value
        expr_cpp = ast_tools.ConvertAST().visit(expr_ast)

        mytemplate = template.Template(filename="templates/map_template.mako", disable_unicode=True)
        rendered = mytemplate.render(num_items=len(arr), expr=expr_cpp)

        import asp.jit.asp_module as asp_module
        mod = asp_module.ASPModule()
        mod.add_function("map_in_c", rendered)
        return mod.map_in_c(arr)

    def map(self, arr):
        for i in range(0, len(arr)):
            arr[i] = self.operation(arr[i])
