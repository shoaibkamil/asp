import ast_tools
import ast



abc = ast.parse("1+2")
goober = ast_tools.CodeGenerator()
goober2 = ast_tools.ASTPrettyPrinter()

print goober2.visit(abc)
print ast.dump(abc)

print goober.visit(abc)
