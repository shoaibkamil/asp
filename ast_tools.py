import ast 

class ASTPrettyPrinter(ast.NodeVisitor):

    def __init__(self):
        self.indent_level = 0
        ast.NodeVisitor.__init__(self)

    def generic_visit(self,node):

        for x in range(self.indent_level):
            print "\t",
        print node.__class__.__name__,
        
        for (fieldname,value) in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                pass
            else:
                print "  ", [fieldname, value], 

        print("\n")

        self.indent_level += 1
        for child in ast.iter_child_nodes(node):
            self.generic_visit(child)
        self.indent_level -= 1


### Code generation
class CodeGenerator(ast.NodeVisitor):
    def generic_visit(self, node):
        str = ""
        
        for child in ast.iter_child_nodes(node):
            str += self.visit(child)

        return str


    def visit_Num(self, node):
        return str(node.n)

    def visit_BinOp(self, node):
        return self.visit(node.left) + self.visit(node.op) + self.visit(node.right)

    def visit_Add(self, node):
        return "+"
    

    
