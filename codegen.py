import ast


### Code generation
class CodeGenerator(ast.NodeVisitor):
    def generic_visit(self, node):
        str = ""

        import sys
        sys.stderr.write("Unsupported node: " + node.__class__.__name__ + "\n")
        
        for child in ast.iter_child_nodes(node):
            str += self.visit(child)

        return str

    # numbers
    def visit_Num(self, node):
        return str(node.n)

    def visit_Name(self, node):
        return str(node.id)

    # function call
    def visit_Call(self, node):
        func = self.visit(node.func)
        # map visit() onto the parameters and join them with commas
        args = ','.join(map(self.visit, node.args))        
        return func + "(" + args + ")"

    # Binary operations

    def visit_BinOp(self, node):
        return self.visit(node.left) + self.visit(node.op) + self.visit(node.right)

    def visit_Add(self, node):
        return "+"
    def visit_Mult(self, node):
        return "*"
    def visit_Sub(self, node):
        return "-"
    def visit_Div(self, node):
        return "/"


    # for iterator
    def visit_For(self, node):
        print "target: ", self.visit(node.target)
        print "iter: ", self.visit(node.iter)
        print "body: ", node.body
        return ""
                                   
