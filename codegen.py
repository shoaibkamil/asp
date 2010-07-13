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

    # numbers/names/etc
    def visit_Num(self, node):
        return str(node.n)
    def visit_Name(self, node):
        return str(node.id)
    def visit_Pass(self, node):
        return ";"

    # function call
    def visit_Call(self, node):
        func = self.visit(node.func)
        # map visit() onto the parameters and join them with commas
        args = ','.join(map(self.visit, node.args))        
        return func + "(" + args + ")"

    # attribute
    # does foo.bar
    def visit_Attribute(self, node):
        return self.visit(node.value) + "." + node.attr

    # subscript
    def visit_Subscript(self, node):
        return self.visit(node.value) + self.visit(node.slice)
    # kinds of subscripts
    def visit_Index(self, node):
        return "[%s]" % self.visit(node.value)


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

    # assignement
    # only support a single assignment by default
    def visit_Assign(self, node):
        return self.visit(node.targets[0]) + " = " + self.visit(node.value)


    # for iterator
    def visit_For(self, node):
        str =  "for " + self.visit(node.target)
        str += " in " +  self.visit(node.iter) + " {\n"
        str +=  ";".join(map(self.visit, node.body)) + "\n}"
        return str
                                   
