import ast 


#### Simple AST

# # convert from python ast to this ast
# def convert_ast(pyast):
#     classname = pyast.__class__.__name__

#     print "Classname is ", classname

#     if classname == 'Num':
#         return IntExp(pyast.n)
#     elif classname == 'Name':
#         return NameExp(pyast.id)
#     elif classname == 'BinOp':
#         left = convert_ast(pyast.left)
#         right = convert_ast(pyast.right)
#         return BinExp(left, pyast.op, right)
#     # these are hacky right now, need to be checked
#     elif classname == 'Module':
#         return convert_ast(pyast.body[0])
#     elif classname == 'Expr':
#         return convert_ast(pyast.value)



# # parent class all nodes inherit from
# class SimpleAST(ast.AST):
#     pass

# class IntExp(SimpleAST):
#     def __init__(self, value):
#         super(IntExp, self).__init__()
#         self.value = value

# class NameExp(SimpleAST):
#     def __init__(self, name):
#         super(NameExp, self).__init__()
#         self.name = name

# class BinExp(SimpleAST):
#     def __init__(self, left, op, right):
#         super(BinExp, self).__init__()
#         self.left = left
#         self.op = op
#         self.right = right



        

#### Pretty Print ASTs
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

    

    
