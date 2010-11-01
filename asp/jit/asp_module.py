import codepy, codepy.jit, codepy.toolchain, codepy.bpl
from asp.util import *

class ASPModule(object):
    
    def __init__(self):
        self.toolchain = codepy.toolchain.guess_toolchain()
        self.module = codepy.bpl.BoostPythonModule()
        self.program_text = None

    def add_library(self, feature, include_dirs, library_dirs=[], libraries=[]):
        self.toolchain.add_library(feature, include_dirs, library_dirs, libraries)

    def add_header(self, include_file):
        import asp.codegen.cpp_ast as cpp_ast
        self.module.add_to_preamble([cpp_ast.Include(include_file, False)])

    def add_to_init(self, stmt):
        self.module.add_to_init(stmt)

    def add_function(self, func):
        """
        self.add_function(func) takes func as either a generable AST or a string.
        """
        if func.__class__ == str:
            self.program_text = func
        else:
            self.module.add_function(func)
            self.program_text = self.module.generate()

        debug_print(self.program_text)

    def compile(self):
        if self.program_text == None:
            raise Exception("No code to generate.")
        
        self.compiled_module = self.module.compile(self.toolchain, debug=True, cache_dir=".")
        
