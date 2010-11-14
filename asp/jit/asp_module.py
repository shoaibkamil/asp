import codepy, codepy.jit, codepy.toolchain, codepy.bpl
from asp.util import *
import asp.codegen.cpp_ast as cpp_ast


class ASPModule(object):

    
    def __init__(self):
        self.toolchain = codepy.toolchain.guess_toolchain()
        self.module = codepy.bpl.BoostPythonModule()
        self.dirty = False
        self.compiled_methods = []
        self.times = {}
        self.timing_enabled = True

    def add_library(self, feature, include_dirs, library_dirs=[], libraries=[]):
        self.toolchain.add_library(feature, include_dirs, library_dirs, libraries)

    def add_header(self, include_file):
        import asp.codegen.cpp_ast as cpp_ast
        self.module.add_to_preamble([cpp_ast.Include(include_file, False)])

    def add_to_preamble(self, pa):
        if isinstance(pa, str):
            pa = cpp_ast.Line(pa)
        self.module.add_to_preamble(pa)

    def add_to_init(self, stmt):
        if isinstance(stmt, str):
            stmt = cpp_ast.Line(stmt)
        self.module.add_to_init(stmt)

    def add_time(self, func_name, time):
        # if no time exists, add this one, otherwise append
        self.times.setdefault(func_name, []).append(time)
    
    def get_name_from_func(self, func):
        """
        returns the name of a function from a CodePy FunctionBody object
        """
        return func.fdecl.subdecl.name

    def add_function(self, func, fname=None):
        """
        self.add_function(func) takes func as either a generable AST or a string.
        """
        if isinstance(func, str):
            if fname == None:
                raise Exception("Cannot add a function as a string without specifying the function's name")
            self.module.add_to_module([cpp_ast.Line(func)])
            self.module.add_to_init([cpp_ast.Statement(
                        "boost::python::def(\"%s\", &%s)" % (fname, fname))])
        else:
            self.module.add_function(func)
            fname = self.get_name_from_func(func)
        
        self.dirty = True
        self.compiled_methods.append(fname)
    

    def compile(self):
        
        self.compiled_module = self.module.compile(self.toolchain, debug=True, cache_dir=".")
        self.dirty = False
        
    def func_with_timing(self, name):
        import time
        def special(*args, **kwargs):
            start_time = time.time()
            real_func = self.compiled_module.__getattribute__(name)
            result = real_func(*args, **kwargs)
            self.add_time(name, (time.time()-start_time))
            return result

        return special

    def __getattr__(self, name):
        if name in self.compiled_methods:
            if self.dirty:
                self.compile()
            if self.timing_enabled == True:
                return self.func_with_timing(name)
            else:
                return self.compiled_module.__getattribute__(name)

        else:
            raise AttributeError("No method %s found; did you add it?" % name)

