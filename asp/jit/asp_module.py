import codepy, codepy.jit, codepy.toolchain, codepy.bpl
from asp.util import *
import asp.codegen.cpp_ast as cpp_ast


class ASPModule(object):

    class Variants(object):
        def __init__(self, func, variant_names):
            self.variant_times = {}
            self.variant_names = variant_names
            self.func_name = func
            self.best_found = False
            self.next_variant_run = 0
        
        def set_best(self):
            self.best_found = self.variant_names[0]
            for (f,t) in self.variant_times.iteritems():
                if t < self.variant_times[self.best_found]:
                    self.best_found = f

    
    def __init__(self):
        self.toolchain = codepy.toolchain.guess_toolchain()
        self.module = codepy.bpl.BoostPythonModule()
        self.dirty = False
        self.compiled_methods = []
        self.compiled_methods_with_variants = {}
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


    def add_function_helper(self, func, fname=None):
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

    def add_function_with_variants(self, variant_funcs, func_name, variant_names):
        variants = ASPModule.Variants(func_name, variant_names)
        for x in range(0,len(variant_funcs)):
            self.add_function_helper(variant_funcs[x], fname=variant_names[x])
        self.compiled_methods_with_variants[func_name] = variants
        self.dirty = True

    def add_function(self, funcs, fname=None, variant_names=None):
        """
        self.add_function(func) takes func as either a generable AST or a string, or
        list of variants in either format.
        """
        if variant_names:
            self.add_function_with_variants(funcs, fname, variant_names)
        else:
            variant_funcs = [funcs]
            if not fname:
                fname = self.get_name_from_func(funcs)
            variant_names = [fname]
            self.add_function_with_variants(variant_funcs, fname, variant_names)
                

    

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

    def func_with_variants(self, name):
        variants = self.compiled_methods_with_variants[name]
        if variants.best_found:
            return self.func_with_timing(variants.best_found)
        else:
            import time
            def special(*args, **kwargs):
                start_time = time.time()
                real_func = self.compiled_module.__getattribute__(variants.variant_names[variants.next_variant_run])
                result = real_func(*args, **kwargs)
                elapsed = time.time() - start_time
                self.add_time(name, elapsed)
                variants.variant_times.setdefault(variants.variant_names[variants.next_variant_run],[]).append(elapsed)
                variants.next_variant_run += 1
                if variants.next_variant_run >= len(variants.variant_names):
                    variants.set_best()
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

        elif name in self.compiled_methods_with_variants.keys():
            if self.dirty:
                self.compile()
            return self.func_with_variants(name)

        else:
            raise AttributeError("No method %s found; did you add it?" % name)

