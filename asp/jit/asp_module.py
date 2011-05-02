import codepy, codepy.jit, codepy.toolchain, codepy.bpl
from asp.util import *
import asp.codegen.cpp_ast as cpp_ast
import pickle
from variant_history import *

class ASPModule(object):
    
    def __init__(self, use_cuda=False):
        self.compiled_methods = {}
        self.helper_method_names = []
        self.module = codepy.bpl.BoostPythonModule()
        self.toolchain = codepy.toolchain.guess_toolchain()
        self.cache_dir = "cache"
        self.dirty = False
        self.timing_enabled = True
        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda_module = codepy.cuda.CudaModule(self.module)
            self.cuda_module.add_to_preamble([cpp_ast.Include('cuda.h', False)])
            self.nvcc_toolchain = codepy.toolchain.guess_nvcc_toolchain()


    def add_library(self, feature, include_dirs, library_dirs=[], libraries=[]):
        self.toolchain.add_library(feature, include_dirs, library_dirs, libraries)

    def add_cuda_library(self, feature, include_dirs, library_dirs=[], libraries=[]):
        self.nvcc_toolchain.add_library(feature, include_dirs, library_dirs, libraries)        

    def add_cuda_arch_spec(self, arch):
        archflag = '-arch='
        if 'sm_' not in arch: archflag += 'sm_' 
        archflag += arch
        self.nvcc_toolchain.cflags += [archflag]

    def add_header(self, include_file):
        self.module.add_to_preamble([cpp_ast.Include(include_file, False)])

    def add_cuda_header(self, include_file):
        self.cuda_module.add_to_preamble([cpp_ast.Include(include_file, False)])

    def add_to_preamble(self, pa):
        if isinstance(pa, str):
            pa = [cpp_ast.Line(pa)]
        self.module.add_to_preamble(pa)

    def add_to_cuda_preamble(self, pa):
        if isinstance(pa, str):
            pa = [cpp_ast.Line(pa)]
        self.cuda_module.add_to_preamble(pa)

    def add_to_init(self, stmt):
        if isinstance(stmt, str):
            stmt = [cpp_ast.Line(stmt)]
        self.module.add_to_init(stmt)

    def add_to_cuda_module(self, block):
        if isinstance(block, str):
            block = [cpp_ast.Line(block)]
        self.cuda_module.add_to_module(block)

    def get_name_from_func(self, func):
        """
        returns the name of a function from a CodePy FunctionBody object
        """
        return func.fdecl.subdecl.name


    def add_function_helper(self, func, fname=None, cuda_func=False):
        if cuda_func:
            module = self.cuda_module
        else:
            module = self.module
        
        if isinstance(func, str):
            if fname == None:
                raise Exception("Cannot add a function as a string without specifying the function's name")
            module.add_to_module([cpp_ast.Line(func)])
            module.add_to_init([cpp_ast.Statement(
                        "boost::python::def(\"%s\", &%s)" % (fname, fname))])
        else:
            module.add_function(func)
        self.dirty = True

    def add_function_with_variants(self, variant_funcs, func_name, variant_names, key_maker=lambda name, *args, **kwargs: (name), limit_funcs=None, compilable=None, param_names=None, cuda_func=False):
        limit_funcs = limit_funcs or [lambda name, *args, **kwargs: True]*len(variant_names) 
        compilable = compilable or [True]*len(variant_names)
        param_names = param_names or ['Unknown']*len(variant_names)
        method_info = self.compiled_methods.get(func_name, None)
        if not method_info:
            method_info = CodeVariants(variant_names, key_maker, param_names)
            method_info.limiter.append(variant_names, limit_funcs, compilable)
        else:
            method_info.append(variant_names)
            method_info.database.clear_oracle()
            method_info.limiter.append(variant_names, limit_funcs, compilable)
        for x in range(0,len(variant_funcs)):
            self.add_function_helper(variant_funcs[x], fname=variant_names[x], cuda_func=cuda_func)
        self.compiled_methods[func_name] = method_info

    def add_function(self, funcs, fname=None, variant_names=None, cuda_func=False):
        """
        self.add_function(func) takes func as either a generable AST or a string, or
        list of variants in either format.
        """
        if variant_names:
            self.add_function_with_variants(funcs, fname, variant_names, cuda_func=cuda_func)
        else:
            variant_funcs = [funcs]
            if not fname:
                fname = self.get_name_from_func(funcs)
            variant_names = [fname]
            self.add_function_with_variants(variant_funcs, fname, variant_names, cuda_func=cuda_func)

    def add_helper_function(self, fname, cuda_func=False):
        self.add_function_helper("", fname=fname, cuda_func=cuda_func)
        self.helper_method_names.append(fname)
                
    def compile(self):
        if self.use_cuda:
            self.compiled_module = self.cuda_module.compile(self.toolchain, self.nvcc_toolchain, debug=True, cache_dir=self.cache_dir)
        else:
            self.compiled_module = self.module.compile(self.toolchain, debug=True, cache_dir=self.cache_dir)
        self.dirty = False
        
    def specialized_func(self, name):
        import time
        def error_func(*args, **kwargs):
            raise Exception("No variant of method found to run on input size %s on the specified device" % str(args))
        def special(*args, **kwargs):
            method_info = self.compiled_methods[name]
            key = method_info.make_key(name,*args,**kwargs)
            v_id = method_info.selector.get_v_id_to_run(method_info.v_id_set, key,*args,**kwargs)
            real_func = self.compiled_module.__getattribute__(v_id) if v_id else error_func
            start_time = time.time() 
            result = real_func(*args, **kwargs)
            elapsed = time.time() - start_time
            method_info.database.add_time( key, elapsed, v_id, method_info.v_id_set)
            return result
        return special

    def helper_func(self, name):
        def helper(*args, **kwargs):
            real_func = self.compiled_module.__getattribute__(name)
            return real_func(*args, **kwargs)
        return helper

    def save_method_timings(self, name, file_name=None):
        method_info = self.compiled_methods[name]
        f = open(file_name or self.cache_dir+'/'+name+'.vardump', 'w')
        d = method_info.get_picklable_obj()
        d.update(method_info.database.get_picklable_obj())
        pickle.dump( d, f)
        f.close()

    def restore_method_timings(self, name, file_name=None):
        method_info = self.compiled_methods[name]
        try: 
	    f = open(file_name or self.cache_dir+'/'+name+'.vardump', 'r')
            obj = pickle.load(f)
            if obj: method_info.set_from_pickled_obj(obj)
            if obj: method_info.database.set_from_pickled_obj(obj, method_info.v_id_set)
            f.close()
        except IOError: pass

    def clear_method_timings(self, name):
        method_info = self.compiled_methods[name]
        method_info.database.clear()

    def __getattr__(self, name):
        if name in self.compiled_methods:
            if self.dirty:
                self.compile()
            return self.specialized_func(name)
        elif name in self.helper_method_names:
            return self.helper_func(name)
        else:
            raise AttributeError("No method %s found; did you add it?" % name)

