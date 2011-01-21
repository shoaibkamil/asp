import codepy, codepy.jit, codepy.toolchain, codepy.bpl
from asp.util import *
import asp.codegen.cpp_ast as cpp_ast
import pickle


class Variants(object):
    def __init__(self, func, variant_names, key_func):
        self.variant_names = variant_names
        self.func_name = func
        self.make_key = key_func     
        self.variant_times = {} #key: (name, *args)  value:[time of each variant]
        self.best_found = {} #Dict of names, key: (name,*args) value: var_name/False
        self.next_variant_run = {} #Dict of indexes, key: (name,*args) value: index into variant_names

    def set_best(self, name, *args, **kwargs):
        key = self.make_key(name, *args, **kwargs)
        times =  self.variant_times[key]
        idx = times.index(min(times)) 
        self.best_found[key] = self.variant_names[idx]

    def get_best(self, name, *args, **kwargs):
        key = self.make_key(name, *args, **kwargs)
        return self.best_found.get(key, False)

    def add_time(self, elapsed, name, *args, **kwargs):
        key = self.make_key(name, *args, **kwargs)
        curr_var = self.next_variant_run.setdefault(key, 0)
        self.variant_times.setdefault(key,[]).append(elapsed)
        self.next_variant_run[key] = curr_var+1
        if curr_var+1 >= len(self.variant_names):
            self.set_best(name, *args, **kwargs)

    def which_to_run(self, name, *args, **kwargs):
        key = self.make_key(name, *args, **kwargs)
        return self.next_variant_run.setdefault(key, 0)

class ASPModule(object):
    
    def __init__(self, use_cuda=False):
        self.toolchain = codepy.toolchain.guess_toolchain()
        self.module = codepy.bpl.BoostPythonModule()
        self.cache_dir = "./.aspcache"
        self.dirty = False
        self.compiled_methods = []
        self.compiled_methods_with_variants = {}
        self.times = {}
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

    def add_time(self, func_name, time):
        # if no time exists, add this one, otherwise append
        self.times.setdefault(func_name, []).append(time)
    
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
            fname = self.get_name_from_func(func)
        
        self.dirty = True
        self.compiled_methods.append(fname)

    def add_function_with_variants(self, variant_funcs, func_name, variant_names, key_maker=lambda name, *args, **kwargs: (name), cuda_func=False):
        variants = Variants(func_name, variant_names, key_maker)
        for x in range(0,len(variant_funcs)):
            self.add_function_helper(variant_funcs[x], fname=variant_names[x], cuda_func=cuda_func)
        self.compiled_methods_with_variants[func_name] = variants
        self.dirty = True

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
                
    def compile(self):
        if self.use_cuda:
            self.compiled_module = self.cuda_module.compile(self.toolchain, self.nvcc_toolchain, debug=True, cache_dir="cache")
        else:
            self.compiled_module = self.module.compile(self.toolchain, debug=True, cache_dir="cache")
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
        import time
        def special(*args, **kwargs):
            variants = self.compiled_methods_with_variants[name]
            start_time = time.time()
            best = variants.get_best(name, *args, **kwargs)
            if best:
                real_func = self.compiled_module.__getattribute__(best)
            else: 
                real_func = self.compiled_module.__getattribute__(variants.variant_names[variants.which_to_run(name,*args,**kwargs)])
            result = real_func(*args, **kwargs)
            elapsed = time.time() - start_time
            if not best:
                variants.add_time(elapsed, name, *args, **kwargs)
            self.add_time(name, elapsed)
            return result
        return special

    def save_func_variant_timings(self, name):
        variants = self.compiled_methods_with_variants[name]
        f = open(self.cache_dir+'/'+name+'.vardump', 'w')
        pickle.dump(variants.variant_times, f)
        pickle.dump(variants.best_found, f)
        pickle.dump(variants.next_variant_run, f)
        f.close()

    def restore_func_variant_timings(self, name):
        variants = self.compiled_methods_with_variants[name]
        f = open(self.cache_dir+'/'+name+'.vardump', 'r')
        variants.variant_times = pickle.load(f)
        variants.best_found = pickle.load(f)
        variants.next_variant_run = pickle.load(f)
        f.close()

    def clear_func_variant_timings(self, name):
        variants = self.compiled_methods_with_variants[name]
        variants.variant_times = {}
        variants.best_found = {}
        variants.next_variant_run = {}
        
    def __getattr__(self, name):
        if name in self.compiled_methods_with_variants.keys():
            if self.dirty:
                self.compile()
            return self.func_with_variants(name)

        else:
            raise AttributeError("No method %s found; did you add it?" % name)

