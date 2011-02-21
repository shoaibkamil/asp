import codepy, codepy.jit, codepy.toolchain, codepy.bpl
from asp.util import *
import asp.codegen.cpp_ast as cpp_ast
import pickle


class CodeVariants(object):
    def __init__(self, func, variant_names, key_func, limit_funcs, compilable, param_names):
        self.variant_names = variant_names
        self.func_name = func
        self.make_key = key_func     
        self.check_limits_arr = limit_funcs    
        self.param_names = param_names
        self.compilable = compilable
        self.variant_times = {}    # Dict of times,   key: (name,*args), value: [time of each variant] for an input
        self.best_found = {}       # Dict of names,   key: (name,*args), value: var_name/False for an input
        self.next_variant_run = {} # Dict of indexes, key: (name,*args), value: index into variant_names for an input

    def set_best(self, name, *args, **kwargs):
        key = self.make_key(name, *args, **kwargs)
        times =  self.variant_times[key]
        succeeded = filter(lambda x: x > 0, times)
        if not succeeded: 
            self.best_found[key] = False
        else: 
            idx = times.index(min(succeeded)) 
            self.best_found[key] = self.variant_names[idx]

    def get_best(self, name, *args, **kwargs):
        key = self.make_key(name, *args, **kwargs)
        return self.best_found.get(key, False)

    def add_time(self, elapsed, name, *args, **kwargs):
        key = self.make_key(name, *args, **kwargs)
        curr_var = self.next_variant_run.setdefault(key, 0)
        self.variant_times.setdefault(key,[]).append(elapsed)
        self.next_variant_run[key] = curr_var+1
        if curr_var+1 == len(self.variant_names):
            self.set_best(name, *args, **kwargs)

    def get_func_to_run(self, compiled_module, name, *args, **kwargs):
        def error_func(*args, **kwargs):
            print "Warning: No functon variant could be found to run on this input size on the specified device"
            return None        
        ret_func = error_func
        ret_tried_all = False
        key = self.make_key(name, *args, **kwargs)
        if len(self.variant_names) == 1: 
            self.best_found[key] = self.variant_names[0]
            ret_func = compiled_module.__getattribute__(self.variant_names[0])
            ret_tried_all = True
        elif self.best_found.get(key, False):
            ret_func = compiled_module.__getattribute__(self.best_found[key])
            ret_tried_all = True
        else:
            name = self.which_variant_name_to_run(key,name,*args,**kwargs)
            if name:
                ret_func = compiled_module.__getattribute__(name)
                ret_tried_all = self.next_variant_run.get(key, 1) >= len(self.variant_names)
        return ret_func, ret_tried_all
    
    def which_variant_name_to_run(self, key, name, *args, **kwargs):
        which_var_idx = self.next_variant_run.setdefault(key, 0)
        while which_var_idx < len(self.variant_names) and \
             (not self.compilable[which_var_idx] or      \
              not self.check_limits_arr[which_var_idx](name, *args, **kwargs)):
            self.add_time(-1., name, *args, **kwargs) 
            which_var_idx += 1
        if(which_var_idx >= len(self.variant_names)):
            return self.best_found.get(key,False)
        else: return self.variant_names[which_var_idx]

    def append(self, variant_names, limit_funcs, compilable):
        self.variant_names.extend(variant_names)
        self.check_limits_arr.extend(limit_funcs)
        self.compilable.extend(compilable)
        self.best_found.clear() #new variant might be the best

    def get_picklable_obj(self):
        return {'func_name': self.func_name,
                'variant_names': self.variant_names,
                'param_names': self.param_names,
                'variant_times': self.variant_times,
                'best_found': self.best_found,
                'compilable': self.compilable,
                'next_variant_run': self.next_variant_run
               }

    def set_from_pickled_obj(self, obj):
        self.func_name = obj['func_name']
        self.variant_names = obj['variant_names']
        self.param_names = obj['param_names']
        self.variant_times = obj['variant_times']
        self.best_found = obj['best_found']
        self.compilable = obj['compilable']
        self.next_variant_run = obj['next_variant_run']

class ASPModule(object):
    
    def __init__(self, use_cuda=False):
        self.toolchain = codepy.toolchain.guess_toolchain()
        self.module = codepy.bpl.BoostPythonModule()
        self.cache_dir = "cache"
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

    def add_function_with_variants(self, variant_funcs, func_name, variant_names, key_maker=lambda name, *args, **kwargs: (name), limit_funcs=None, compilable=None, param_names=None, cuda_func=False):
        limit_funcs = limit_funcs or [lambda name, *args, **kwargs: True]*len(variant_names) 
        compilable = compilable or [True]*len(variant_names)
        param_names = param_names or ['Unknown']*len(variant_names)
        variants = self.compiled_methods_with_variants.get(func_name, None)
        if not variants:
            variants = CodeVariants(func_name, variant_names, key_maker, limit_funcs, compilable, param_names)
        else:
            variants.append(variant_names, limit_funcs, compilable)
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
            self.compiled_module = self.cuda_module.compile(self.toolchain, self.nvcc_toolchain, debug=True, cache_dir=self.cache_dir)
        else:
            self.compiled_module = self.module.compile(self.toolchain, debug=True, cache_dir=self.cache_dir)
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
            real_func, have_tried_all = variants.get_func_to_run(self.compiled_module, name,*args,**kwargs)
            result = real_func(*args, **kwargs)
            elapsed = time.time() - start_time
            if not have_tried_all:
                variants.add_time(elapsed, name, *args, **kwargs)
            self.add_time(name, elapsed)
            return result
        return special

    def save_func_variant_timings(self, name):
        variants = self.compiled_methods_with_variants[name]
        f = open(self.cache_dir+'/'+name+'.vardump', 'w')
        pickle.dump(variants.get_picklable_obj(), f)
        f.close()

    def restore_func_variant_timings(self, name, file_name=None):
        variants = self.compiled_methods_with_variants[name]
        f = open(file_name or self.cache_dir+'/'+name+'.vardump', 'r')
        variants.set_from_pickled_obj(pickle.load(f))
        f.close()

    def clear_func_variant_timings(self, name):
        variants = self.compiled_methods_with_variants[name]
        variants.variant_times = {}
        variants.best_found = {}
        variants.next_variant_run = {}

    def func_variant_is_not_compilable(self, variant_name, name):
        variants = self.compiled_methods_with_variants[name]
        variants.variant_is_not_compilable(variant_name)
        
    def __getattr__(self, name):
        if name in self.compiled_methods_with_variants.keys():
            if self.dirty:
                self.compile()
            return self.func_with_variants(name)

        else:
            raise AttributeError("No method %s found; did you add it?" % name)

