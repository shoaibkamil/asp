import codepy, codepy.jit, codepy.toolchain, codepy.bpl, codepy.cuda
from asp.util import *
import asp.codegen.cpp_ast as cpp_ast
import pickle
from variant_history import *
import sqlite3

class ASPDB(object):

    def __init__(self, specializer):
        """
        specializer must be specified so we avoid namespace collisions.
        """
        self.specializer = specializer
        # currently creating an in-memory db, but eventually this should get written to disk
        self.connection = sqlite3.connect(":memory:")

    def create_specializer_table(self):
        self.connection.execute('create table '+self.specializer+' (fname text, key text, perf real)')
        self.connection.commit()

    def close(self):
        self.connection.close()

    def key_function(self, *args, **kwargs):
        """
        Function to generate keys.  This should almost always be overridden by a specializer, to make
        sure the information stored in the key is actually useful.
        """
        import hashlib
        return hashlib.md5(str(args)+str(kwargs)).hexdigest()

    def table_exists(self):
        """
        Test if a table corresponding to this specializer exists.
        """
        cursor = self.connection.cursor()
        cursor.execute('select name from sqlite_master where name="%s"' % self.specializer)
        result = cursor.fetchall()
        return len(result) > 0

    def insert(self, fname, key, value):
        if (not self.table_exists()):
                self.create_specializer_table()
        self.connection.execute('insert into '+self.specializer+' values (?,?,?)',
            (fname, key, value))
        self.connection.commit()

    def get(self, fname, key=None):
        """
        Return a list of entries.  If key is not specified, all entries from
        fname are returned.
        """
        if (not self.table_exists()):
            self.create_specializer_table()
            return []

        cursor = self.connection.cursor()
        if key:
            cursor.execute('select * from '+self.specializer+' where fname=? and key=?', 
                (fname,key))
        else:
            cursor.execute('select * from '+self.specializer+' where fname=?', 
                (fname,))
        return cursor.fetchall()


class SpecializedFunction(object):
    """
    Class that encapsulates a function that is specialized.  It keeps track of variants,
    their timing information, which backend, and whether the function is a helper function
    or not.
    """
    
    def __init__(self, name, backend, db, variant_names=[], variant_funcs=[]):
        self.name = name
        self.backend = backend
        self.db = db
        self.variant_names = []
        self.variant_funcs = []
        self.variant_times = self.db.get(name)
        
        for x in xrange(len(variant_names)):
            self.add_variant(variant_names[x], variant_funcs[x])

    def add_variant(self, variant_name, variant_func):
        """
        Add a variant of this function.  Must have same call signature.  Variant names must be unique.
        The variant_func parameter should be a CodePy Function object or a string defining the function.
        """
        if variant_name in self.variant_names:
            raise Exception("Attempting to add a variant with an already existing name %s to %s" %
                            (variant_name, self.name))
        self.variant_names.append(variant_name)
        self.variant_funcs.append(variant_func)
        
        if isinstance(variant_func, str):
            self.backend.module.add_to_module([cpp_ast.Line(variant_func)])
            self.backend.module.add_to_init([cpp_ast.Statement("boost::python::def(\"%s\", &%s)" % (variant_name, variant_name))])
        else:
            self.backend.module.add_function(variant_func)

        self.backend.dirty = True

    def __call__(self, *args, **kwargs):
        """
        Calling an instance SpecializedFunction will actually call either the next variant to test,
        or the already-determined best variant.
        """
        if self.backend.dirty:
            self.backend.compile()

        if len(self.variant_times) == len(self.variant_names):
            return self.backend.get_compiled_function(self.variant_names[0]).__call__(*args, **kwargs)
        else:
            import time
            
            which = len(self.variant_times)
            start = time.time()
            ret_val = self.backend.get_compiled_function(self.variant_names[which]).__call__(*args, **kwargs)

            elapsed = time.time() - start
            self.variant_times.append(elapsed)
            #FIXME: where should key function live?
            self.db.insert(self.name, self.db.key_function(args, kwargs), elapsed)
            return ret_val

class HelperFunction(SpecializedFunction):
    """
    HelperFunction defines a SpecializedFunction that is not timed, and usually not called directly
    (although it can be).
    """
    def __init__(self, name, func, backend):
        self.name = name
        self.backend = backend
        self.variant_names, self.variant_funcs = [], []
        self.add_variant(name, func)

    def __call__(self, *args, **kwargs):
        if self.backend.dirty:
            self.backend.compile()
        return self.backend.get_compiled_function(self.name).__call__(*args, **kwargs)



class ASPModule(object):

    class ASPBackend(object):
        """
        Class to encapsulate a backend for Asp.  A backend is the combination of a CodePy module
        (which contains the actual functions) and a CodePy compiler toolchain.
        """
        def __init__(self, module, toolchain):
            self.module = module
            self.toolchain = toolchain
            self.compiled_module = None
            self.cache_dir="cache"
            self.dirty = True

        def compile(self):
            """
            Trigger a compile of this backend.  Note that CUDA needs to know about the C++
            backend as well.
            """
            if isinstance(self.module, codepy.cuda.CudaModule):
                self.compiled_module = self.backends["cuda"].module.compile(self.module.boost_module,
                                                                            self.backends["cuda"].toolchain,
                                                                            debug=True, cache_dir=self.cache_dir)
            else:
                self.compiled_module = self.module.compile(self.toolchain,
                                                           debug=True, cache_dir=self.cache_dir)
            self.dirty = False

        def get_compiled_function(self, name):
            """
            Return a callable for a raw compiled function (that is, this must be a variant name rather than
            a function name).
            """
            try:
                func = getattr(self.compiled_module, name)
            except:
                raise Error("Function %s not found in compiled module." % (name,))

            return func

    
    #FIXME: specializer should be required.
    def __init__(self, specializer="default_specializer", key_func=None, use_cuda=False):
        self.specialized_functions= {}
        self.helper_method_names = []

        self.db = ASPDB(specializer)
        if key_func:
            self.set_key_function(key_func)

        self.cache_dir = "cache"
        self.dirty = False
        self.timing_enabled = True
        self.use_cuda = use_cuda

        self.backends = {}
        self.backends["c++"] = ASPModule.ASPBackend(codepy.bpl.BoostPythonModule(),
                                          codepy.toolchain.guess_toolchain())
        if use_cuda:
            self.backends["cuda"] = ASPModule.ASPBackend(codepy.cuda.CudaModule(self.backends["c++"].module),
                                               codepy.toolchain.guess_nvcc_toolchain())
            self.backends["cuda"].module.add_to_preamble([cpp_ast.Include('cuda.h', False)])


    def set_key_function(self, key_func):
        """
        Set the key function to use for entering things into the db. key_func should be a function that
        takes (self, *args, **kwargs) and returns a string.
        """
        self.db.key_function = key_func



    def add_library(self, feature, include_dirs, library_dirs=[], libraries=[], backend="c++"):
        self.backends[backend].toolchain.add_library(feature, include_dirs, library_dirs, libraries)
        
    def add_cuda_library(self, feature, include_dirs, library_dirs=[], libraries=[]):
        """
        Deprecated.  Use add_library(..., backend="cuda")
        """
        self.add_library(feature, include_dirs, library_dirs, libraries, backend="cuda")

    def add_cuda_arch_spec(self, arch):
        archflag = '-arch='
        if 'sm_' not in arch: archflag += 'sm_' 
        archflag += arch
        self.backends["cuda"].toolchain.cflags += [archflag]

    def add_header(self, include_file, backend="c++"):
        self.backends[backend].module.add_to_preamble([cpp_ast.Include(include_file, False)])

    def add_cuda_header(self, include_file):
        """
        Deprecated.  Use add_header(..., backend="cuda").
        """
        self.add_header(include_file, backend="cuda")

    def add_to_preamble(self, pa, backend="c++"):
        if isinstance(pa, str):
            pa = [cpp_ast.Line(pa)]
        self.backends[backend].module.add_to_preamble(pa)

    def add_to_cuda_preamble(self, pa):
        """
        Deprecated.  Use add_to_preamble(..., backend="cuda").
        """
        if isinstance(pa, str):
            pa = [cpp_ast.Line(pa)]
        self.add_to_preamble(pa, backend="cuda")
        
    def add_to_init(self, stmt, backend="c++"):
        if isinstance(stmt, str):
            stmt = [cpp_ast.Line(stmt)]
        self.backends[backend].module.add_to_init(stmt)

    def add_to_cuda_module(self, block):
        #FIXME: figure out use case for this and replace
        if isinstance(block, str):
            block = [cpp_ast.Line(block)]
        self.backends["cuda"].module.add_to_module(block)
        

    def get_name_from_func(self, func):
        """
        returns the name of a function from a CodePy FunctionBody object
        """
        return func.fdecl.subdecl.name


#    def add_function_helper(self, func, fname=None, cuda_func=False, backend="c++"):
        #FIXME: want to deprecate cuda_func parameter.  this should just pickup the module
        #from the backend parameter.

        # OLD code:
        # if cuda_func:
        #     module = self.backends["cuda"].module
        # else:
        #     module = self.backends["c++"].module
        
        # if isinstance(func, str):
        #     if fname == None:
        #         raise Exception("Cannot add a function as a string without specifying the function's name")
        #     module.add_to_module([cpp_ast.Line(func)])
        #     module.add_to_init([cpp_ast.Statement(
        #                 "boost::python::def(\"%s\", &%s)" % (fname, fname))])
        # else:
        #     module.add_function(func)
        # self.dirty = True


        

    # def add_function_with_variants(self, variant_funcs, func_name, variant_names, key_maker=lambda name, *args, **kwargs: (name), limit_funcs=None, compilable=None, param_names=None, cuda_func=False):
        
    #     limit_funcs = limit_funcs or [lambda name, *args, **kwargs: True]*len(variant_names) 
    #     compilable = compilable or [True]*len(variant_names)
    #     param_names = param_names or ['Unknown']*len(variant_names)
    #     method_info = self.compiled_methods.get(func_name, None)
    #     if not method_info:
    #         method_info = CodeVariants(variant_names, key_maker, param_names)
    #         method_info.limiter.append(variant_names, limit_funcs, compilable)
    #     else:
    #         method_info.append(variant_names)
    #         method_info.database.clear_oracle()
    #         method_info.limiter.append(variant_names, limit_funcs, compilable)
    #     for x in range(0,len(variant_funcs)):
    #         self.add_function_helper(variant_funcs[x], fname=variant_names[x], cuda_func=cuda_func)
    #     self.compiled_methods[func_name] = method_info

    # def add_function(self, funcs, fname=None, variant_names=None, cuda_func=False):
    #     """
    #     self.add_function(func) takes func as either a generable AST or a string, or
    #     list of variants in either format.
    #     """
    #     if variant_names:
    #         self.add_function_with_variants(funcs, fname, variant_names, cuda_func=cuda_func)
    #     else:
    #         variant_funcs = [funcs]
    #         if not fname:
    #             fname = self.get_name_from_func(funcs)
    #         variant_names = [fname]
    #         self.add_function_with_variants(variant_funcs, fname, variant_names, cuda_func=cuda_func)

    def add_function(self, fname, funcs, variant_names=None, backend="c++"):
        """
        Add a specialized function to the Asp module.  funcs can be a list of variants, but then
        variant_names is required (also a list).  Each item in funcs should be a string function or
        a cpp_ast FunctionDef.
        """
        if not isinstance(funcs, list):
            funcs = [funcs]
            variant_names = [fname]

        self.specialized_functions[fname] = SpecializedFunction(fname, self.backends[backend], self.db, variant_names,
                                                                variant_funcs=funcs)

    def add_helper_function(self, fname, func, backend="c++"):
        """
        Add a helper function, which is a specialized function that it not timed and has a single variant.
        """
        self.specialized_functions[fname] = HelperFunction(fname, func, self.backends[backend])
        


                
    def compile(self):
        if self.use_cuda:
            self.compiled_module = self.backends["cuda"].module.compile(self.backends["c++"].module,
                                                                        self.backends["cuda"].toolchain,
                                                                        debug=True, cache_dir=self.cache_dir)
        else:
            self.compiled_module = self.backends["c++"].module.compile(self.backends["c++"].toolchain,
                                                                       debug=True, cache_dir=self.cache_dir)
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
        if name in self.specialized_functions:
            if self.dirty:
                self.compile()
            return self.specialized_functions[name]
        elif name in self.helper_method_names:
            return self.helper_func(name)
        else:
            raise AttributeError("No method %s found; did you add it?" % name)

