import codepy, codepy.jit, codepy.toolchain, codepy.bpl, codepy.cuda
from asp.util import *
import asp.codegen.cpp_ast as cpp_ast
import pickle
from variant_history import *
import sqlite3
import asp

class ASPDB(object):

    def __init__(self, specializer, persistent=False):
        """
        specializer must be specified so we avoid namespace collisions.
        """
        self.specializer = specializer

        if persistent:
            # create db file or load db
            # create a per-user cache directory
            import tempfile, os
            if os.name == 'nt':
                username = os.environ['USERNAME']
            else:
                username = os.environ['LOGNAME']

            self.cache_dir = tempfile.gettempdir() + "/asp_cache_" + username

            if not os.access(self.cache_dir, os.F_OK):
                os.mkdir(self.cache_dir)
            self.db_file = self.cache_dir + "/aspdb.sqlite3"
            self.connection = sqlite3.connect(self.db_file)
            self.connection.execute("PRAGMA temp_store = MEMORY;")
            self.connection.execute("PRAGMA synchronous = OFF;")
            
        else:
            self.db_file = None
            self.connection = sqlite3.connect(":memory:")


    def create_specializer_table(self):
        self.connection.execute('create table '+self.specializer+' (fname text, variant text, key text, perf real)')
        self.connection.commit()

    def close(self):
        self.connection.close()

    def table_exists(self):
        """
        Test if a table corresponding to this specializer exists.
        """
        cursor = self.connection.cursor()
        cursor.execute('select name from sqlite_master where name="%s"' % self.specializer)
        result = cursor.fetchall()
        return len(result) > 0

    def insert(self, fname, variant, key, value):
        if (not self.table_exists()):
                self.create_specializer_table()
        self.connection.execute('insert into '+self.specializer+' values (?,?,?,?)',
            (fname, variant, key, value))
        self.connection.commit()

    def get(self, fname, variant=None, key=None):
        """
        Return a list of entries.  If key and variant not specified, all entries from
        fname are returned.
        """
        if (not self.table_exists()):
            self.create_specializer_table()
            return []

        cursor = self.connection.cursor()
        query = "select * from %s where fname=?" % (self.specializer,)
        params = (fname,)

        if variant:
            query += " and variant=?"
            params += (variant,)
        
        if key:
            query += " and key=?"
            params += (key,)

        cursor.execute(query, params)

        return cursor.fetchall()

    def update(self, fname, variant, key, value):
        """
        Updates an entry in the db.  Overwrites the timing information with value.
        If the entry does not exist, does an insert.
        """
        if (not self.table_exists()):
            self.create_specializer_table()
            self.insert(fname, variant, key, value)
            return

        # check if the entry exists
        query = "select count(*) from "+self.specializer+" where fname=? and variant=? and key=?;"
        cursor = self.connection.cursor()
        cursor.execute(query, (fname, variant, key))
        count = cursor.fetchone()[0]
        
        # if it exists, do an update, otherwise do an insert
        if count > 0:
            query = "update "+self.specializer+" set perf=? where fname=? and variant=? and key=?"
            self.connection.execute(query, (value, fname, variant, key))
            self.connection.commit()
        else:
            self.insert(fname, variant, key, value)


    def delete(self, fname, variant, key):
        """
        Deletes an entry from the db.
        """
        if (not self.table_exists()):
            return

        query = "delete from "+self.specializer+" where fname=? and variant=? and key=?"
        self.connection.execute(query, (fname, variant, key))
        self.connection.commit()

    def destroy_db(self):
        """
        Delete the database.
        """
        if not self.db_file:
            return True

        import os
        try:
            self.close()
            os.remove(self.db_file)
        except:
            return False
        else:
            return True


class SpecializedFunction(object):
    """
    Class that encapsulates a function that is specialized.  It keeps track of variants,
    their timing information, which backend, functions to determine if a variant
    can run, as well as a function to generate keys from parameters.

    The signature for any run_check function is run(*args, **kwargs).
    The signature for the key function is key(self, *args, **kwargs), where the args/kwargs are
    what are passed to the specialized function.

    """
    
    def __init__(self, name, backend, db, variant_names=[], variant_funcs=[], run_check_funcs=[], 
                 key_function=None):
        self.name = name
        self.backend = backend
        self.db = db
        self.variant_names = []
        self.variant_funcs = []
        self.run_check_funcs = []
        
        if variant_names != [] and run_check_funcs == []:
            run_check_funcs = [lambda *args,**kwargs: True]*len(variant_names)
        
        for x in xrange(len(variant_names)):
            self.add_variant(variant_names[x], variant_funcs[x], run_check_funcs[x])

        if key_function:
            self.key = key_function

    def key(self, *args, **kwargs):
        """
        Function to generate keys.  This should almost always be overridden by a specializer, to make
        sure the information stored in the key is actually useful.
        """
        import hashlib
        return hashlib.md5(str(args)+str(kwargs)).hexdigest()


    def add_variant(self, variant_name, variant_func, run_check_func=lambda *args,**kwargs: True):
        """
        Add a variant of this function.  Must have same call signature.  Variant names must be unique.
        The variant_func parameter should be a CodePy Function object or a string defining the function.
        The run_check_func parameter should be a lambda function with signature run(*args,**kwargs).
        """
        if variant_name in self.variant_names:
            raise Exception("Attempting to add a variant with an already existing name %s to %s" %
                            (variant_name, self.name))
        self.variant_names.append(variant_name)
        self.variant_funcs.append(variant_func)
        self.run_check_funcs.append(run_check_func)
        
        if isinstance(variant_func, str):
            self.backend.module.add_to_module([cpp_ast.Line(variant_func)])
            self.backend.module.add_to_init([cpp_ast.Statement("boost::python::def(\"%s\", &%s)" % (variant_name, variant_name))])
        else:
            self.backend.module.add_function(variant_func)

        self.backend.dirty = True

    def pick_next_variant(self, *args, **kwargs):
        """
        Logic to pick the next variant to run.  If all variants have been run, then this should return the
        fastest variant.
        """
        # get variants that have run
        already_run = self.db.get(self.name, key=self.key(*args, **kwargs))


        if already_run == []:
            already_run_variant_names = []
        else:
            already_run_variant_names = map(lambda x: x[1], already_run)

        # which variants haven't yet run
        candidates = set(self.variant_names) - set(already_run_variant_names)

        # of these candidates, which variants *can* run
        for x in candidates:
            if self.run_check_funcs[self.variant_names.index(x)](*args, **kwargs):
                return x

        # if none left, pick fastest from those that have already run
        return sorted(already_run, lambda x,y: cmp(x[3],y[3]))[0][1]

    def __call__(self, *args, **kwargs):
        """
        Calling an instance of SpecializedFunction will actually call either the next variant to test,
        or the already-determined best variant.
        """
        if self.backend.dirty:
            self.backend.compile()

        which = self.pick_next_variant(*args, **kwargs)

        import time
        start = time.time()
        ret_val = self.backend.get_compiled_function(which).__call__(*args, **kwargs)
        elapsed = time.time() - start
        #FIXME: where should key function live?
        #print "doing update with %s, %s, %s, %s" % (self.name, which, self.key(args, kwargs), elapsed)
        self.db.update(self.name, which, self.key(*args, **kwargs), elapsed)
        #TODO: Should we use db.update instead of db.insert to avoid O(N) ops on already_run_variant_names = map(lambda x: x[1], already_run)?

        return ret_val

class HelperFunction(SpecializedFunction):
    """
    HelperFunction defines a SpecializedFunction that is not timed, and usually not called directly
    (although it can be).
    """
    def __init__(self, name, func, backend):
        self.name = name
        self.backend = backend
        self.variant_names, self.variant_funcs, self.run_check_funcs = [], [], []
        self.add_variant(name, func)

    def __call__(self, *args, **kwargs):
        if self.backend.dirty:
            self.backend.compile()
        return self.backend.get_compiled_function(self.name).__call__(*args, **kwargs)

class ASPBackend(object):
    """
    Class to encapsulate a backend for Asp.  A backend is the combination of a CodePy module
    (which contains the actual functions) and a CodePy compiler toolchain.
    """
    def __init__(self, module, toolchain, cache_dir):
        self.module = module
        self.toolchain = toolchain
        self.compiled_module = None
        self.cache_dir = cache_dir
        self.dirty = True
        self.compilable = True

    def compile(self):
        """
        Trigger a compile of this backend.  Note that CUDA needs to know about the C++
        backend as well.
        """
        if not self.compilable: return
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
            raise AttributeError("Function %s not found in compiled module." % (name,))

        return func


class ASPModule(object):
    """
    ASPModule is the main coordination class for specializers.  A specializer creates an ASPModule to contain
    all of its specialized functions, and adds functions/libraries/etc to the ASPModule.

    ASPModule uses ASPBackend instances for each backend, ASPDB for its backing db for recording timing info,
    and instances of SpecializedFunction and HelperFunction for specialized and helper functions, respectively.
    """

    #FIXME: specializer should be required.
    def __init__(self, specializer="default_specializer", cache_dir=None, use_cuda=False, use_cilk=False):
            
        self.specialized_functions= {}
        self.helper_method_names = []

        self.db = ASPDB(specializer)
        
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            # create a per-user cache directory
            import tempfile, os
            if os.name == 'nt':
                username = os.environ['USERNAME']
            else:
                username = os.environ['LOGNAME']

            self.cache_dir = tempfile.gettempdir() + "/asp_cache_" + username
            if not os.access(self.cache_dir, os.F_OK):
                os.mkdir(self.cache_dir)

        self.dirty = False
        self.timing_enabled = True
        self.use_cuda = use_cuda

        self.backends = {}
        self.backends["c++"] = ASPBackend(codepy.bpl.BoostPythonModule(),
                                          codepy.toolchain.guess_toolchain(),
                                          self.cache_dir)
        if use_cuda:
            self.backends["cuda"] = ASPBackend(codepy.cuda.CudaModule(self.backends["c++"].module),
                                               codepy.toolchain.guess_nvcc_toolchain(),
                                               self.cache_dir)
            self.backends["cuda"].module.add_to_preamble([cpp_ast.Include('cuda.h', False)])

        if use_cilk:
            self.backends["cilk"] = self.backends["c++"]
            self.backends["cilk"].toolchain.cc = "icc"



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

    def add_header(self, include_file, brackets=False, backend="c++"):
        """
        Add a header (e.g. #include "foo.h") to the module source file.
        With brackets=True, it will be C++-style #include <foo> instead.
        """
        self.backends[backend].module.add_to_preamble([cpp_ast.Include(include_file, brackets)])

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
        

    def add_function(self, fname, funcs, variant_names=[], run_check_funcs=[], key_function=None, 
                     backend="c++"):
        """
        Add a specialized function to the Asp module.  funcs can be a list of variants, but then
        variant_names is required (also a list).  Each item in funcs should be a string function or
        a cpp_ast FunctionDef.
        """
        if not isinstance(funcs, list):
            funcs = [funcs]
            variant_names = [fname]

        self.specialized_functions[fname] = SpecializedFunction(fname, self.backends[backend], self.db, variant_names,
                                                                variant_funcs=funcs, 
                                                                run_check_funcs=run_check_funcs,
                                                                key_function=key_function)

    def add_helper_function(self, fname, func, backend="c++"):
        """
        Add a helper function, which is a specialized function that it not timed and has a single variant.
        """
        self.specialized_functions[fname] = HelperFunction(fname, func, self.backends[backend])


    def __getattr__(self, name):
        if name in self.specialized_functions:
            if self.dirty:
                self.compile()
            return self.specialized_functions[name]
        elif name in self.helper_method_names:
            return self.helper_func(name)
        else:
            raise AttributeError("No method %s found; did you add it?" % name)

    def generate(self):
        """
        Utility function for, during development, dumping out the generated
        source from all the underlying backends.
        """
        src = ""
        for x in self.backends.keys():
            src += str(self.backends[x].module.generate())

        return src

