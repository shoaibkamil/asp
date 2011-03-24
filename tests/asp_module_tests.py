import unittest
import asp.jit.asp_module as asp_module

class TimerTest(unittest.TestCase):
    def test_timer(self):
        pass
#         mod = asp_module.ASPModule()
#         mod.add_function("void test(){;;;;}", "test")
# #        mod.test()
#         self.failUnless("test" in mod.times.keys())
    

class SingleFuncTests(unittest.TestCase):
    pass

class MultipleFuncTests(unittest.TestCase):
    def test_adding_multiple_funcs(self):
        mod = asp_module.ASPModule()
        mod.add_function("void test(){return;}", fname="test")
        mod.add_function("void another(){return;}", fname="another")
        mod.compile()


    def test_adding_multiple_versions(self):
        mod = asp_module.ASPModule()
        mod.add_function_with_variants(
            ["void test_1(){return;}", "void test_2(){return;}"],
            "test",
            ["test_1", "test_2"])
        mod.compile()
        self.failUnless("test" in mod.compiled_methods.keys())
        self.failUnless("test_1" in mod.compiled_methods["test"])

    def test_running_multiple_variants(self):
        mod = asp_module.ASPModule()
        mod.add_function_with_variants(
            ["PyObject* test_1(PyObject* a){return a;}", 
             "PyObject* test_2(PyObject* b){Py_RETURN_NONE;}"],
            "test",
            ["test_1", "test_2"])
        result1 = mod.test("a")
        result2 = mod.test("a")
        self.assertEqual(set([result1,result2]) == set(["a", None]), True)
        self.assertNotEqual(
            mod.compiled_methods["test"].database.get_oracular_best("test"),
            False)
        
    def test_running_multiple_variants_and_inputs(self):
        mod = asp_module.ASPModule()
	key_func = lambda name, *args, **kwargs: (name, args) 
        mod.add_function_with_variants(
            ["void test_1(PyObject* a, PyObject* b){ long c = PyInt_AS_LONG(a); for(; c > 0; c--) b = PyNumber_Add(b,a); }", 
             "void test_2(PyObject* a, PyObject* b){ long c = PyInt_AS_LONG(b); for(; c > 0; c--) a = PyNumber_Add(a,b); }"] ,
            "test",
            ["test_1", "test_2"],
            key_func )
        val = 2000000
        result1 = mod.test(1,val)
        result2 = mod.test(1,val)
        result3 = mod.test(val,1)
        self.assertNotEqual(
            mod.compiled_methods["test"].database.get_oracular_best(key_func("test",1,val)), # best time found for this input
            False)
        self.assertEqual(
            mod.compiled_methods["test"].database.get_oracular_best(key_func("test",7,7)), # this input never previously tried
            False)
        self.assertEqual(
            mod.compiled_methods["test"].database.get_oracular_best(key_func("test",val,1)), # only one variant timed for this input
            False)
        result4 = mod.test(val,1)
        self.assertNotEqual(
            mod.compiled_methods["test"].database.get_oracular_best(key_func("test",val,1)), # now both variants have been timed
            False)
        self.assertEqual(mod.compiled_methods["test"].database.get_oracular_best(key_func("test",1,val)), 'test_1')
        self.assertEqual(mod.compiled_methods["test"].database.get_oracular_best(key_func("test",val,1)), 'test_2')

    def test_adding_variants_incrementally(self):
        mod = asp_module.ASPModule()
	key_func = lambda name, *args, **kwargs: (name, args) 
        mod.add_function_with_variants(
            ["PyObject* test_1(PyObject* a, PyObject* b){ long c = PyInt_AS_LONG(a); for(; c > 0; c--) b = PyNumber_Add(b,a); return a;}"], 
            "test",
            ["test_1"],
            key_func )
        result1 = mod.test(1,20000)
        self.assertNotEqual(
            mod.compiled_methods["test"].database.get_oracular_best(key_func("test",1,20000)), # best time found for this input
            False)
        mod.add_function_with_variants(
             ["PyObject* test_2(PyObject* a, PyObject* b){ long c = PyInt_AS_LONG(b); for(; c > 0; c--) a = PyNumber_Add(a,b); return b;}"] ,
            "test",
            ["test_2"] )
        self.assertEqual(
            mod.compiled_methods["test"].database.get_oracular_best(key_func("test",1,20000)), # time is no longer definitely best
            False)
        result1 = mod.test(1,20000)
        result2 = mod.test(1,20000)
        self.assertNotEqual(
            mod.compiled_methods["test"].database.get_oracular_best(key_func("test",1,20000)), # best time found again
            False)
        self.assertEqual(mod.compiled_methods["test"].database.get_oracular_best(key_func("test",1,20000)), 'test_1')

    def test_pickling_variants_data(self):
        mod = asp_module.ASPModule()
	key_func = lambda name, *args, **kwargs: (name, args) 
        mod.add_function_with_variants(
            ["PyObject* test_1(PyObject* a, PyObject* b){ long c = PyInt_AS_LONG(a); for(; c > 0; c--) b = PyNumber_Add(b,a); return a;}", 
             "PyObject* test_2(PyObject* a, PyObject* b){ long c = PyInt_AS_LONG(b); for(; c > 0; c--) a = PyNumber_Add(a,b); return b;}"] ,
            "test",
            ["test_1", "test_2"],
            key_func )
        result1 = mod.test(1,2)
        result2 = mod.test(1,2)
        result3 = mod.test(2,1)
        mod.save_method_timings("test")
        mod.clear_method_timings("test")
        mod.restore_method_timings("test")
        self.assertNotEqual(
            mod.compiled_methods["test"].database.variant_times[key_func("test",1,2)], # time found for this input
            False)
        self.assertEqual(
            key_func("test",7,7) not in mod.compiled_methods["test"].database.variant_times, # this input never previously tried
            True)
        self.assertEqual(
            len(mod.compiled_methods["test"].database.variant_times[key_func("test",2,1)]), # only one variant timed for this input
            1)

    def test_dealing_with_preidentified_compilation_errors(self):
        mod = asp_module.ASPModule()
        key_func = lambda name, *args, **kwargs: (name, args)
        mod.add_function_with_variants(
            ["PyObject* test_1(PyObject* a, PyObject* b){ long c = PyInt_AS_LONG(a); for(; c > 0; c--) b = PyNumber_Add(b,a); return a;}", 
             "PyObject* test_2(PyObject* a, PyObject* b){ /*Dummy*/}",
             "PyObject* test_3(PyObject* a, PyObject* b){ long c = PyInt_AS_LONG(b); for(; c > 0; c--) a = PyNumber_Add(a,b); return b;}"] ,
            "test",
            ["test_1", "test_2", "test_3"],
            key_func,
            [lambda name, *args, **kwargs: True]*3,
            [True, False, True],
            ['a', 'b'] )
        result1 = mod.test(1,20000)
        result2 = mod.test(1,20000)
        result3 = mod.test(1,20000)
        self.assertNotEqual(
            mod.compiled_methods["test"].database.get_oracular_best(key_func("test",1,20000)), # best time found for this input
            False)
        self.assertEqual(
            mod.compiled_methods["test"].database.variant_times[("test",(1,20000))]['test_2'], # second variant was uncompilable
            -1)

    def test_dealing_with_preidentified_runtime_errors(self):
        mod = asp_module.ASPModule()
        key_func = lambda name, *args, **kwargs: (name, args)
        mod.add_function_with_variants(
            ["PyObject* test_1(PyObject* a, PyObject* b){ long c = PyInt_AS_LONG(a); for(; c > 0; c--) b = PyNumber_Add(b,a); return a;}", 
             "PyObject* test_2(PyObject* a, PyObject* b){ long c = PyInt_AS_LONG(a); for(; c > 0; c--) b = PyNumber_Add(b,a); return a;}", 
             "PyObject* test_3(PyObject* a, PyObject* b){ long c = PyInt_AS_LONG(b); for(; c > 0; c--) a = PyNumber_Add(a,b); return b;}"] ,
            "test",
            ["test_1", "test_2", "test_3"],
            key_func,
            [lambda name, *args, **kwargs: True, lambda name, *args, **kwargs: args[1] < 10001, lambda name, *args, **kwargs: True],
            [True]*3,
            ['a', 'b'] )
        result1 = mod.test(1,20000)
        result2 = mod.test(1,20000)
        result3 = mod.test(1,20000)
        result1 = mod.test(1,10000)
        result2 = mod.test(1,10000)
        result3 = mod.test(1,10000)
        self.assertNotEqual(
            mod.compiled_methods["test"].database.get_oracular_best(key_func("test",1,20000)), # best time found for this input
            False)
        self.assertNotEqual(
            mod.compiled_methods["test"].database.get_oracular_best(key_func("test",1,10000)), # best time found for this input
            False)
        self.assertEqual(
            mod.compiled_methods["test"].database.variant_times[("test",(1,20000))]['test_2'], # second variant was unrannable for 20000
            -1)
        self.assertNotEqual(
            mod.compiled_methods["test"].database.variant_times[("test",(1,10000))]['test_2'], # second variant was runnable for 10000
            -1)

    def test_dealing_with_preidentified_runtime_errors(self):
        pass

if __name__ == '__main__':
    unittest.main()
