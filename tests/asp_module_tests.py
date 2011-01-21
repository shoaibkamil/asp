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
        self.failUnless("test" in mod.compiled_methods_with_variants.keys())
        self.failUnless("test_1" in mod.compiled_methods)

    def test_running_multiple_variants(self):
        mod = asp_module.ASPModule()
        mod.add_function_with_variants(
            ["PyObject* test_1(PyObject* a){return a;}", 
             "PyObject* test_2(PyObject* b){Py_RETURN_NONE;}"],
            "test",
            ["test_1", "test_2"])
        result1 = mod.test("a")
        result2 = mod.test("a")
        self.assertEqual(result1, "a")
        self.assertEqual(result2, None)
        self.assertNotEqual(
            mod.compiled_methods_with_variants["test"].best_found,
            False)
        
    def test_running_multiple_variants_and_inputs(self):
        mod = asp_module.ASPModule()
        mod.add_function_with_variants(
            ["PyObject* test_1(PyObject* a, PyObject* b){ long c = PyInt_AS_LONG(a); for(; c > 0; c--) b = PyNumber_Add(b,a); return a;}", 
             "PyObject* test_2(PyObject* a, PyObject* b){ long c = PyInt_AS_LONG(b); for(; c > 0; c--) a = PyNumber_Add(a,b); return b;}"] ,
            "test",
            ["test_1", "test_2"],
            lambda name, *args, **kwargs: (name, args) )
        result1 = mod.test(1,2)
        result2 = mod.test(1,2)
        result3 = mod.test(2,1)
        mod.save_func_variant_timings("test")
        mod.clear_func_variant_timings("test")
        mod.restore_func_variant_timings("test")        
        self.assertEqual(result1, 1)
        self.assertEqual(result2, 2)
        self.assertEqual(result3, 2)
        self.assertNotEqual(
            mod.compiled_methods_with_variants["test"].get_best("test",1,2), # best time found for this input
            False)
        self.assertEqual(
            mod.compiled_methods_with_variants["test"].get_best("test",7,7), # this input never previously tried
            False)
        self.assertEqual(
            mod.compiled_methods_with_variants["test"].get_best("test",2,1), # only one variant timed for this input
            False)
        result4 = mod.test(2,1)
        self.assertEqual(result4, 1)
        self.assertNotEqual(
            mod.compiled_methods_with_variants["test"].get_best("test",2,1), # now both variants have been timed
            False)
        self.assertEqual(mod.compiled_methods_with_variants["test"].get_best("test",1,2), 'test_2')
        self.assertEqual(mod.compiled_methods_with_variants["test"].get_best("test",2,1), 'test_1')

if __name__ == '__main__':
    unittest.main()
