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
        
        

if __name__ == '__main__':
    unittest.main()
