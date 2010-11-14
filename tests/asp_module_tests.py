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


if __name__ == '__main__':
    unittest.main()
