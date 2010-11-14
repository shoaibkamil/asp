import unittest
import asp.jit.asp_module as asp_module

class TimerTest(unittest.TestCase):
    def test_timer(self):
        mod = asp_module.ASPModule()
        mod.add_function("void test(){;;;;}", "test")
        mod.test()
        self.failUnless("test" in mod.times.keys())
    

class SingleFuncTests(unittest.TestCase):
    pass

class MultipleFuncTests(unittest.TestCase):
    def test_adding_multiple_versions(self):
        pass

if __name__ == '__main__':
    unittest.main()
