import unittest2 as unittest
from asp.tracer import *
import asp.codegen.python_ast as past

class TracedFuncTests(unittest.TestCase):
    def setUp(self):
        self.func = "def foo():\n a=10\n b=5\n"
        self.func_ast = past.parse(self.func)
    
    def test_creation(self):
        f = TracedFunc(self.func_ast)
        self.assertTrue(callable(f))
    
    def test_simple(self):
        f = TracedFunc(self.func_ast)
        f()
        self.assertEqual(f.types["a"], type(10))

    def test_with_return(self):
        func = "def foo():\n a=10\n return a\n"
        f = TracedFunc(past.parse(func))
        self.assertEqual(f(), 10)
        self.assertEqual(f.types["a"], type(10))

if __name__ == '__main__':
    unittest.main()
