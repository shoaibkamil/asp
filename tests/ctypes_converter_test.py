import unittest
from ctypes import *
from asp.codegen.ctypes_converter import *

def normalize(a):
    return ''.join(a.split())

class BasicConversionTests(unittest.TestCase):
    def test_int_field(self):
        class Foo(Structure):
            _fields_ = [("x", c_int), ("y", c_int)]
            
        self.assertEquals(normalize(StructConverter().convert(Foo)), normalize("struct Foo { int x; int y; };"))
        
    def test_mixed_fields(self):
        class Foo2(Structure):
            _fields_ = [("x", c_int), ("y", c_bool), ("z", c_void_p), ("a", c_ushort)]
            
        self.assertEquals(normalize(StructConverter().convert(Foo2)), normalize("struct Foo2 { int x; bool y; void* z; unsigned short a; };"))

    def test_pointer_field(self):
        class Foo(Structure):
            _fields_ = [("x", c_int), ("y", POINTER(c_int)), ("z", POINTER(POINTER(c_double)))]
            
        self.assertEquals(normalize(StructConverter().convert(Foo)), normalize("struct Foo { int x; int* y; double** z;};"))
    
    def test_array_field(self):
        class Foo(Structure):
            _fields_ = [("x", c_int * 4)]
        
        self.assertEquals(normalize(StructConverter().convert(Foo)), normalize("struct Foo {int x[4];};"))
        
        
if __name__ == '__main__':
    unittest.main()
