from ctypes import *

# converts from a class that is a ctypes Structure into the C declaration of the datatype
class StructConverter(object):
    # we currently do not support int8/int16/etc
    _typehash_ = {'c_int':'int',
                  'c_byte': 'byte',
                  'c_char': 'char',
                  'c_char_p': 'char*',
                  'c_double': 'double',
                  'c_longdouble': 'long double',
                  'c_float': 'float',
                  'c_long': 'long',
                  'c_longlong': 'long long',
                  'c_short': 'short',
                  'c_size_t': 'size_t',
                  'c_ssize_t': 'ssize_t',
                  'c_ubyte': 'unsigned char',
                  'c_uint': 'unsigned int',
                  'c_ulong': 'unsigned long',
                  'c_ulonglong': 'unsigned long long',
                  'c_ushort': 'unsigned short',
                  'c_void_p': 'void*',
                  'c_wchar': 'wchcar_t',
                  'c_wchar_p': 'wchar_t*',
                  'c_bool': 'bool'}
                  
    def __init__(self):
        self.all_structs = {}
    
    def visitor(self, item):
        if type(item) == type(POINTER(c_int)):
            return self.visitor(item._type_) + "*"
        elif type(item) == type(c_int * 4):
            # if it is an array:
            return (self.visitor(item._type_), item._length_)
        elif item.__name__ in self._typehash_:
            return self._typehash_[item.__name__]
        else:
            if item.__name__ not in self.all_structs.keys():
                self.convert(item)
            return item.__name__
    
    def convert(self, cl):
        """Top-level function for converting from ctypes Structure to it's C++ equivalent declaration.
        
        The function returns a hash with keys corresponding to structure names encountered, and values
        corresponding to the definition of the type.
        """
        def mapfunc(x):
            ret = self.visitor(x[1])
            if type(ret) is tuple:
                return "%s %s[%s];" % (ret[0], x[0], ret[1])
            else:
                return "%s %s;" % (ret, x[0])
        
        # try to avoid infinite recursion for types defined with self-recursion or mutual recursion
        self.all_structs[cl.__name__] = None
        
        fields = map(mapfunc, cl._fields_)
        self.all_structs[cl.__name__] = "struct %s { %s };" % (cl.__name__, '\n'.join(fields))
        
        return self.all_structs