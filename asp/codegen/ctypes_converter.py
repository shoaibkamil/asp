from ctypes import *

# converts from a class that is a ctypes Structure into the C declaration of the datatype
class StructConverter(object):
    # we currently do not support int8/int16/etc
    #FIXME: do we need to support nested definitions?
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
    
    def visitor(self, item):
        if type(item) == type(POINTER(c_int)):
            return self.visitor(item._type_) + "*"
        else:
            return self._typehash_[item.__name__]
    
    def convert(self, cl):
        
        fields = map(lambda x: "%s %s;" % (self.visitor(x[1]), x[0]), cl._fields_)
        return "struct %s { %s };" % (cl.__name__, '\n'.join(fields))