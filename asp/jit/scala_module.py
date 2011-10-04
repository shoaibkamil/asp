import os
import os.path
import subprocess

class ScalaFunction:
    def __init__(self, classname, source_dir):
        self.classname = classname
        self.source_dir = source_dir

    def __call__(self, *args, **kwargs):
        # Should support more than just floats and lists of floats
        result = subprocess.Popen(['scala', '-classpath', self.source_dir,
                  self.classname] + [str(arg) for arg in args],
                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result.wait()
        if result.returncode != 0:
            raise Error("Bad return code")
        output = result.communicate()[0]
        nums = [float(x) for x in output.split()]
        print nums
        return nums

class PseudoModule:
    '''Pretends to be a Python module that contains the generated functions.'''
    def __init__(self):
        self.__dict__["__special_functions"] = {}

    def __getattr__(self, name):
        if name in self.__dict__["__special_functions"].keys():
            return self.__dict__["__special_functions"][name]
        else:
            raise Error

    def __setattr__(self, name, value):
        self.__dict__["__special_functions"][name] = value

class ScalaModule:
    def __init__(self):
        self.mod_body = []
        self.init_body = []

    def add_to_init(self, body):
        self.init_body.extend([body])

    def add_function(self):
        # This is only for already compiled functions, I think
        pass

    def add_to_module(self, body):
        self.mod_body.extend(body)

    def add_to_preamble(self):
        pass

    def generate(self):
        s = ""
        for line in self.mod_body:
            if type(line) != str:
                raise Error("Not a string")
            s += line
        return s

    def compile(self, toolchain, debug=True, cache_dir=None):
        if cache_dir is None:
            import tempfile
            cache_dir = tempfile.gettempdir()
        else: 
            if not os.path.isdir(cache_dir):
                os.makedirs(cache_dir)

        source_string = self.generate()
        hex_checksum = self.calculate_hex_checksum(source_string)
        mod_cache_dir = os.path.join(cache_dir, hex_checksum)

        # Should we assume that if the directory exists, then we don't need to
        # recompile?
        if not os.path.isdir(mod_cache_dir):
            os.makedirs(mod_cache_dir)
            filepath = os.path.join(mod_cache_dir, "asp_tmp.scala")

            source = open(filepath, 'w')
            source.write(source_string)
            source.close()
            os.system("scalac -d %s %s" % (mod_cache_dir, filepath))
            os.remove(filepath)

        mod = PseudoModule()
        for fname in self.init_body:
            self.func = ScalaFunction(fname, mod_cache_dir)
            setattr(mod, fname, self.func)
        return mod

    # Method borrowed from codepy.jit
    def calculate_hex_checksum(self, source_string):
        try:
            import hashlib
            checksum = hashlib.md5()
        except ImportError:
            # for Python << 2.5
            import md5
            checksum = md5.new()

        checksum.update(source_string)
        #checksum.update(str(toolchain.abi_id()))
        return checksum.hexdigest()


class ScalaToolchain:
    pass
