import re
import yaml
import os
import asp.jit.asp_module as asp_module
from codepy.cgen import Include


class CompilerDetector(object):
    """
    Detect if a particular compiler is available by trying to run it.
    """
    def detect(self, compiler):
        from pytools.prefork import call_capture_output
        try:
            retcode, stdout, stderr = call_capture_output([compiler, "--version"])
        except:
            return False

        return (retcode == 0)
        
class PlatformDetector(object):
    def __init__(self):
        self.rawinfo = []
        self.cuda_util_mod = asp_module.ASPModule(use_cuda=True)
        cuda_util_funcs = [ ("""
            void set_device(int dev) {
              int GPUCount;
              cudaGetDeviceCount(&GPUCount);
              if(GPUCount == 0) {
                dev = 0;
              } else if (dev >= GPUCount) {
                dev  = GPUCount-1;
              }
              cudaSetDevice(dev);
            }""", "set_device"),
            ("""
            boost::python::tuple device_compute_capability(int dev) {
              int major, minor;
              cuDeviceComputeCapability(&major, &minor, dev);
              return boost::python::make_tuple(major, minor);
            }""", "device_compute_capability"),
            ("""
            int get_device_count() {
              int count;
              cudaGetDeviceCount(&count);
              return count;
            }""", "get_device_count"),
            ("""
            int device_get_attribute( int attr, int dev) {
              int pi;
              cuDeviceGetAttribute(&pi, (CUdevice_attribute)attr, dev);
              return pi;
            }""", "device_get_attribute"),
            ("""
            size_t device_total_mem(int dev) {
                size_t bytes;
                cuDeviceTotalMem(&bytes, dev);
                return bytes;
            }""", "device_total_mem") ]
        for fbody, fname in cuda_util_funcs:
            self.cuda_util_mod.add_helper_function(fname, fbody, backend='cuda')
        self.cuda_device_id = None

    def get_num_cuda_devices(self):
        return self.cuda_util_mod.get_device_count()

    def set_cuda_device(self, device_id):
        self.cuda_device_id = device_id
        self.cuda_util_mod.set_device(device_id)
        
    def get_cuda_info(self):
        info = {}
        if self.cuda_device_id == None:
            raise RuntimeError("No CUDA device selected. Set device before querying.")
        attribute_list = [ # from CUdevice_attribute_enum at cuda.h:259
            ('max_threads_per_block',1),
            ('max_block_dim_x',2),
            ('max_block_dim_y',3),
            ('max_block_dim_z',4),
            ('max_grid_dim_x',5),
            ('max_grid_dim_y',6),
            ('max_grid_dim_z',7),
            ('max_shared_memory_per_block',8) ]
        d = self.cuda_device_id
        for key, attr in attribute_list:
            info[key] = self.cuda_util_mod.device_get_attribute(attr, d)
        info['total_mem']  = self.cuda_util_mod.device_total_mem(d)
        version = self.cuda_util_mod.device_compute_capability(d)
        info['capability'] = version
        info['supports_int32_atomics_in_global'] = False if version in [(1,0)] else True
        info['supports_int32_atomics_in_shared'] = False if version in [(1,0),(1,1)] else True
        info['supports_int64_atomics_in_global'] = False if version in [(1,0),(1,1)] else True
        info['supports_warp_vote_functions'] = False if version in [(1,0),(1,1)] else True
        info['supports_float64_arithmetic'] = False if version in [(1,0),(1,1),(1,2)] else True
        info['supports_int64_atomics_in_global'] = False if version[0] == 1 else True
        info['supports_float32_atomic_add'] = False if version[0] == 1 else True
        return info

    def get_cpu_info(self):
        self.rawinfo = self.read_cpu_info()
        info = {}
        info['numCores'] = self.parse_num_cores()
        info['vendorID'] = self.parse_cpu_info('vendor_id')
        info['model'] = int(self.parse_cpu_info('model'))
        info['cpuFamily'] = int(self.parse_cpu_info('cpu family'))
        info['cacheSize'] = int(self.parse_cpu_info('cache size'))
        info['capabilities'] = self.parse_capabilities()
        return info

    def get_compilers(self):
        return filter(CompilerDetector().detect, ["gcc", "icc", "nvcc"])

    def parse_capabilities(self):
        matcher = re.compile("flags\s+:")
        for line in self.rawinfo:
            if re.match(matcher, line):
                return line.split(":")[1].split(" ")
        
    def parse_num_cores(self):
        matcher = re.compile("processor\s+:")
        count = 0
        for line in self.rawinfo:
            if re.match(matcher, line):
                count +=1
        return count
        
    def parse_cpu_info(self, item):
        matcher = re.compile(item +"\s+:\s*(\w+)")
        for line in self.rawinfo:
            if re.match(matcher, line):
                return re.match(matcher, line).group(1)
        
    def read_cpu_info(self):
        return open("/proc/cpuinfo", "r").readlines()


class ConfigReader(object):
    """
    Interface for reading a per-user configuration file in YAML format.  The
    config file lives in ~/.asp_config.yml (on windows, ~ is equivalent to 
    \Users\<current user>).  
    
    On initialization, specify the specializer whose settings are going to be read.

    The format of the file should contain a specializer's
    settings in its own hash.  E.g.:
    specializer_foo:
      setting_one: value
      setting_two: value
    specializer_bar:
      setting_etc: value

    """
    def __init__(self, specializer):
        try:
            self.stream = open(os.path.expanduser("~")+'/.asp_config.yml')
            self.configs = yaml.load(self.stream)
        except:
            print "No configuration file ~/.asp_config.yml found."
            self.configs = {}

        self.specializer = specializer
            
        #translates from YAML file to Python dictionary


    # add functionality to iterate keys? 
    def get_option(self, key):
        """
        Given a key, return the value for that key, or None.
        """
        try:
            return self.configs[self.specializer][key]
        except KeyError:
            print "Configuration key %s not found" % key
            return None

