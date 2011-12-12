import re
import yaml
import os

class CompilerDetector(object):
    """
    Detect if a particular compiler is available by trying to run it.
    """
    def detect(self, compiler):
        import subprocess
        try:
            retcode = subprocess.call([compiler, "--version"])
        except:
            return False

        return (retcode == 0)
        

class PlatformDetector(object):
    def __init__(self):
        self.rawinfo = []

    def get_gpu_nfo(self):
        raise NotImplementedError

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
    \Users\<current user>).  The format of the file should contain a specializer's
    settings in its own hash.  E.g.:
    specializer_foo:
      setting_one: value
      setting_two: value
    specializer_bar:
      setting_etc: value

    """
    def __init__(self):
        try:
            self.stream = open(os.path.expanduser("~")+'/.asp_config.yml')
            self.configs = yaml.load(self.stream)
        except:
            print "No configuration file ~/.asp_config.yml found."
            self.configs = {}
            
        #translates from YAML file to Python dictionary

    # given a key, return corresponding configs
    # add functionality to iterate keys? 
    def get_option(self, key):
        try:
            return self.configs[key]
        except KeyError:
            print "Configuration key %s not found" % key
            return None

