import unittest

from asp.config import *

class CompilerDetectorTests(unittest.TestCase):
    def test_detect(self):
        self.assertTrue(CompilerDetector().detect("gcc"))

        self.assertFalse(CompilerDetector().detect("lkasdfj"))

class CPUInfoTests(unittest.TestCase):

    def test_num_cores(self):
        def read_cpu_info(self):
            return open("tests/cpuinfo").readlines()
        
        PlatformDetector.read_cpu_info = read_cpu_info
        pd = PlatformDetector()

        info = pd.get_cpu_info()
        self.assertEqual(info['numCores'], 8)
    
    def test_vendor_and_model(self):
        def read_cpu_info(self):
            return open("tests/cpuinfo").readlines()
        
        PlatformDetector.read_cpu_info = read_cpu_info
        pd = PlatformDetector()

        info = pd.get_cpu_info()
        self.assertEqual(info['vendorID'], "GenuineIntel")
        self.assertEqual(info['model'], 30)
        self.assertEqual(info['cpuFamily'], 6)

    def test_cache_size(self):
        def read_cpu_info(self):
            return open("tests/cpuinfo").readlines()
        
        PlatformDetector.read_cpu_info = read_cpu_info
        pd = PlatformDetector()

        info = pd.get_cpu_info()
        self.assertEqual(info['cacheSize'], 8192)
    
    def test_capabilities(self):
        def read_cpu_info(self):
            return open("tests/cpuinfo").readlines()
        
        PlatformDetector.read_cpu_info = read_cpu_info
        pd = PlatformDetector()
       
        info = pd.get_cpu_info()
        self.assertEqual(info['capabilities'].count("sse"), 1)

    def test_compilers(self):
        compilers = PlatformDetector().get_compilers()
        self.assertTrue("gcc" in compilers)

class GPUInfoTest(unittest.TestCase):

    def test_properties(self):
        pd = PlatformDetector()
        compilers = pd.get_compilers()
        if "nvcc" in compilers and pd.get_num_cuda_devices() > 0:
            info = {}
            pd.set_cuda_device(0)
            info = pd.get_cuda_info()
            self.assertTrue(info['total_mem'] > 0)
        else: self.assertTrue(True) # Undesirable to have the test fail on machinces without GPUs

if __name__ == '__main__':
    unittest.main()
