import unittest

import asp.config

class CPUInfoTests(unittest.TestCase):

    def test_numCores(self):
        def readCPUInfo(self):
            return open("tests/cpuinfo").readlines()
        
        asp.config.PlatformDetector.readCPUInfo = readCPUInfo
        pd = asp.config.PlatformDetector()

        info = pd.getCPUInfo()
        self.assertEqual(info['numCores'], 8)
    
    def testVendorandModel(self):
        def readCPUInfo(self):
            return open("tests/cpuinfo").readlines()
        
        asp.config.PlatformDetector.readCPUInfo = readCPUInfo
        pd = asp.config.PlatformDetector()

        info = pd.getCPUInfo()
        self.assertEqual(info['vendorID'], "GenuineIntel")
        self.assertEqual(info['model'], 30)
        self.assertEqual(info['cpuFamily'], 6)

    def testCacheSize(self):
        def readCPUInfo(self):
            return open("tests/cpuinfo").readlines()
        
        asp.config.PlatformDetector.readCPUInfo = readCPUInfo
        pd = asp.config.PlatformDetector()

        info = pd.getCPUInfo()
        self.assertEqual(info['cacheSize'], 8192)
        

if __name__ == '__main__':
    unittest.main()
