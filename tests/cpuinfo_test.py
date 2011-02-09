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
        
        

if __name__ == '__main__':
    unittest.main()
