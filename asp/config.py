import re

class PlatformDetector(object):
    def __init__(self):
        self.rawinfo = []


    def getCPUInfo(self):
        self.rawinfo = self.readCPUInfo()
        info = {}
        info['numCores'] = self.parseNumCores()
        info['vendorID'] = self.parseCPUInfo('vendor_id')
        info['model'] = int(self.parseCPUInfo('model'))
        info['cpuFamily'] = int(self.parseCPUInfo('cpu family'))
        info['cacheSize'] = int(self.parseCPUInfo('cache size'))
        return info

    def parseNumCores(self):
        matcher = re.compile("processor\s+:")
        count = 0
        for line in self.rawinfo:
            if re.match(matcher, line):
                count +=1
        return count
        
    def parseCPUInfo(self, item):
        matcher = re.compile(item +"\s+:\s*(\w+)")
        for line in self.rawinfo:
            if re.match(matcher, line):
                return re.match(matcher, line).group(1)
        
    def readCPUInfo(self):
        pass
