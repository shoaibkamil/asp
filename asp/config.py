import re

class PlatformDetector(object):
    def __init__(self):
        self.rawinfo = []


    def getCPUInfo(self):
        self.rawinfo = self.readCPUInfo()
        info = {}
        info['numCores'] = self.parseNumCores()
        return info

    def parseNumCores(self):
        matcher = re.compile("processor\s+:")
        count = 0
        for line in self.rawinfo:
            if re.match(matcher, line):
                count +=1
        return count
        

    def readCPUInfo(self):
        pass
