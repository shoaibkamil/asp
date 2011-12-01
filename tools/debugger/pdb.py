import subprocess
import re
import os

class pdb(object):
    def __init__(self, python_file, python_start_line):
        self.process = subprocess.Popen(["PYTHONPATH=../../specializers/stencil:../.. python -u /usr/bin/pdb " + python_file],shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE, cwd='.')
        for line in python_start_line:
            self.process.stdin.write(line + "\n")

    # Read up to current position in output
    def sync_pos(self):
        self.process.stdin.write("print 'sync375023'\n")
        line = "\n"
        while line:
            line = self.process.stdout.readline()
            if 'sync375023' in line:
                break
        line = self.process.stdout.read(len("(Pdb) "))

    def get_current_stack(self):
        self.sync_pos()
        self.process.stdin.write("bt\n")
        while True:
            line = self.process.stdout.readline().strip()
            if len(line) > 0 and line[0] == '>':
                m = re.match(r'^> (.*)\(([0-9]+)\)([A-Za-z0-9_]+)\(\)', line)
                if m:
                    result = dict()
                    result['filename'] = m.group(1)
                    result['line_no'] = int(m.group(2))
                    result['method_name'] = m.group(3)
                    return result
                else:
                    raise RuntimeError('Could not match regex on stack line:', line)

    def next(self):
        self.process.stdin.write("n\n")

    def quit(self):
        self.process.stdin.write("quit\n")
        self.process.stdout.read() # Read to end

    def read_expr(self, expr):
        self.sync_pos()
        self.process.stdin.write("print " + expr + "\n")
        return self.process.stdout.readline().strip()

if __name__ == '__main__':
    pdb = pdb()
    for x in range(10):
        stack = pdb.get_current_stack()
        print stack['line_no']
        print 'x:', pdb.read_expr('x')
        print 'y:', pdb.read_expr('y')
        print 'out_grid[x]:', pdb.read_expr('out_grid[x]')
        print 'in_grid[y]:', pdb.read_expr('in_grid[y]')
        pdb.next()

    pdb.quit()
