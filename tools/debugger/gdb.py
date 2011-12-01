import subprocess
import re

class gdb(object):
    def __init__(self, python_file, cpp_file, cpp_start_line):
        self.process = subprocess.Popen(["PYTHONPATH=stencil:../.. gdb python"],shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE, cwd='.')
        self.process.stdin.write("run " + python_file + "\n")
        self.process.stdin.write("break " + cpp_file + ":" + str(cpp_start_line) + "\n")
        self.process.stdin.write("run " + python_file + "\n")
        self.process.stdin.write("delete 0\n")

    # Read up to current position in output
    def sync_pos(self):
        self.process.stdin.write("echo sync375023\\n\n")
        line = "\n"
        while line:
            line = self.process.stdout.readline()
            if 'sync375023' in line:
                break
        line = self.process.stdout.read(len("(gdb) "))

    def get_current_stack(self):
        self.sync_pos()
        self.process.stdin.write("back\n")
        line = self.process.stdout.readline().strip()
        m = re.match(r'^#([0-9]+)\s+(.*::)?([A-Za-z0-9_]+)\s+(\(.*\))? at (.*):([0-9]+)$', line)
        if m and m.group(1) == '0':
            result = dict()
            result['stack_frame_number'] = m.group(1)
            result['namespace'] = m.group(2)
            result['method_name'] = m.group(3)
            result['params'] = m.group(4)
            result['filename'] = m.group(5)
            result['line_no'] = int(m.group(6))
            return result
        else:
            raise RuntimeError('Could not match regex on stack line:', line)

    def next(self):
        self.process.stdin.write("next\n")

    def quit(self):
        self.process.stdin.write("quit\n")
        self.process.stdout.read() # Read to end

    def read_expr(self, expr):
        self.sync_pos()
        self.process.stdin.write("print " + expr + "\n")
        self.process.stdin.write("echo sentinel07501923\\n\n")
        line = self.process.stdout.readline().strip()
        if 'sentinel07501923' in line:
            return None
        else:
            m = re.match(r'^\$([0-9]+)\s+=\s+(.*)$', line)
            if m:
                return m.group(2)
            else:
                raise RuntimeError('Could not match regex on expression print:', line)

if __name__ == '__main__':
    gdb = gdb()
    for x in range(10):
        stack = gdb.get_current_stack()
        print stack['line_no']
        print 'x1:', gdb.read_expr('x1')
        print 'x2:', gdb.read_expr('x2')
        print 'x3:', gdb.read_expr('x3')
        gdb.next()

    gdb.quit()
