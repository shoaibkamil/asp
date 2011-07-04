import unittest2 as unittest
from stencil_python_front_end import *
from assert_utils import *
import ast

class BasicTests(unittest.TestCase):
    def test_parse(self):
        x = StencilPythonFrontEnd(dict()).parse(ast.parse(
'''
def kernel(self, in_grid, out_grid):
    for x in out_grid.interior_points():
        for y in in_grid.neighbors(x, 1):
            out_grid[x] = out_grid[x] + in_grid[y]
'''
                                               ))
        assert_has_type(x, StencilModel)

if __name__ == '__main__':
    unittest.main()
