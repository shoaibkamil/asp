import unittest
from stencil_grid import *

class BasicTests(unittest.TestCase):
    def test_init(self):
        grid = StencilGrid([10,10])
        self.failIf(grid.dim != 2)
        self.failIf(grid.data.shape != (10,10))
        self.failIf(grid.interior != [8,8])

    def test_iterator(self):
        grid = StencilGrid([5,5])
        pts = [x for x in grid.interior_points()]
        self.failIf(pts[0] != [1,1])
        self.failIf(len(pts) != 9)

if __name__ == '__main__':
    unittest.main()
