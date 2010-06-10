import unittest
from stencil_grid import *

class BasicTests(unittest.TestCase):
    def test_init(self):
        grid = StencilGrid([10,10])
        self.failIf(grid.dim != 2)
        self.failIf(grid.data.shape != (10,10))
        self.failIf(grid.interior != [8,8])
        self.failIf(grid.grid_variables[0] != "DIM0")
        # test to make sure default neighbor definition is correct
        self.failIf(grid.neighbor_definition[0] != ["DIM0","DIM1"])
        self.failIf(len(grid.neighbor_definition[1]) != 5)
        # test to make sure default neighbor definition is correct in 3D
        grid = StencilGrid([5,5,5])
        self.failIf(len(grid.neighbor_definition[1]) != 7)


    def test_interior_iterator(self):
        # 2D
        grid = StencilGrid([5,5])
        pts = [x for x in grid.interior_points()]
        self.failIf(pts[0] != [1,1])
        self.failIf(len(pts) != 9)
        # 3D
        grid = StencilGrid([5,5,5])
        pts = [x for x in grid.interior_points()]
        self.failIf(pts[0] != [1,1,1])
        self.failIf(len(pts) != 27)
    
    def test_neighbors_iterator(self):
        grid = StencilGrid([10,10])
        self.failIf(len([x for x in grid.neighbors([1,1],1)]) != 5)

        grid = StencilGrid([5,5,5])
        self.failIf(len([x for x in grid.neighbors([1,1,1],1)]) != 7)
        

if __name__ == '__main__':
    unittest.main()
