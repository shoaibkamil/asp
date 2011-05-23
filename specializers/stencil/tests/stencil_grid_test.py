import unittest
from stencil_grid import *

class BasicTests(unittest.TestCase):
    def test_init(self):
        grid = StencilGrid([10,10])
        self.failIf(grid.dim != 2)
        self.failIf(grid.data.shape != (10,10))
        self.failIf(grid.interior != [8,8])
        self.failIf(grid.grid_variables[0] != "DIM0")

    def test_neighbor_definition_2D(self):
        grid = StencilGrid([10,10])
        # test to make sure default neighbor definition is correct
        self.failIf(grid.neighbor_definition[0] != [(0,0)])
        self.failIf(len(grid.neighbor_definition[1]) != 4)

    def test_neighbor_definition_3D(self):
        # test to make sure default neighbor definition is correct in 3D
        grid = StencilGrid([5,5,5])
        self.failIf(len(grid.neighbor_definition[1]) != 6)

    def test_neighbor_definition_1D(self):
        grid = StencilGrid([10])
        self.assertEquals(len(grid.neighbor_definition[1]), 2)

    def test_interior_iterator_1D(self):
        grid = StencilGrid([10])
        pts = [x for x in grid.interior_points()]
        self.assertEqual(len(pts), 8)


    def test_interior_iterator_2D(self):
        # 2D
        grid = StencilGrid([5,5])
        pts = [x for x in grid.interior_points()]
        self.failIf(pts[0] != [1,1])
        self.failIf(len(pts) != 9)

    def test_interior_iterator_3D(self):
        # 3D
        grid = StencilGrid([5,5,5])
        pts = [x for x in grid.interior_points()]
        self.failIf(pts[0] != [1,1,1])
        self.failIf(len(pts) != 27)
    
    def test_neighbors_iterator(self):
        grid = StencilGrid([10,10])
        self.failIf(len([x for x in grid.neighbors([1,1],1)]) != 4)

        grid = StencilGrid([5,5,5])
        self.failIf(len([x for x in grid.neighbors([1,1,1],1)]) != 6)

        grid = StencilGrid([5])
        self.assertEquals(len([x for x in grid.neighbors([1],1)]), 2)
        

if __name__ == '__main__':
    unittest.main()
