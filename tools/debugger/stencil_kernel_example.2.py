from stencil_kernel import *
import stencil_grid
import numpy

class ExampleKernel(object):
    def kernel(self, in_grid, out_grid):
        for x in out_grid.interior_points():
            for y in in_grid.neighbors(x, 1):
                out_grid[x] = out_grid[x] + in_grid[y]

in_grid = StencilGrid([5,5])
for x in range(0,5):
    for y in range(0,5):
        in_grid.data[x,y] = x + y

out_grid = StencilGrid([5,5])
ExampleKernel().kernel(in_grid, out_grid)
print out_grid
