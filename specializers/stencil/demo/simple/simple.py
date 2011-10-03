from stencil_kernel import *
import stencil_grid
import numpy

class Kernel(StencilKernel):
    def kernel(self, in_img, out_img):
        for x in out_img.interior_points():
            for y in in_img.neighbors(x, 1):
                out_img[x] += in_img[y]

in_grid = StencilGrid([10,10])
in_grid.data = numpy.ones([10,10])
out_grid = StencilGrid([10,10])
Kernel().kernel(in_grid, out_grid)
print out_grid
