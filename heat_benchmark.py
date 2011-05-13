import asp
from stencil_grid import *
from stencil_kernel import *
import numpy
import time

class Heat3D(StencilKernel):
	def kernel(self, in_grid, out_grid):
		for x in in_grid.interior_points():
			for y in in_grid.neighbors(x, 1):
				out_grid[x] = out_grid[x] + (1.0/6.0)*in_grid[y]


in_grid = StencilGrid([258,258,258])
out_grid = StencilGrid([258,258,258])
cache_clear_grid = numpy.arange(258*258*258)

def clear_cache(arr):
    for x in xrange(258*258*258):
        cache_clear_grid[x] -= 1

k = Heat3D()
for x in xrange(10):
	clear_cache(cache_clear_grid)
	k.kernel(in_grid, out_grid)

#in_grid = StencilGrid([258,258,258])
#out_grid = StencilGrid([258,258,258])

#start_time = time.time()
#for x in range(1,257):
#	for y in range(1,257):
#		for z in range(1,257):
#			out_grid[(x,y,z)] = out_grid[(x,y,z)] + (1.0/6.0)*(in_grid[(x+1,y,z)] +
#				in_grid[(x-1,y,z)] +
#				in_grid[(x,y+1,z)] +
#				in_grid[(x,y-1,z)] +
#				in_grid[(x,y,z+1)] +
#				in_grid[(x,y,z-1)] )
			
#elapsed_time = time.time() - start_time

print(k.mod.compiled_methods["kernel"].database.variant_times)
#print(elapsed_time)

