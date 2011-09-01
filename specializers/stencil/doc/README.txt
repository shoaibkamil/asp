A typical example of a stencil problem instance looks like this:

from stencil_kernel import *
import stencil_grid

class ExampleKernel(StencilKernel):
    def kernel(self, in_grid, out_grid):
        for x in out_grid.interior_points():
            for y in in_grid.neighbors(x, 1):
                out_grid[x] = out_grid[x] + in_grid[y]

in_grid = StencilGrid([5,5])
in_grid.neighbor_definition[1] = [(-1, 0), (0, -1), (1, 0), (0, 1)]
in_grid.data = numpy.ones([5,5])
out_grid = StencilGrid([5,5])
out_grid.ghost_depth = 1
ExampleKernel().kernel(in_grid, out_grid)

The application programmer defines a class which inherits from
StencilKernel in module stencil_kernel. They implement the required
function "kernel" on this class, which takes a "self" argument (not
used), a series of input grids, and an output grid (must be last).
Both input and output grids are represented using instances of the
StencilGrid class, which wraps a (possibly multidimensional) Numpy
array. The stencil is invoked by calling the kernel() function.

The field "ghost_depth" of out_grid determines which points are
considered the interior points. Any point within distance ghost_depth
of the boundary will be considered part of the border, and the rest
are interior points. The default is 1.

At the top level, a kernel is a sequence of loops over either the
interior_points() or the border_points() method of the output grid
(zero or more of each, in any order), e.g. this is okay:

def kernel(self, in_grid, out_grid):
    for x in out_grid.interior_points():
        out_grid[x] = 1
    for x in out_grid.border_points():
        out_grid[x] = 0
    for x in out_grid.interior_points():
        out_grid[x] = 2

Each of these loops contains a sequence of statements, where a
statement is either:

* An assignment to the current output grid element
* A loop over the neighbors of an input grid point, as in:

    for y in in_grid.neighbors(x, 1):

Here x indicates the point to take neighbors of (although x is an
index into the output grid, here it refers to the corresponding index
in in_grid). The number "1" refers to the neighbor definition
ID. Before calling kernel(), we specified:

in_grid.neighbor_definition[1] = [(-1, 0), (0, -1), (1, 0), (0, 1)]

Therefore this neighbor loop will loop over 4 points whose relative
positions are as given in this neighbor definition. Each input grid
has its own set of neighbor definitions and may assign different
neighbor definitions to different IDs. By default, each input grid
has two neighbor definitions defined as follows:

neighbor_definition[0] = [(0, 0)]
neighbor_definition[1] = [(-1, 0), (0, -1), (1, 0), (0, 1)]

The interior of neighbor loops may only be a sequence of assignments
to the current output grid element. Neighbor loops cannot currently be
nested.

The right-hand side of each assignment to the current output grid
element is an expression. Assume the assignment falls inside the
following loop:

for x in out_grid.interior_points():
    for y in in_grid.neighbors(x, 1):
	out_grid[x] = (expression goes here)

Following are the possible expressions in this context:

* A numeric constant, like 1 or 3.14.

* out_grid[x] : Reads the current output element.

* in_grid[x] : Reads the input element at the same position as the
  current output element.

* in_grid[y] : Reads the current neighbor in the input grid.

* out_grid[x] + (2 * in_grid[y]) : Arithmetic operators +, -, *, /,
  and % can be used to combine expressions.

* distance(x, y) : Gives the Euclidean distance between two positions
  (e.g. (0,0) and (1,1) are sqrt(2) apart)

* abs(in_grid[y] + int(3.14)) : A set of predefined math functions can
  be called on expressions, including abs() for absolute value and
  int() to convert a double to an integer (rounding towards zero).

* in_grid_2[in_grid[y]] : If an input grid is one-dimensional, you can
  index into it using an expression.

Here is the bilateral filter example, using many of these:

class Kernel(StencilKernel):
   def kernel(self, in_img, filter_d, filter_s, out_img):
       for x in out_img.interior_points():
           for y in in_img.neighbors(x, 1):
               out_img[x] += in_img[y] * filter_d[int(distance(x, y))] * filter_s[abs(int(in_img[x]-in_img[y]))]

If any construct not obeying these restrictions is used in the
kernel() function, the program will fail during construction of an
instance of the StencilKernel subclass.
