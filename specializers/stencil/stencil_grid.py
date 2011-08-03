"""A two-dimension grid of numeric values, used for input and output to a stencil kernel.
"""

import numpy
import math

class StencilGrid(object):

    def __init__(self, size):
        self.dim = len(size)
        self.data = numpy.zeros(size)
        self.shape = size
        self.ghost_depth = 1

        self.set_grid_variables()
        self.set_interior()
        # add default neighbor definition
        self.set_default_neighbor_definition()

    # want this to be indexable
    def __getitem__(self, x):
        return self.data[x]

    def __setitem__(self, x, y):
        self.data[x] = y

    def set_grid_variables(self):
        self.grid_variables = ["DIM"+str(x) for x in range(0,self.dim)]

    def set_interior(self):
        """
        Sets the number of interior points in each dimension
        """
        self.interior = [x-2*self.ghost_depth for x in self.shape]

    def set_default_neighbor_definition(self):
        """
        Sets the default for neighbors[0] and neighbors[1].  Note that neighbors[1]
        does not include the center point.
        """
        self.neighbor_definition = []

        self.neighbor_definition.append([tuple([0 for x in range(self.dim)])])
        self.neighbor_definition.append([])

        for x in range(self.dim):
            for y in [0, 1, -1]:
                tmp = list(self.neighbor_definition[0][0])
                tmp[x] += y
                tmp = tuple(tmp)
                if tmp != self.neighbor_definition[0][0]:
                    self.neighbor_definition[1].append(tmp)



    def interior_points(self):
        """
        Iterator over the interior points of the grid.  Only executed
        in pure Python mode; in SEJITS mode, it should be executed only
        in the translated language/library.
        """
        import itertools
        all_dims = [range(self.ghost_depth,self.shape[x]-self.ghost_depth) for x in range(0,self.dim)]
        for item in itertools.product(*all_dims):
            yield tuple(item)

    def border_points(self):
        """
        Iterator over the border points of a grid.  Only executed in pure Python
        mode; in SEJITS mode, it should be executed only in the translated
        language/library.
        """
        # TODO
        return []


    def neighbors(self, center, neighbors_id):
        """
        Returns the list of neighbors with the given neighbors_id. By
        default, IDs 0 and 1 give the list consisting of all
        points at a distance of 0 and 1 from the center point,
        respectively. Uses neighbor_definition to determine what the
        neighbors are.
        """
        # return tuples for each neighbor
        for neighbor in self.neighbor_definition[neighbors_id]:
            yield tuple(map(lambda a,b: a+b, list(center), list(neighbor)))

    def __repr__(self):
        return self.data.__repr__()

def distance(x,y):
    return math.sqrt(sum([(x[i]-y[i])**2 for i in range(0,len(x))]))
