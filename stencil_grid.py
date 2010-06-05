import numpy

class StencilGrid(object):

    def __init__(self, size):
        self.dim = len(size)
        self.data = numpy.zeros(size)
        self.shape = size
        self.ghost_depth = 1
        self.interior = [x-2*self.ghost_depth for x in self.shape]

    def interior_points(self):
        # just do 2 and 3 dimension cases for now
        if self.dim == 2:
            for x in range(self.ghost_depth,self.shape[0]-self.ghost_depth):
                for y in range(self.ghost_depth,self.shape[1]-self.ghost_depth):
                    yield [x,y]
