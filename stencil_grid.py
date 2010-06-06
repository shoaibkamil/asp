import numpy

class StencilGrid(object):

    def __init__(self, size):
        self.dim = len(size)
        self.data = numpy.zeros(size)
        self.shape = size
        self.ghost_depth = 1
        self.interior = [x-2*self.ghost_depth for x in self.shape]
        
        self.grid_variables = ["DIM"+str(x) for x in range(0,self.dim)]


    def interior_points(self):
        """
        Iterator over the interior points of the grid.  Only executed
        in pure Python mode; in SEJITS mode, it should be executed only
        in the translated language/library.
        """
        # just do 2 and 3 dimension cases for now
        if self.dim == 2:
            for x in range(self.ghost_depth,self.shape[0]-self.ghost_depth):
                for y in range(self.ghost_depth,self.shape[1]-self.ghost_depth):
                    yield [x,y]
        elif self.dim == 3:
            for x in range(self.ghost_depth,self.shape[0]-self.ghost_depth):
                for y in range(self.ghost_depth,self.shape[1]-self.ghost_depth):
                    for z in range(self.ghost_depth,self.shape[2]-self.ghost_depth):
                        yield [x,y,z]

    def border_points(self):
        """
        Iterator over the border points of a grid.  Only executed in pure Python
        mode; in SEJITS mode, it should be executed only in the translated
        language/library.
        """
        pass


    def neighbors(center, dist):
        """
        Returns a list of neighbors that are at distance dist from the center
        point.  Uses neighbor_definition to determine what the neighbors are.
        """
        pass
