import numpy

class StencilGrid(object):

    def __init__(self, size):
        self.dim = len(size)
        self.data = numpy.zeros(size)
        self.shape = size
        self.ghost_depth = 1
        self.interior = [x-2*self.ghost_depth for x in self.shape]
        
        self.grid_variables = ["DIM"+str(x) for x in range(0,self.dim)]

        # add default neighbor definition
        self.neighbor_definition = []
        self.neighbor_definition.append( [x for x in self.grid_variables] )
        self.neighbor_definition.append([])

        import copy
        for x in range(self.dim):
            for y in ["", "+1", "-1"]:
                tmp = copy.deepcopy(self.neighbor_definition[0])
                tmp[x] += y
                if tmp != self.grid_variables or len(self.neighbor_definition[1]) < 1:
                    self.neighbor_definition[1].append(tmp)




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


    def neighbors(self, center, dist):
        """
        Returns a list of neighbors that are at distance dist from the center
        point.  Uses neighbor_definition to determine what the neighbors are.
        """
        import copy
        # create a dictionary mapping the grid variables to the actual center
        center_point = dict(zip(self.grid_variables, center))
        for neighbor in self.neighbor_definition[dist]:
            yield [eval(x,center_point) for x in copy.deepcopy(neighbor)]

            
            
