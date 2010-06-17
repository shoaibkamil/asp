import numpy
import inspect
from stencil_grid import *

# may want to make this inherit from something else...
class StencilKernel(object):
    def __init__(self):
        # we want to raise an exception if there is no kernel()
        # method defined.
        try:
            dir(self).index("kernel")
        except ValueError:
            raise Exception("No kernel method defined.")

        
            





