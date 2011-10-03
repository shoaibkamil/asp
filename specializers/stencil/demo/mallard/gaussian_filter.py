from stencil_kernel import *
import sys
import numpy
import math

width = int(sys.argv[2])
height = int(sys.argv[3])
image_in = open(sys.argv[1], 'rb')
stdev_d = 3
radius = stdev_d * 3

class Kernel(StencilKernel):
    def run(self, in_grid, out_grid):
       self.kernel(in_grid, self.gaussian(stdev_d, radius*2), self.gaussian(70, 256), out_grid)

    def gaussian(self, stdev, length):
        result = StencilGrid([length])
        scale = 1.0/(stdev*math.sqrt(2.0*math.pi))
        divisor = -1.0 / (2.0 * stdev * stdev)
        for x in xrange(length):
           result[x] = scale * math.exp(float(x) * float(x) * divisor)
        return result

    def kernel(self, in_img, filter_d, filter_s, out_img):
        for x in out_img.interior_points():
            for y in in_img.neighbors(x, 1):
                out_img[x] += in_img[y] * filter_d[int(distance(x, y))] * filter_s[abs(in_img[x] - in_img[y])]

pixels = map(ord, list(image_in.read(width * height))) # Read in grayscale values
intensity = float(sum(pixels))/len(pixels)

kernel = Kernel()
kernel.should_unroll = False
out_grid = StencilGrid([width,height])
out_grid.ghost_depth = radius
in_grid = StencilGrid([width,height])
in_grid.ghost_depth = radius
for x in range(-radius,radius+1):
    for y in range(-radius,radius+1):
        in_grid.neighbor_definition[1].append( (x,y) )

for x in range(0,width):
    for y in range(0,height):
        in_grid.data[(x, y)] = pixels[y * width + x]

kernel.run(in_grid, out_grid)

for x in range(0,width):
    for y in range(0,height):
        pixels[y * width + x] = out_grid.data[(x, y)]
out_intensity = float(sum(pixels))/len(pixels)
for i in range(0, len(pixels)):
    pixels[i] = min(255, max(0, int(pixels[i] * (intensity/out_intensity))))

image_out = open(sys.argv[4], 'wb')
image_out.write(''.join(map(chr, pixels)))
