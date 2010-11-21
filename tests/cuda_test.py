# adapted from CodePy's nvcc example.
# requires PyCuda, CodePy, ASP, and CUDA 3.0+

from codepy.cgen import *
from codepy.bpl import BoostPythonModule
from codepy.cuda import CudaModule
from codepy.cgen.cuda import CudaGlobal
import asp.jit.asp_module as asp_module
import unittest

class CUDATest(unittest.TestCase):
    def test_cuda(self):

        # The host module should include a function which is callable from Python
        #host_mod = BoostPythonModule()
        
        mod = asp_module.ASPModule(use_cuda=True)

        # This host function extracts a pointer and shape information from a PyCUDA
        # GPUArray, and then sends them to a CUDA function.  The CUDA function
        # returns a pointer to an array of the same type and shape as the input array.
        # The host function then constructs a GPUArray with the result.

        statements = [
            # Extract information from incoming GPUArray
            'PyObject* shape = PyObject_GetAttrString(gpuArray, "shape")',
            'PyObject* type = PyObject_GetAttrString(gpuArray, "dtype")',
            'PyObject* pointer = PyObject_GetAttrString(gpuArray, "gpudata")',
            'CUdeviceptr cudaPointer = boost::python::extract<CUdeviceptr>(pointer)',
            'PyObject* length = PySequence_GetItem(shape, 0)',
            'int intLength = boost::python::extract<int>(length)',
            # Call CUDA function
            'CUdeviceptr diffResult = diffInstance(cudaPointer, intLength)',
            # Build resulting GPUArray
            'PyObject* args = Py_BuildValue("()")',
            'PyObject* newShape = Py_BuildValue("(i)", intLength)',
            'PyObject* kwargs = Py_BuildValue("{sOsOsi}", "shape", newShape, "dtype", type, "gpudata", diffResult)',
            'PyObject* GPUArrayClass = PyObject_GetAttrString(gpuArray, "__class__")',
            'PyObject* remoteResult = PyObject_Call(GPUArrayClass, args, kwargs)',
            'return remoteResult']
        mod.add_function(
            FunctionBody(
                FunctionDeclaration(Pointer(Value("PyObject", "adjacentDifference")),
                                    [Pointer(Value("PyObject", "gpuArray"))]),
                Block([Statement(x) for x in statements])))
        mod.add_to_preamble([Include('boost/python/extract.hpp')])
        
        
        globalIndex = 'int index = blockIdx.x * blockDim.x + threadIdx.x'
        compute_diff = 'outputPtr[index] = inputPtr[index] - inputPtr[index-1]'
        launch = ['CUdeviceptr output',
                  'cuMemAlloc(&output, sizeof(T) * length)',
                  'int bSize = 256',
                  'int gSize = (length-1)/bSize + 1',
                  'diffKernel<<<gSize, bSize>>>((T*)inputPtr, length, (T*)output)',
                  'return output']
        diff =[
            Template('typename T',
                     CudaGlobal(FunctionDeclaration(Value('void', 'diffKernel'),
                                                    [Value('T*', 'inputPtr'),
                                                     Value('int', 'length'),
                                                     Value('T*', 'outputPtr')]))),
            Block([Statement(globalIndex),
                   If('index == 0',
                      Statement('outputPtr[0] = inputPtr[0]'),
                      If('index < length',
                         Statement(compute_diff),
                         Statement('')))]),

            Template('typename T',
                     FunctionDeclaration(Value('CUdeviceptr', 'difference'),
                                         [Value('CUdeviceptr', 'inputPtr'),
                                          Value('int', 'length')])),
            Block([Statement(x) for x in launch])]
        
        # right now the global stuff is not handled as a function; they're templated.
        # so you have to add them using mod.add_to_cuda_module().  the actual function
        # that is called by the C++ (i.e. diffInstance()) is added below, using
        # mod.add_function()

        mod.add_to_cuda_module(diff)

        diff_instance = FunctionBody(
            FunctionDeclaration(Value('CUdeviceptr', 'diffInstance'),
                                [Value('CUdeviceptr', 'inputPtr'),
                                 Value('int', 'length')]),
            Block([Statement('return difference<int>(inputPtr, length)')]))

        
        mod.add_function(diff_instance, cuda_func=True)

        
        

        import pycuda.autoinit
        import pycuda.driver
        
        
        import pycuda.gpuarray
        import numpy as np
        length = 25
        constantValue = 2
        # This is a strange way to create a GPUArray, but is meant to illustrate
        # how to construct a GPUArray if the GPU buffer it owns has been
        # created by something else
        
        pointer = pycuda.driver.mem_alloc(length * 4)
        pycuda.driver.memset_d32(pointer, constantValue, length)
        a = pycuda.gpuarray.GPUArray((25,), np.int32, gpudata=pointer)
        b = mod.adjacentDifference(a).get()
        
        golden = [constantValue] + [0] * (length - 1)
        difference = [(x-y)*(x-y) for x, y in zip(b, golden)]
        error = sum(difference)
        self.assertEquals(error, 0)


if __name__ == '__main__':
    unittest.main()
