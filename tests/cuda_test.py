# adapted from CodePy's nvcc example.
# requires PyCuda, CodePy, ASP, and CUDA 3.0+

from codepy.cgen import *
from codepy.bpl import BoostPythonModule
from codepy.cuda import CudaModule
from codepy.cgen.cuda import CudaGlobal
import asp.jit.asp_module as asp_module
import unittest2 as unittest

class CUDATest(unittest.TestCase):
    def test_cuda(self):
        mod = asp_module.ASPModule(use_cuda=True)

        # create the host code
        mod.add_to_preamble("""
        #define N 10
        void add_launch(int*,int*,int*);
        """,backend="c++") 
        mod.add_helper_function("foo_1", """int foo_1(){
        int a[N], b[N], c[N];
        int *dev_a, *dev_b, *dev_c;
        cudaMalloc( (void**)&dev_a, N * sizeof(int) );
        cudaMalloc( (void**)&dev_b, N * sizeof(int) );
        cudaMalloc( (void**)&dev_c, N * sizeof(int) );
        for (int i=0; i<N; i++) {
            a[i] = -i;
            b[i] = i * i;
        }
        cudaMemcpy( dev_a, a, N * sizeof(int),
                              cudaMemcpyHostToDevice );
        cudaMemcpy( dev_b, b, N * sizeof(int),
                              cudaMemcpyHostToDevice );
        cudaMemcpy( c, dev_c, N * sizeof(int),
                              cudaMemcpyDeviceToHost );
        add_launch(dev_a, dev_b, dev_c);
        cudaFree( dev_a );
        cudaFree( dev_b );
        cudaFree( dev_c );
        return 0;}""",backend="cuda")
        # create device code
        mod.add_to_module("""
        #define N 10
        __global__ void add( int *a, int *b, int *c ) {
            int tid = blockIdx.x;    // handle the data at this index
            if (tid < N)
                c[tid] = a[tid] + b[tid];
        }
        void add_launch(int *a, int *b, int *c) {
            add<<<N,1>>>( a, b, c );
        }
        """, backend='cuda')
        # test a call
        ret = mod.foo_1() 
        self.assertTrue(ret == 0)
        
if __name__ == '__main__':
    unittest.main()
