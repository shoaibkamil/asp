#!/bin/bash

if [ -n "${CUDA_TESTS+x}" ]
then
	PYTHONPATH=`pwd` python tests/cuda_test.py
fi

PYTHONPATH=`pwd` python tests/stencil_grid_test.py
PYTHONPATH=`pwd` python tests/asp_module_tests.py
PYTHONPATH=`pwd` python tests/stencil_kernel_test.py
PYTHONPATH=`pwd` python tests/arraydoubler_test.py