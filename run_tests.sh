#!/bin/bash

PYTHON=python
PYTHONARGS=
#PYTHON=pychecker
#PYTHONARGS=-#1000

if [ -n "${CUDA_TESTS+x}" ]
then
	PYTHONPATH=`pwd` python tests/cuda_test.py
fi

PYTHONPATH=`pwd` ${PYTHON} ${PYTHONARGS} tests/stencil_grid_test.py
PYTHONPATH=`pwd` ${PYTHON} ${PYTHONARGS} tests/asp_module_tests.py
PYTHONPATH=`pwd` ${PYTHON} ${PYTHONARGS} tests/stencil_kernel_test.py
PYTHONPATH=`pwd` ${PYTHON} ${PYTHONARGS} tests/arraydoubler_test.py
PYTHONPATH=`pwd` ${PYTHON} ${PYTHONARGS} tests/cpp_ast_test.py
PYTHONPATH=`pwd` ${PYTHON} ${PYTHONARGS} tests/ast_utils_test.py
