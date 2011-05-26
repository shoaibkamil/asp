#!/bin/bash

if [ -z "${PYTHON}" ]
then
    PYTHON=python
    #PYTHON=pychecker
fi
if [ -z "${PYTHONARGS}" ]
then
    PYTHONARGS=
    #PYTHONARGS=-#1000
fi

if [ -n "${CUDA_TESTS+x}" ]
then
	PYTHONPATH=`pwd`:${PYTHONPATH} python tests/cuda_test.py
fi

PYTHONPATH=`pwd` python tests/stencil_grid_test.py
PYTHONPATH=`pwd` python tests/asp_module_tests.py
PYTHONPATH=`pwd` python tests/stencil_kernel_test.py
PYTHONPATH=`pwd` python tests/arraydoubler_test.py
PYTHONPATH=`pwd` python tests/cpp_ast_test.py
PYTHONPATH=`pwd` python tests/ast_tools_test.py
