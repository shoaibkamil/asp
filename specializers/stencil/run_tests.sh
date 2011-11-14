#!/bin/bash
PYTHONPATH=../../:tests:$PYTHONPATH

if [ -z "${PYTHON}" ]
then
    PYTHON=python
fi
if [ -z "${PYTHONARGS}" ]
then
    PYTHONARGS=
fi

rm -fr /tmp/asp_cache
PYTHONPATH=`pwd`:${PYTHONPATH} ${PYTHON} ${PYTHONARGS} -m unittest \
assert_utils_test \
stencil_cache_block_test \
stencil_convert_test \
stencil_grid_test \
stencil_kernel_test \
stencil_model_interpreter_test \
stencil_model_test \
stencil_python_front_end_test \
stencil_unroll_neighbor_iter_test
