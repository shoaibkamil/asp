#!/bin/bash

if [ -z "${PYTHON}" ]
then
    export PYTHON=python
    #PYTHON=pychecker
fi
if [ -z "${PYTHONARGS}" ]
then
    export PYTHONARGS=-tt
    #PYTHONARGS=-#1000
fi

if [ -n "${CUDA_TESTS+x}" ]
then
	PYTHONPATH=`pwd`:${PYTHONPATH} python tests/cuda_test.py
fi

PYTHONPATH=`pwd`:${PYTHONPATH} ${PYTHON} ${PYTHONARGS} tests/asp_module_tests.py
PYTHONPATH=`pwd`:${PYTHONPATH} ${PYTHON} ${PYTHONARGS} tests/cpp_ast_test.py
PYTHONPATH=`pwd`:${PYTHONPATH} ${PYTHON} ${PYTHONARGS} tests/ast_tools_test.py

# Test setup by making specializer tests use a version of asp installed
# to a temporary directory.
#rm -fr test_install; mkdir -p test_install/lib/python2.7/site-packages/
#export PYTHONPATH=`cd test_install;pwd`/lib/python2.7/site-packages:${PYTHONPATH}
#python setup.py build install --prefix=test_install

cd specializers; ./run_tests.sh; cd ..
