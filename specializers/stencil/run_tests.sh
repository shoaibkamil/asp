#!/bin/bash
PYTHONPATH=../../:$PYTHONPATH

if [ -z "${PYTHON}" ]
then
    PYTHON=python
fi
if [ -z "${PYTHONARGS}" ]
then
    PYTHONARGS=
fi

rm -fr /tmp/asp_cache
PYTHONPATH=`pwd`:${PYTHONPATH} unit2 discover tests '*.py'
