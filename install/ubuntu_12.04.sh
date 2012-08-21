#!/bin/bash
sudo apt-get install python-pip python-dev libboost-python-dev g++
sudo pip install asp
echo "BOOST_PYTHON_LIBNAME=['boost_python-mt-py27']" >> ~/.aksetup-defaults.py
