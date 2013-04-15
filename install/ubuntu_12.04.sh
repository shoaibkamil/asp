#!/bin/bash
sudo apt-get install python-pip python-dev libboost-python-dev g++
sudo easy_install -U distribute
sudo pip --default-timeout=1000 install --use-mirrors asp
echo "BOOST_PYTHON_LIBNAME=['boost_python-mt-py27']" >> ~/.aksetup-defaults.py
