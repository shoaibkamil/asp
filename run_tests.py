#!/usr/bin/python
import sys, os, fnmatch

sys.path.append(os.getcwd())

for testfile in os.listdir('tests'):
	if fnmatch.fnmatch(testfile, '*test.py'):
		full_test_path = os.getcwd() + '/tests/' + testfile
		os.spawnle(os.P_WAIT, '/usr/bin/python', 'python', full_test_path, {'PYTHONPATH':os.getcwd()})



