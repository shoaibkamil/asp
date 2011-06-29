# common utilities for all asp.* to use
from __future__ import print_function
import os

def debug_print(*args):
    if 'ASP_DEBUG' in os.environ:
        for arg in args:
            print(arg, end='')
        print()
