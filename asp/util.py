# common utilities for all asp.* to use

import os

def debug_print(string):
    if 'ASP_DEBUG' in os.environ:
        print string
