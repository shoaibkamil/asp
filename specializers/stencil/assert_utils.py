"""Utilities for checking object types using assertions.
"""

from types import *

def assert_has_type(x, t, x_name='obj'):
    if type(t) is ListType:
        type_found = False
        for t2 in t:
            if isinstance(x, t2):
                type_found = True
        assert type_found, "%s is not one of the types %s: %s" % (x_name, t, `x`)
    else:
        assert isinstance(x, t), "%s is not %s: %s" % (x_name, t, `x`)

def assert_is_list_of(lst, t, lst_name='list'):
    assert_has_type(lst, ListType, lst_name)
    for x in lst:
        assert_has_type(x, t, "%s element" % lst_name)
