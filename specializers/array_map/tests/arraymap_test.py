import unittest

from array_map import *

class ArrayMapExample(ArrayMap):
    def operation(self, x):
        return 2*x+5

class BasicTests(unittest.TestCase):
    def test_pure_python(self):
        example = ArrayMapExample()
        arr = [1.0, 2.0, 3.0, 4.0]
        example.map(arr)
        self.assertEquals(arr[0], 7.0)

    def test_generated(self):
        example = ArrayMapExample()
        arr = [1.0, 2.0, 3.0, 4.0]
        example.map_using_trees(arr)
        self.assertEquals(arr[0], 7.0)


if __name__ == '__main__':
    unittest.main()
