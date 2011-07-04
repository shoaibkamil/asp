import unittest2 as unittest
from types import *
from assert_utils import *

class BasicTests(unittest.TestCase):
    def test_assert_has_type_right_type(self):
        assert_has_type(2, IntType)
        assert_has_type(2, object)

    def test_assert_has_type_wrong_type(self):
        with self.assertRaises(AssertionError):
            assert_has_type(2, StringType)

    def test_assert_has_type_list_right_type(self):
        assert_has_type(2, [IntType, StringType])
        assert_has_type('hello', [IntType, StringType])
        assert_has_type([2], [IntType, StringType, object])

    def test_assert_has_type_list_wrong_type(self):
        with self.assertRaises(AssertionError):
            assert_has_type([2], [IntType, StringType])

    def test_assert_is_list_of_not_list(self):
        with self.assertRaises(AssertionError):
            assert_is_list_of(2, object)

    def test_assert_is_list_of_right_type(self):
        assert_is_list_of([2, 3, 4], IntType)

    def test_assert_is_list_of_wrong_type(self):
        with self.assertRaises(AssertionError):
            assert_has_type([2, 'string', 4], IntType)

    def test_assert_is_list_of_list_right_type(self):
        assert_is_list_of([2, 'string', 4], [IntType, StringType])
        assert_is_list_of([2, 'string', [2], 4], [IntType, StringType, object])

    def test_assert_is_list_of_list_wrong_type(self):
        with self.assertRaises(AssertionError):
            assert_has_type([2, 'string', [2], 4], [IntType, StringType])

if __name__ == '__main__':
    unittest.main()
