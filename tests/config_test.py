import unittest

from asp.config import *

class ConfigReaderTest(unittest.TestCase):
    def test_getOption(self):
        self.assertNotEqual(ConfigReader().get_option('gmm'), None)
        self.assertEqual(ConfigReader().get_option('qwerty'), None)


if __name__ == '__main__':
    unittest.main()
