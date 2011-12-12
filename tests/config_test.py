import unittest

from asp.config import *

class ConfigReaderTest(unittest.TestCase):
    def test_get_option(self):
        config = ConfigReader()
        config.configs = yaml.load("""
                                   gmm:
                                     option1: True
                                     option2: something
                                   """)
        self.assertNotEqual(config.get_option('gmm'), None)
        self.assertEqual(config.get_option('qwerty'), None)


if __name__ == '__main__':
    unittest.main()
