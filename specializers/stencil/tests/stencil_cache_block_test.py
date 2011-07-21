import unittest2 as unittest
from asp.codegen.cpp_ast import *
from stencil_cache_block import *

class CacheBlockerTests(unittest.TestCase):
    def test_2d(self):
        loop = For("i",
                       CNumber(0),
                       CNumber(7),
                       CNumber(1),
                       Block(contents=[For("j",
                                       CNumber(0),
                                       CNumber(3),
                                       CNumber(1),
                                       Block(contents=[Assign(CName("v"), CName("i"))]))]))
        
        
        wanted = """for (int ii = 0; (ii <= 7); ii = (ii + (1 * 2)))
        {
         for (int jj = 0; (jj <= 3); jj = (jj + (1 * 2)))
         {
          for (int i = ii; (i <= min((ii + 2),7)); i = (i + 1))
          {
           for (int j = jj; (j <= min((jj + 2),3)); j = (j + 1))
           {
            v = i;
           }
          }
         }
        }"""
        
        
        self.assertEqual(str(StencilCacheBlocker().block(loop, (2, 2))).replace(' ',''), wanted.replace(' ',''))

    def test_3d(self):
        loop = For("i",
                   CNumber(0),
                   CNumber(7),
                   CNumber(1),
                   Block(contents=[For("j",
                                       CNumber(0),
                                       CNumber(3),
                                       CNumber(1),
                                       Block(contents=[For("k",
                                                           CNumber(0),
                                                           CNumber(4),
                                                           CNumber(1),
                                                           Block(contents=[Assign(CName("v"), CName("i"))]))]))]))

        print StencilCacheBlocker().block(loop, (2,2,3))
        assert(False)
        
if __name__ == '__main__':
    unittest.main()


