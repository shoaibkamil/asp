import unittest

from blb import BLB

class ScalaTests(unittest.TestCase):
    mean = "return arr.reduce(_+_)/arr.length"
    sd = """val mean = arr.reduce(_+_)/arr.length
            return math.sqrt(arr.map(x => (x-mean)*(x-mean)).reduce(_+_))/(arr.length - 1)
    """

    def test_MeanMean(self):
        arr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        blb = BLB()
        blb.compute_estimates = ScalaTests.mean
        blb.reduce_bootstraps = ScalaTests.mean
        blb.average = ScalaTests.mean

        result = blb.run(arr)
        self.assertTrue(abs(result[0] - 3.5) < 1.3)

    def test_SDMean(self):
        arr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        blb = BLB()
        blb.compute_estimates = ScalaTests.mean
        blb.reduce_bootstraps = ScalaTests.sd
        blb.average = ScalaTests.mean

        result = blb.run(arr)
        self.assertTrue(result[0] < 0.3)

    def test_MeanSD(self):
        arr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        blb = BLB()
        blb.compute_estimates = ScalaTests.sd
        blb.reduce_bootstraps = ScalaTests.mean
        blb.average = ScalaTests.mean

        result = blb.run(arr)
        self.assertTrue(result[0] < 1.3)

    def test_SDSD(self):
        arr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        blb = BLB()
        blb.compute_estimates = ScalaTests.sd
        blb.reduce_bootstraps = ScalaTests.sd
        blb.average = ScalaTests.mean

        result = blb.run(arr)
        self.assertTrue(result[0] < 0.3)

if __name__ == '__main__':
    unittest.main()
