import utils.toolbox as toolbox
import unittest
import numpy as np



class Test_LCG(unittest.TestCase):
    def test_LCG(self):
        self.result = toolbox.generate_ts(36, 717, ["LCG", 18])
        self.assertEqual( self.result.shape, (717, 20 ) )
        # np.savetxt('{}.txt'.format(["LCG", 18]), self.result)

    def test_MT(self):
        self.result = toolbox.generate_ts(36, 717, ["MT", 18])
        self.assertEqual( self.result.shape, (717, 20 ) )

    def test_bigdeal(self):
        self.result = toolbox.generate_ts(36, 717, ["bigdeal", 1000000])
        self.assertEqual( self.result.shape, (717, 20 ) )


if __name__ == '__main__':
    pass