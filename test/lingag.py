import unittest
import numpy as np
from libs.model_helpers import activators
from libs.model_helpers import costs
from libs.model_helpers import linalg

class LinAlgTest(unittest.TestCase):
    def test_dcost_preva_3to1(self):
        preva = np.array([1, 1, 1])
        b = np.array([1])
        w = np.array([
            [10, 10, 10],
        ])

        z = np.dot(w, preva) + b
        a = activators.relu(z)
        y = np.array([0])

        dc_da = costs.dabs_squared(a, y)
        da_dz = activators.drelu(z)

        dc_db = linalg.dcost_db(dc_da, da_dz)
        dc_dpreva = linalg.dcost_dpreva(dc_db, w)

        expected = np.array([620, 620, 620])
        self.assertTrue(np.array_equal(expected, dc_dpreva), f"expected: {expected}\n actual: {dc_dpreva}")


        h = 1/999999999999
        preva = np.array([1 - h, 1, 1])
        da = activators.relu(np.dot(w, preva) + b)

        f = costs.abs_squared(a, y)
        df = costs.abs_squared(da, y)

        lim_df_da = (f - df)/h

        self.assertAlmostEqual(lim_df_da[0], expected[0], places=0)

    def test_dcost_preva_4to4(self):
        preva = np.array([1, 1, 1, 10])
        b = np.array([0, 0, 0, 0])
        w = np.array([
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 2],
        ])

        z = np.dot(w, preva) + b
        a = activators.relu(z)
        y = np.array([0, 0, 0, 0])

        dc_da = costs.dabs_squared(a, y)
        da_dz = activators.drelu(z)

        dc_db = linalg.dcost_db(dc_da, da_dz)
        dc_dpreva = linalg.dcost_dpreva(dc_db, w)

        expected = np.array([64, 64, 64, 92])
        self.assertTrue(np.array_equal(expected, dc_dpreva), f"expected: {expected}\n actual: {dc_dpreva}")


        h = 1/999999999999
        preva = np.array([1, 1, 1, 10 - h])
        da = activators.relu(np.dot(w, preva) + b)

        f = costs.abs_squared(a, y)
        df = costs.abs_squared(da, y)

        lim_df_da = np.sum(f - df)/h

        self.assertAlmostEqual(lim_df_da, expected[3], places=0)




if __name__ == '__main__':
    unittest.main()
