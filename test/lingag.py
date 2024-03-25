import unittest
import numpy as np
from libs.model_helpers import activators
from libs.model_helpers import costs
from libs.model_helpers import linalg
import scipy


def d_convolve_num(m, kern, Cost=lambda m, kern: scipy.signal.correlate(m, kern, mode='valid')):
    h = 1 / 10 ** 7

    cost1 = Cost(m, kern)

    dConvolve = np.zeros(kern.shape)

    for idx, weight in np.ndenumerate(kern):
        dKern = kern.copy()
        dKern[idx] += h

        cost2 = Cost(m, dKern)

        dw = (cost2 - cost1) / h
        dConvolve[idx] = dw.sum()

    return dConvolve

def convolve(m, kernel):
    return scipy.signal.correlate(m, kernel, 'valid')

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

        h = 1 / 999999999999
        preva = np.array([1 - h, 1, 1])
        da = activators.relu(np.dot(w, preva) + b)

        f = costs.abs_squared(a, y)
        df = costs.abs_squared(da, y)

        lim_df_da = (f - df) / h

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

        h = 1 / 999999999999
        preva = np.array([1, 1, 1, 10 - h])
        da = activators.relu(np.dot(w, preva) + b)

        f = costs.abs_squared(a, y)
        df = costs.abs_squared(da, y)

        lim_df_da = np.sum(f - df) / h

        self.assertAlmostEqual(lim_df_da, expected[3], places=0)

    def test_dconvolve(self):
        images = [
            np.arange(25).reshape((5, 5)),
            np.arange(49).reshape((7, 7)),
            np.arange(81).reshape((9, 9)),
            np.arange(784).reshape((28, 28)),
        ]

        funcs = [
            lambda m, kern: scipy.signal.correlate(m, kern, mode='valid')
        ]

        for image in images:
            N = image.shape[0]
            for func in funcs:
                kernel1 = np.ones((3, 3))
                imagePadded = np.pad(image, [(1, 1), (1, 1)])
                numericDConv = d_convolve_num(imagePadded, kernel1, func)
                formDCov = convolve(imagePadded, np.ones(image.shape))

                self.assertAlmostEqual(numericDConv.sum(), formDCov.sum(), places=0)

                kernel2 = np.ones((5, 5))
                imagePadded = np.pad(image, [(2, 2), (2, 2)])
                numericDConv = d_convolve_num(imagePadded, kernel2, func)
                formDCov = convolve(imagePadded, np.ones(image.shape))

                self.assertAlmostEqual(numericDConv.sum(), formDCov.sum(), places=0)

    def test_dcost_preva_4to4_2(self):
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

        h = 1 / 999999999999
        preva = np.array([1, 1, 1, 10 - h])
        da = activators.relu(np.dot(w, preva) + b)

        f = costs.abs_squared(a, y)
        df = costs.abs_squared(da, y)

        lim_df_da = np.sum(f - df) / h

        self.assertAlmostEqual(lim_df_da, expected[3], places=0)


if __name__ == '__main__':
    unittest.main()
