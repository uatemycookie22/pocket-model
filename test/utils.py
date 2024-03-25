import numpy as np

from libs.model.node_model import NodeModel


def numeric_grad(sut: NodeModel, x, label):
    h = 1 / 10 ** 6
    cost1 = np.square(sut.predict(x) - label)

    numericW = []
    for W in sut.nn.weights:
        dW = np.zeros(W.shape)
        for idx, weight in np.ndenumerate(W):
            tmp = W[idx]
            W[idx] += h

            cost2 = np.square(sut.predict(x) - label)

            dw = (cost2 - cost1) / h
            dW[idx] = dw.sum()

            W[idx] = tmp

        numericW.append(dW)

    numericB = []
    for B in sut.nn.biases:
        dB = np.zeros(B.shape)
        for idx, bias in np.ndenumerate(B):
            tmp = B[idx]
            B[idx] += h

            cost2 = np.square(sut.predict(x) - label)

            db = (cost2 - cost1) / h
            dB[idx] = db.sum()

            B[idx] = tmp

        numericB.append(dB)

    return numericW, numericB


def cmp_arr(x: np.ndarray, y: np.ndarray, allowed_error=0.05):
    cmp_shape = x.shape == y.shape
    print(x.shape, y.shape)

    x_sig = significant_abs(x)
    y_sig = significant_abs(y)

    print(f"Numeric:")
    print(x_sig)
    print(f"Auto:")
    print(y_sig)

    error: np.ndarray = abs((x_sig - y_sig) / x_sig)
    expected_error = abs((x_sig - x_sig * allowed_error) / x_sig)
    diff = expected_error - error
    diff[diff >= 0] = True
    diff[diff < 0] = False

    return {
        "cmp_shape": cmp_shape,
        "cmp_val": diff.all()
    }


def significant_abs(x: np.ndarray) -> np.ndarray:
    x[abs(x) < 10 ** -3] = 10**-3
    return x
