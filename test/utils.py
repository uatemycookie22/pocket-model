import numpy as np

from libs.model.node_model import NodeModel


def numeric_grad(sut: NodeModel, x):
    h = 1 / 10 ** 6
    cost1 = np.square(sut.predict(x))

    numericW = []
    for W in sut.nn.weights:
        dW = np.zeros(W.shape)
        for idx, weight in np.ndenumerate(W):
            tmp = W[idx]
            W[idx] += h

            cost2 = np.square(sut.predict(x))

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

            cost2 = np.square(sut.predict(x))

            db = (cost2 - cost1) / h
            dB[idx] = db.sum()

            B[idx] = tmp

        numericB.append(dB)

    return numericW, numericB
