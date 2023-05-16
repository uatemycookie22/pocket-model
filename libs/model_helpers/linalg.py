import numpy as np


def dcost_dpreva(dc_db: np.ndarray, w: np.ndarray):
    return np.asarray([np.dot(np.array(w.T[k, :]), dc_db) for k in range(w.shape[1])])


def dcost_dw(dc_da_dz: np.ndarray, preva: np.ndarray):
    return np.outer(dc_da_dz, preva)


def dcost_db(dc_da: np.ndarray, da_dz: np.ndarray):
    return dc_da * da_dz

