import numpy as np
import scipy
from numpy import ndarray


def dcost_preva_deprecated(dc_db: np.ndarray, w: np.ndarray):
    return np.asarray([np.dot(np.array(w.T[k, :]), dc_db) for k in range(w.shape[1])])


def dcost_dpreva(dc_db: np.ndarray, w: np.ndarray):
    return np.einsum('i, ij -> j', dc_db, w)


def dcost_dw(dc_da_dz: np.ndarray, preva: np.ndarray):
    return np.outer(dc_da_dz, preva)


def dcost_db(dc_da: np.ndarray, da_dz: np.ndarray):
    return dc_da * da_dz


def avg_gradient(gradients: list[list[np.ndarray]]):
    avg_gradients = []
    gradients_size = len(gradients)
    gradient_size = len(gradients[0])

    for L in range(gradient_size):
        avg_gradient = gradients[0][L]

        if avg_gradient is None:
            continue

        for gradient in gradients[min(1, gradient_size):]:
            avg_gradient += gradient[L]

        avg_gradient = avg_gradient / gradients_size

        avg_gradients.append(avg_gradient)

    return avg_gradients


def convolve(m: np.ndarray, kernel: np.ndarray):
    return scipy.signal.correlate(m, kernel, 'valid')


def full_convolve(m: np.ndarray, kernel: np.ndarray):
    return scipy.signal.correlate(m, kernel, 'full')


def dconvolve(m: np.ndarray, dc_da: np.ndarray):
    return scipy.signal.correlate(m, dc_da, 'valid')


def matvec(m: np.ndarray, v: np.ndarray):
    return m.dot(v)


def pad_to_axis(x: np.ndarray, P: int, axis=0) -> np.ndarray:
    pad_width = ((P, P),) * (axis + 1)
    D = len(x.shape)
    empty_pad = (((0, 0),) * (D - axis))[:D]
    pad_width = pad_width + empty_pad
    return np.pad(x, pad_width[:D])


def assert_shape(x: np.ndarray) -> np.ndarray:
    if len(x.shape) <= 2:
        return np.expand_dims(x, axis=2)
    return x


def reduce_shape(x: np.ndarray) -> np.ndarray:
    if len(x.shape) > 2:
        return np.squeeze(x, axis=2)
    return x


# Matches x to y and returns x
def match_shape(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x.shape) > len(y.shape):
        return reduce_shape(x)
    if len(x.shape) < len(y.shape):
        return assert_shape(x)
    return x


def shuffle(x: np.ndarray, y: np.ndarray) -> tuple[ndarray, ndarray]:
    data_train = list(zip(x, y))
    np.random.shuffle(data_train)
    train_x, train_y = zip(*data_train)
    train_x, train_y = np.array(train_x), np.array(train_y)

    return train_x, train_y


def abs_mean(x: np.ndarray):
    gt_zero: np.ndarray = abs(x[abs(x) > 10 ** -3])
    if (len(gt_zero) > 0):
        return gt_zero.mean()
    else:
        return np.array([0])


def maxpool(x: np.ndarray, filter_size: int, stride: int):
    return poolingOverlap(x, filter_size, stride, pad=False, return_max_pos=True)


def dmaxpool(x: np.ndarray, p: np.ndarray, input_shape: tuple[int], stride: int):
    return unpooling(x, p, input_shape, stride)


def poolingOverlap(mat, f, stride=None, method='max', pad=False,
                   return_max_pos=False):
    '''Overlapping pooling on 2D or 3D data.

    Args:
        mat (ndarray): input array to do pooling on the first 2 dimensions.
        f (int): pooling kernel size.
    Keyword Args:
        stride (int or None): stride in row/column. If None, same as <f>,
            i.e. non-overlapping pooling.
        method (str): 'max for max-pooling,
                      'mean' for average-pooling.
        pad (bool): pad <mat> or not. If true, pad <mat> at the end in
               y-axis with (f-n%f) number of nans, if not evenly divisible,
               similar for the x-axis.
        return_max_pos (bool): whether to return an array recording the locations
            of the maxima if <method>=='max'. This could be used to back-propagate
            the errors in a network.
    Returns:
        result (ndarray): pooled array.

    See also unpooling().
    '''
    m, n = mat.shape[:2]
    if stride is None:
        stride = f
    _ceil = lambda x, y: x // y + 1

    if pad:
        ny = _ceil(m, stride)
        nx = _ceil(n, stride)
        size = ((ny - 1) * stride + f, (nx - 1) * stride + f) + mat.shape[2:]
        mat_pad = np.full(size, 0)
        mat_pad[:m, :n, ...] = mat
    else:
        mat_pad = mat[:(m - f) // stride * stride + f, :(n - f) // stride * stride + f, ...]

    view = asStride(mat_pad, (f, f), stride)
    if method == 'max':
        result = np.nanmax(view, axis=(2, 3), keepdims=return_max_pos)
    else:
        result = np.nanmean(view, axis=(2, 3), keepdims=return_max_pos)

    if return_max_pos:
        pos = np.where(result == view, 1, 0)
        result = np.squeeze(result)
        return result, pos
    else:
        return result


def asStride(arr, sub_shape, stride):
    '''Get a strided sub-matrices view of an ndarray.
    Args:
        arr (ndarray): input array of rank 2 or 3, with shape (m1, n1) or (m1, n1, c).
        sub_shape (tuple): window size: (m2, n2).
        stride (int): stride of windows in both y- and x- dimensions.
    Returns:
        subs (view): strided window view.
    See also skimage.util.shape.view_as_windows()
    '''
    s0, s1 = arr.strides[:2]
    m1, n1 = arr.shape[:2]
    m2, n2 = sub_shape[:2]
    view_shape = (1 + (m1 - m2) // stride, 1 + (n1 - n2) // stride, m2, n2) + arr.shape[2:]
    strides = (stride * s0, stride * s1, s0, s1) + arr.strides[2:]
    subs = np.lib.stride_tricks.as_strided(
        arr, view_shape, strides=strides, writeable=False)
    return subs


def unpooling(mat, pos, ori_shape, stride):
    '''Undo a max-pooling of a 2d or 3d array to a larger size
    Args:
        mat (ndarray): 2d or 3d array to unpool on the first 2 dimensions.
        pos (ndarray): array recording the locations of maxima in the original
            array. If <mat> is 2d, <pos> is 4d with shape (iy, ix, cy, cx).
            Where iy/ix are the numbers of rows/columns in <mat>,
            cy/cx are the sizes of the each pooling window.
            If <mat> is 3d, <pos> is 5d with shape (iy, ix, cy, cx, cc).
            Where cc is the number of channels in <mat>.
        ori_shape (tuple): original shape to unpool to.
        stride (int): stride used during the pooling process.
    Returns:
        result (ndarray): <mat> unpoolled to shape <ori_shape>.
    '''
    assert np.ndim(pos) in [4, 5], '<pos> should be rank 4 or 5.'
    result = np.zeros(ori_shape)
    if np.ndim(pos) == 5:
        iy, ix, cy, cx, cc = np.where(pos == 1)
        iy2 = iy * stride
        ix2 = ix * stride
        iy2 = iy2 + cy
        ix2 = ix2 + cx
        values = mat[iy, ix, cc].flatten()
        result[iy2, ix2, cc] = values
    else:
        iy, ix, cy, cx = np.where(pos == 1)
        iy2 = iy * stride
        ix2 = ix * stride
        iy2 = iy2 + cy
        ix2 = ix2 + cx
        values = mat[iy, ix].flatten()
        result[iy2, ix2] = values
    return result
