import numpy as np

np.random.seed()

# k = Rows = number of activation neurons in L + 1
# n = Columns = number of activation neurons in L

# A weights matrix is a ndarray of shape (k, n)

# For example, here is a nd array of shape (1, 1)
one_one_w = np.array([
    [5]
])

# (2, 1)
two_one_w = np.array([
    [5],
    [7]
])

# (4, 16)
four_sixteen_wr = np.random.random((4, 16))
