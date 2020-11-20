import numpy as np


def is_nonetype(obj):
    return isinstance(obj, type(None))


def indices_to_hot(hot_indices, length, dtype=np.bool):
    """
    Only works for 1D-vector.
    """
    vector = np.zeros((length,), dtype=dtype)

    if dtype == np.bool:
        hot = np.True_
    else:
        hot = 1

    if len(hot_indices):
        vector[hot_indices] = hot
    return vector


def hot_to_indices(bool_array):
    """
    Only works for 1D-vector.
    """
    index_array = np.flatnonzero(bool_array)
    return index_array
