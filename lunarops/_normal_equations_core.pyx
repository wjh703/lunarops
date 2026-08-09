# cython: language_level=3

cimport numpy as cnp
from libc.math cimport isfinite


def accumulate_sparse_batch(
    cnp.float64_t[:, ::1] normal_matrix,
    cnp.float64_t[::1] right_hand_side,
    cnp.intp_t[::1] row_offsets,
    cnp.intp_t[::1] column_indices,
    cnp.float64_t[::1] values,
    cnp.float64_t[::1] observations,
    cnp.float64_t[::1] weights,
):
    """Accumulate validated CSR-like sparse rows into normal equations."""
    cdef Py_ssize_t parameter_count = normal_matrix.shape[0]
    cdef Py_ssize_t row_count = observations.shape[0]
    cdef Py_ssize_t row, start, stop, left, right, index_left, index_right
    cdef double weight, observation, value_left, weighted_left, contribution
    cdef double lpl = 0.0

    if normal_matrix.shape[1] != parameter_count:
        raise ValueError("normal_matrix must be square.")
    if right_hand_side.shape[0] != parameter_count:
        raise ValueError("right_hand_side does not match normal_matrix.")
    if weights.shape[0] != row_count:
        raise ValueError("weights and observations must have the same length.")
    if row_offsets.shape[0] != row_count + 1:
        raise ValueError("row_offsets must contain one entry per row plus the terminal offset.")
    if values.shape[0] != column_indices.shape[0]:
        raise ValueError("column_indices and values must have the same length.")
    if row_offsets[0] != 0 or row_offsets[row_count] != values.shape[0]:
        raise ValueError("row_offsets do not span the sparse values.")

    for row in range(row_count):
        start = row_offsets[row]
        stop = row_offsets[row + 1]
        if start < 0 or stop < start or stop > values.shape[0]:
            raise ValueError("row_offsets must be monotonic and within the sparse values.")
        if not isfinite(observations[row]) or not isfinite(weights[row]) or weights[row] < 0.0:
            raise ValueError("observations and weights must be finite, with non-negative weights.")
        for left in range(start, stop):
            if column_indices[left] < 0 or column_indices[left] >= parameter_count:
                raise ValueError("column index is outside the normal matrix.")
            if not isfinite(values[left]):
                raise ValueError("sparse values must be finite.")

    with nogil:
        for row in range(row_count):
            start = row_offsets[row]
            stop = row_offsets[row + 1]
            weight = weights[row]
            observation = observations[row]
            lpl += weight * observation * observation
            for left in range(start, stop):
                index_left = column_indices[left]
                value_left = values[left]
                weighted_left = weight * value_left
                right_hand_side[index_left] += weighted_left * observation
                for right in range(start, stop):
                    index_right = column_indices[right]
                    contribution = weighted_left * values[right]
                    normal_matrix[index_left, index_right] += contribution

    return lpl
