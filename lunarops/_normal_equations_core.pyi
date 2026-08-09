from numpy import ndarray


def accumulate_sparse_batch(
    normal_matrix: ndarray,
    right_hand_side: ndarray,
    row_offsets: ndarray,
    column_indices: ndarray,
    values: ndarray,
    observations: ndarray,
    weights: ndarray,
) -> float: ...
