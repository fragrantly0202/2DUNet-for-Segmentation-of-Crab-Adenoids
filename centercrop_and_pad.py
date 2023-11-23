import numpy as np

def center_crop_and_fill(matrix, target_size, fill_value=0):
    matrix_size = matrix.shape
    target_height, target_width = target_size

    start_row = max(0, (matrix_size[0] - target_height) // 2)
    start_col = max(0, (matrix_size[1] - target_width) // 2)

    end_row = min(matrix_size[0], start_row + target_height)
    end_col = min(matrix_size[1], start_col + target_width)

    cropped_matrix = matrix[start_row:end_row, start_col:end_col]

    result_matrix = np.full(target_size, fill_value, dtype=matrix.dtype)

    insert_start_row = max(0, (target_height - cropped_matrix.shape[0]) // 2)
    insert_start_col = max(0, (target_width - cropped_matrix.shape[1]) // 2)

    result_matrix[insert_start_row:insert_start_row + cropped_matrix.shape[0],
                  insert_start_col:insert_start_col + cropped_matrix.shape[1]] = cropped_matrix

    return result_matrix