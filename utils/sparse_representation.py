import numpy
import numpy as np
from utils.dataset_utils import process_rgb, create_rgb_col

def process_col_row_idx(col_row_idx):
    if isinstance(col_row_idx, (int, np.int64)):
        return col_row_idx
    elif isinstance(col_row_idx, (list, np.ndarray)):
        return sum(col_row_idx)
    else:
        raise Exception(f'col_row_idx must be an int or a list, obtained {type(col_row_idx)}')


def process_delta_value(delta_value):
    result = []
    for value in delta_value:
        if isinstance(value, list):
            result.append(sum(value))
        elif isinstance(value, (int, numpy.uint8, np.int64)):
            result.append(value)
        else:
            raise Exception(f"Unexpected value type {type(value)}")
    return result


class SparseRepresentation:
    def __init__(self, img_shape, offset):
        self.img_shape = img_shape
        self.offset = offset
    """
    Sparse representation of a sparse matrix
    """
    @staticmethod
    def get_sparse_representation(delta: np.array) -> np.array:
        """
         e.g. delta_image    : [[[0, -2, 0], [1, 0, 0]],
                                [[0, 0, 0], [110, -50, 0]]]
              output_format: [(row_num, col_num, [r_delta, g_delta, b_delta]) ...]
              expected_output: [(0, 0, [0, -2, 0]), (0, 1, [1, 0, 0]),
                                           (1, 0, [110, -50, 0])]
        """
        sparse_matrix = []
        full_zero = 0
        for row_idx, row in enumerate(delta):
            for col_idx, col in enumerate(row):
                r, g, b = create_rgb_col(col)
                row_idx_ = process_rgb(row_idx)
                col_idx_ = process_rgb(col_idx)
                # print(r, g, b)
                if r == 0 and g == 0 and b == 0:
                    full_zero += 1
                else:
                    sparse_matrix.append([row_idx_, col_idx_, r, g, b])
        # print(f"{full_zero = }")
        # print(f"{sparse_matrix = }")
        item = sparse_matrix[0]
        row_idx, col_idx, delta_value = item[0], item[1], [item[2], item[3], item[4]]
        assert isinstance(row_idx, (float, int, list)), f"row_idx is not a float, received {row_idx} of type {type(row_idx)}"
        assert isinstance(col_idx, (float, int, list))
        assert isinstance(delta_value, list)
        return sparse_matrix

    def get_dense_representation(self, sparse_repr):
        """
        In this method, we convert the sparse representation back to the dense representation
        i.e sparse to delta again.
        basically the reverse of the example above
        """
        delta = np.ones(shape=self.img_shape, dtype=np.int64)
        for item in sparse_repr:
            row_idx, col_idx, delta_value = (np.int64(item[0]), np.int64(item[1]),
                                             np.int64([item[2], item[3], item[4]]))
            assert isinstance(row_idx, (int, np.int64, list, np.ndarray)), f"row_idx is not a int, received {row_idx} of type {type(row_idx)}"
            assert isinstance(col_idx, (int, np.int64, list, np.ndarray))
            assert isinstance(delta_value, (list, np.ndarray))
            delta_value = process_delta_value(delta_value)
            row_idx = process_col_row_idx(row_idx)
            col_idx = process_col_row_idx(col_idx)
            # print(f"row_idx: {row_idx}, col_idx: {col_idx}")
            delta[int(row_idx), int(col_idx)] = delta_value - (self.offset * np.ones_like(delta_value))
        return delta


if __name__ == "__main__":
    # test_case 1
    delta_img_ = np.array([[[0, -2, 0], [1, 0, 0]], [[0, 0, 0], [110, -50, 0]]], dtype=np.int8)
    sparse_matrix = SparseRepresentation.get_sparse_representation(delta_img_)


