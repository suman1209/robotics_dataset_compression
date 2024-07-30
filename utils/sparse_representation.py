import numpy as np


class SparseRepresentation:
    def __init__(self, img_shape):
        self.img_shape = img_shape
    """
    Sparse representation of a sparse matrix
    """
    @staticmethod
    def get_sparse_representation(delta: np.array):
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
                r, g, b = col
                if r == 0 and g == 0 and b == 0:
                    full_zero += 1
                else:
                    sparse_matrix.append([row_idx, col_idx, r, g, b])
        # print(f"{full_zero = }")
        # print(f"{sparse_matrix = }")
        item = sparse_matrix[0]
        row_idx, col_idx, delta_value = item[0], item[1], [item[2], item[3], item[4]]
        assert isinstance(row_idx, (float, int)), f"row_idx is not a float, received {row_idx} of type {type(row_idx)}"
        assert isinstance(col_idx, (float, int))
        assert isinstance(delta_value, list)
        # @todo there is an overflow error here when using , dtype=np.int8
        return sparse_matrix

    def get_dense_representation(self, sparse_repr):
        """
        In this method, we convert the sparse representation back to the dense representation
        i.e sparse to delta again.
        basically the reverse of the example above
        """
        delta = np.zeros(shape=self.img_shape)
        for item in sparse_repr:
            row_idx, col_idx, delta_value = item[0], item[1], [item[2], item[3], item[4]]
            assert isinstance(row_idx, int), f"row_idx is not a float, received {row_idx} of type {type(row_idx)}"
            assert isinstance(col_idx, int)
            assert isinstance(delta_value, list)
            delta[int(row_idx), int(col_idx)] = delta_value
        return delta

    @staticmethod
    def get_sparse_representation_gray(delta: np.array) -> np.array:
        sparse_matrix = []
        for row_idx, row in enumerate(delta):
            for col_idx, value in enumerate(row):
                if value != 0:
                    sparse_matrix.append([row_idx, col_idx, value])
        return np.array(sparse_matrix)

    def get_dense_representation_gray(self, sparse_repr: np.array) -> np.array:
        delta = np.zeros(shape=self.img_shape, dtype=np.float32)
        for item in sparse_repr:
            row_idx, col_idx, delta_value = item
            # Ensure row_idx and col_idx are integers
            delta[int(row_idx), int(col_idx)] = delta_value
        return delta


if __name__ == "__main__":
    # test_case 1
    delta_img_ = np.array([[[0, -2, 0], [1, 0, 0]], [[0, 0, 0], [110, -50, 0]]], dtype=np.int8)
    sparse_matrix = SparseRepresentation.get_sparse_representation(delta_img_)

    delta_reformed = SparseRepresentation.get_dense_representation(sparse_matrix)
    print(f'{delta_reformed = }')

