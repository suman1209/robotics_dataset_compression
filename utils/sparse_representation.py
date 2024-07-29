import numpy as np

class SparseRepresentation:
    def __init__(self, img_shape):
        self.img_shape = img_shape

    @staticmethod
    def get_sparse_representation(delta: np.array) -> np.array:
        sparse_matrix = []
        for row_idx, row in enumerate(delta):
            for col_idx, value in enumerate(row):
                if value != 0:
                    sparse_matrix.append((row_idx, col_idx, value))
        return np.array(sparse_matrix)

    def get_dense_representation(self, sparse_repr: np.array) -> np.array:
        delta = np.zeros(shape=self.img_shape, dtype=np.float32)
        for item in sparse_repr:
            row_idx, col_idx, delta_value = item
            # Ensure row_idx and col_idx are integers
            delta[int(row_idx), int(col_idx)] = delta_value
        return delta

if __name__ == "__main__":
    delta_img_ = np.array([[0, -2, 0], [1, 0, 0], [0, 0, 0], [110, -50, 0]], dtype=np.int8)
    sp = SparseRepresentation(img_shape=delta_img_.shape)
    sparse_matrix = sp.get_sparse_representation(delta_img_)

    delta_reformed = sp.get_dense_representation(sparse_matrix)
    print(f'{delta_reformed = }')
