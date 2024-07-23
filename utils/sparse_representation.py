
class SparseRepresentation:

    def __init__(self, img_shape: list):
        self.img_shape = img_shape

    def get_sparse_representation(self, delta_array):
        delta_array = delta_array.reshape(self.img_shape[-1], self.img_shape[0], self.img_shape[1])
        print(f"{delta_array.shape = }")

    def get_delta_from_sparse_representation(self):
        pass

