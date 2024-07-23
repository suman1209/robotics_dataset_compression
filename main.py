import numpy as np
from matplotlib import pyplot as plt

from utils.utils import get_storage, write_to_file
from utils.dataset_utils import (OriginalDataset,
                                 delta_between_images,
                                 plot_image_array,
                                 plot_hist_array)

from utils.sparse_representation import SparseRepresentation as SP


class TensorStorage(dict):
    def __init__(self, checkpoint, original_dataset):
        """

        Parameters
        ----------
        checkpoint : (int) how often to store the reference frame
        original_dataset: (OriginalDataset) the original dataset
        """
        super().__init__()
        self.checkpoint = checkpoint
        self.original_dataset = original_dataset

    def add(self):
        """this function adds the data to the dictionary"""
        idx = len(self)
        # special case where the checkpoint is zero
        if self.checkpoint == 0:
            self[idx] = self.original_dataset[idx]
        elif idx % self.checkpoint == 0:
            self[idx] = self.original_dataset[idx]
        else:
            """Encoding"""
            ref_img_idx = (idx // self.checkpoint) * self.checkpoint
            ref_img = self[ref_img_idx]
            orig_img = self.original_dataset[idx]
            delta = delta_between_images(ref_img, orig_img)
            # store the sparse representation
            self.sp = SP(delta.shape)
            sparse_matrix = self.sp.get_sparse_representation(delta)
            # print(f"{delta = }")
            # print(f"fraction of zero deltas: {np.count_nonzero(delta==0)/len(np.ndarray.flatten(delta)) = }")
            self[idx] = sparse_matrix

    def get_image(self, idx):
        """Here we need to reconstruct the original image and also verify that it is correct by
        comparing to the original image in the original dataset"""
        assert idx < len(self), (f"Trying to access idx {idx} which is not available in the TensorStorage,"
                                 f"available indices are {self.keys()}")
        # special case where the checkpoint is 0
        if self.checkpoint == 0:
            return self[0]
        elif idx % self.checkpoint == 0:
            return self[idx]
        else:
            # print(f"{self[idx // self.checkpoint][0,0,-1]} + {self[idx][0,0,-1]}")
            ref_img_idx = (idx // self.checkpoint) * self.checkpoint
            out = self[ref_img_idx] + self.sp.get_dense_representation(self[idx])

            return np.array(out, dtype=np.int64)

    def get_size(self):
        total_size = 0
        for idx, array in self.items():
            total_size += get_storage(array)
        return total_size

    def plot_img(self, idx):
        plot_image_array(self[idx])

    def plot_hist(self, idx):
        plot_hist_array(self[idx])

    def plot_encoded_data_sizes(self):
        sparse_matrices_size = [len(self[i])/(180*320) for i in range(1, len(self))]
        print(f"{sparse_matrices_size = }")
        plt.title(f"Sparse matrices sizes with checkpoint = {self.checkpoint}")
        plt.xlabel(f"img idx")
        plt.ylabel(f"len(sparse_matrix) %")
        plt.plot(sparse_matrices_size)
        plt.show()


if __name__ == "__main__":
    original_dataset_ = OriginalDataset(data_path="./datasets/droid_100_sample_pictures")
    tensor_storage = TensorStorage(checkpoint=10,
                                   original_dataset=original_dataset_)
    for idx in range(len(original_dataset_)):
        print(f"adding image#{idx + 1} to the tensor storage")
        tensor_storage.add()
    # print(tensor_storage)
    img_0 = tensor_storage.get_image(0)
    print(f'{img_0.shape = }')
    img_1 = tensor_storage.get_image(1)
    img_1_original = original_dataset_[1]
    img_1_tensor_storage = tensor_storage.get_image(1)
    ref_img1 = tensor_storage.get_image(0)
    delta = delta_between_images(ref_img1, img_1_original)
    print(f"{img_1_original[0,0] = }")
    print(f"{img_1_tensor_storage[0,0] = }")
    print(f"{ref_img1[0,0] = }")
    print(f"{delta[0,0] = }")

    orig_img18 = original_dataset_[18]
    tensor_img18 = tensor_storage.get_image(18)
    ref_img_for_img18 = tensor_storage.get_image(10)
    delta = delta_between_images(ref_img_for_img18, orig_img18)
    print(f"{ref_img_for_img18[7,194] = }")
    print(f"{orig_img18[7,194] = }")
    print(f"{tensor_img18[7,194] = }")
    print(f"{delta[7,194] = }")
    sp_mat = tensor_storage[1]
    # print(f'{sp_mat[2316] = }, {len(sp_mat)}')
    write_to_file(img_1_original, "img_1_original.txt")
    write_to_file(img_1_tensor_storage, "img_1_tensor_storage.txt")

    for i in range(len(original_dataset_)):
        assert np.array_equal(original_dataset_[i], tensor_storage.get_image(idx=i)),\
            f"The original image({i}) and reconstructed img({i}) dont match!"


    print(f"total size in MB of tensor storage: {tensor_storage.get_size()}")
    print(f"total size in MB of original dataset: {original_dataset_.get_storage_size()}")

    # plot a decoded image

    tensor_storage.plot_img(0)
    tensor_storage.plot_hist(1)

    tensor_storage.plot_encoded_data_sizes()