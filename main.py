import numpy as np
from utils.utils import get_storage
from utils.dataset_utils import (OriginalDataset,
                                 delta_between_images,
                                 plot_image_array,
                                 plot_hist_array)


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
        if idx % self.checkpoint == 0:
            self[idx] = self.original_dataset[idx]
        else:
            """@todo here we need to perform some encoding and store the encoded tensor"""
            ref_img = self[idx // self.checkpoint]
            orig_img = self.original_dataset[idx]
            delta = delta_between_images(ref_img, orig_img)
            # print(f"{delta = }")
            # print(f"fraction of zero deltas: {np.count_nonzero(delta==0)/len(np.ndarray.flatten(delta)) = }")
            self[idx] = delta

    def get_image(self, idx):
        """Here we need to reconstruct the original image and also verify that it is correct by
        comparing to the original image in the original dataset"""
        assert idx < len(self), (f"Trying to access idx {idx} which is not available in the TensorStorage,"
                                 f"available indices are {self.keys()}")
        if idx % self.checkpoint == 0:
            return self[idx]
        else:
            # print(f"{self[idx // self.checkpoint][0,0,-1]} + {self[idx][0,0,-1]}")
            return self[idx // self.checkpoint] + self[idx]

    def get_size(self):
        total_size = 0
        for idx, array in self.items():
            total_size += get_storage(array)
        return total_size

    def plot_img(self, idx):
        plot_image_array(self[idx])

    def plot_hist(self, idx):
        plot_hist_array(self[idx])


if __name__ == "__main__":
    original_dataset_ = OriginalDataset(data_path="./datasets/droid_100_sample_pictures")
    img_0 = original_dataset_[0]
    tensor_storage = TensorStorage(checkpoint=10,
                                   original_dataset=original_dataset_)
    for idx in range(len(original_dataset_)):
        tensor_storage.add()
    print(tensor_storage)
    img_0 = tensor_storage.get_image(0)
    print(f'{img_0.shape = }')
    img_1 = tensor_storage.get_image(1)
    img_1_original = original_dataset_[1]
    print(f"{img_1[0,0,-1] = }")
    print(f"{img_1_original[0,0,-1] = }")
    assert np.array_equal(img_1, img_1_original), "The original image and reconstructed img dont match!"
    print(f"total size in MB of tensor storage: {tensor_storage.get_size()}")
    print(f"total size in MB of original dataset: {original_dataset_.get_storage_size()}")

    # plot a decoded image

    tensor_storage.plot_img(0)
    tensor_storage.plot_hist(1)