import numpy as np
import time
from utils.utils import get_storage
from utils.dataset_utils import (OriginalDataset,
                                 delta_between_images,
                                 plot_image_array,
                                 print_image_array,
                                 print_image_array,
                                 plot_hist_array)


class TensorStorage(dict):
    def __init__(self, checkpoint, original_dataset):
        """

        Parameters
        ----------
        checkpoint : (int) how often to store the reference frame
        original_dataset: (OriginalDataset) the original dataset
        """
        super().__init__(selfself)
        self.checkpoint = checkpoint
        self.original_dataset = original_dataset
        self.enlarge_factor = 35
        self.enlarge_factor = 35

    def add(self):
        """this function adds the data to the dictionary"""
        idx = len(self)
        # special case where the checkpoint is zero
        if self.checkpoint == 0:
            if idx == 0:
                self[idx] = self.original_dataset[idx]
                return None
            else:
                ref_img_idx = 0
                self.encode_img(idx, ref_img_idx)
        else:
            if idx % self.checkpoint == 0:
                self[idx] = self.original_dataset[idx]
            else:
                ref_img_idx = (idx // self.checkpoint) * self.checkpoint
                self.encode_img(idx, ref_img_idx)
            return None

    def encode_img(self, idx: int, ref_img_idx: int):
        """Encoding"""
        if self.encoding_scheme == "FOR":
            # ref_img_idx = (idx // self.checkpoint) * self.checkpoint
            ref_img = self[ref_img_idx]
            orig_img = self.original_dataset[idx]
            delta = delta_between_images(ref_img, orig_img)
            # store the sparse representation
            self.sp = SP(delta.shape)
            sparse_matrix = self.sp.get_sparse_representation(delta)
            self[idx] = sparse_matrix
            return
        elif self.encoding_scheme == "delta":
            # the reference image becomes the previous image
            if idx == 1:
                ref_img_idx_ = 0
                ref_img = self[ref_img_idx_]
                orig_img = self.original_dataset[idx]
                delta = delta_between_images(ref_img, orig_img)
                # store the sparse representation
                self.sp = SP(delta.shape)
                sparse_matrix = self.sp.get_sparse_representation(delta)
                self[idx] = sparse_matrix
                return
            ref_img_idx_ = idx - 1
            # Note that we want to decode the previous image everytime
            ref_img = self.original_dataset[ref_img_idx_]
            orig_img = self.original_dataset[idx]
            delta = delta_between_images(ref_img, orig_img)
            # store the sparse representation
            self.sp = SP(delta.shape)
            sparse_matrix = self.sp.get_sparse_representation(delta)
            self[idx] = sparse_matrix
        else:
            raise Exception(f"Unknown encoding scheme: {self.encoding_scheme}")

    def get_image(self, idx):
        """Here we need to reconstruct the original image and also verify that it is correct by
        comparing to the original image in the original dataset"""
        assert idx < len(self), (f"Trying to access idx {idx} which is not available in the TensorStorage,"
                                 f"available indices are {self.keys()}")
        # special case where the checkpoint is 0
        if self.checkpoint == 0:
            if idx == 0:
                return self[0]
            else:
                ref_img_idx = 0
                return self.decompress_img(idx, ref_img_idx)
        else:
            if idx % self.checkpoint == 0:
                return self[idx]
            else:
                ref_img_idx = (idx // self.checkpoint) * self.checkpoint
                out = self.decompress_img(idx, ref_img_idx)
                return np.array(out, dtype=np.int64)

    def decompress_img(self, idx, ref_img_idx):
        """decoding"""
        if self.encoding_scheme == "FOR":
            # print(f"{self[idx // self.checkpoint][0,0,-1]} + {self[idx][0,0,-1]}")
            # ref_img_idx = (idx // self.checkpoint) * self.checkpoint
            out = self[ref_img_idx] + self.sp.get_dense_representation(self[idx])
        elif self.encoding_scheme == "delta":
            # ref_img_idx = (idx // self.checkpoint) * self.checkpoint
            out = self[ref_img_idx].astype(np.float64)
            i = 1
            while i < idx + 1:
                out += self.sp.get_dense_representation(self[i])
                i += 1
        else:
            raise Exception(f"Unknown encoding scheme: {self.encoding_scheme}")
        return out

    def get_size(self):
        total_size = 0
        for idx, array in self.items():
            total_size += get_storage(array)
        return total_size

    def plot_img(self, idx):
        plot_image_array(self[idx])

    def plot_hist(self, idx):
        plot_hist_array(self[idx])

    def plot_modified_image(self, idx):
        delta_image = self[idx]
        delta_image[delta_image != 0]*=self.enlarge_factor
        plot_image_array(delta_image)

    def printImageArray(self,idx):
        delta_image = self[idx]
        delta_image[delta_image != 0] *=self.enlarge_factor
        print("shape of delta image: ",delta_image.shape )
        print_image_array(delta_image)


    def plot_modified_image(self, idx):
        delta_image = self[idx]
        delta_image[delta_image != 0] *= self.enlarge_factor
        plot_image_array(delta_image)





if __name__ == "__main__":
    original_dataset_ = OriginalDataset(data_path="./datasets/droid_100_sample_pictures")
    img_0 = original_dataset_[0]
    tensor_storage = TensorStorage(checkpoint=20,
                                   original_dataset=original_dataset_)
    for idx in range(len(original_dataset_)):
        tensor_storage.add()
    ##print(tensor_storage)
    img_0 = tensor_storage.get_image(0)
    print(f'{img_0.shape = }')
    img_1_original = original_dataset_[1]
    print(f"{img_1[0,0,-1] = }")
    print(f"{img_1_original[0,0,-1] = }")
    assert np.array_equal(img_1, img_1_original), "The original image and reconstructed img dont match!"
    print(f"total size in MB of tensor storage: {tensor_storage.get_size()}")
    print(f"total size in MB of original dataset: {original_dataset_.get_storage_size()}")

    # plot a decoded image
    tensor_storage.plot_modified_image(4)
    tensor_storage.printImageArray(4)
    #tensor_storage.plot_hist(2)
