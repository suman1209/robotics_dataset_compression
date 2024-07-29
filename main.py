import numpy as np
from matplotlib import pyplot as plt
from utils.utils import get_storage, write_to_file
from utils.dataset_utils import (OriginalDataset,
                                 delta_between_images,
                                 plot_image_array,
                                 plot_hist_array,
                                 print_image_array,
                                 plot_modified_image_array)
from utils.sparse_representation import SparseRepresentation as SP


class TensorStorage(dict):
    def __init__(self, checkpoint, original_dataset, encoding_scheme: str,enlarge_factor):
        """
        Parameters
        ----------
        checkpoint : (int) how often to store the reference frame
        original_dataset: (OriginalDataset) the original dataset
        """
        super().__init__()
        self.checkpoint = checkpoint
        self.original_dataset = original_dataset
        self.encoding_scheme = encoding_scheme
        self.enlarge_factor = enlarge_factor

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
            i = ref_img_idx + 1
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

    def plot_encoded_data_sizes(self):
        sparse_matrices_size = [len(self[i])/(180*320) for i in range(1, len(self))]
        print(f"{sparse_matrices_size = }")
        plt.title(f"Sparse matrices sizes with checkpoint = {self.checkpoint}")
        plt.xlabel(f"img idx")
        plt.ylabel(f"len(sparse_matrix) %")
        plt.plot(sparse_matrices_size)
        plt.show()

    def plot_modified_image(self, idx):
        delta_image = self.sp.get_dense_representation(self[idx])
        print("shape: ", delta_image.shape)
        delta_image[delta_image != 0] *= self.enlarge_factor
        plot_image_array(delta_image)

    def plot_white_pixels_image(self, idx):
        delta_image = self.sp.get_dense_representation(self[idx])
        print("shape: ", delta_image.shape)
        count = 0
        
        modified_image = np.zeros_like(delta_image)

        for i in range(delta_image.shape[0]):
            for j in range(delta_image.shape[1]):
                if not np.array_equal(delta_image[i, j], [0, 0, 0]):
                    modified_image[i, j] = [255, 255, 255]
                    count += 1
        print('Non-Zero pixels count: ',count)
        plot_modified_image_array(modified_image,count)

    def printImageArray(self, idx):
        delta_image = self[idx]
        delta_image[delta_image != 0] *= self.enlarge_factor
        print("shape of delta image: ", delta_image.shape)
        print_image_array(delta_image)


if __name__ == "__main__":
    original_dataset_ = OriginalDataset(data_path="datasets/droid_100_sample_pictures_grayscale")
    tensor_storage = TensorStorage(checkpoint=10,
                                   original_dataset=original_dataset_,
                                   encoding_scheme="delta",
                                   enlarge_factor=20)
    num_images = len(original_dataset_)
    #num_images = 100
    print(f"#### Compressing and storing {num_images} images #### ")
    for idx in range(num_images):
        tensor_storage.add()

    img_0 = tensor_storage.get_image(0)
    print(f'{img_0.shape = }')
    img_1_original = original_dataset_[1]
    img_1_tensor_storage = tensor_storage.get_image(1)
    ref_img1 = tensor_storage.get_image(0)
    delta = delta_between_images(ref_img1, img_1_original)
    print(f"{img_1_original[0,0] = }")
    print(f"{img_1_tensor_storage[0,0] = }")
    print(f"{ref_img1[0,0] = }")
    print(f"{delta[0,0] = }")

    # orig_img18 = original_dataset_[18]
    # tensor_img18 = tensor_storage.get_image(18)
    # ref_img_for_img18 = tensor_storage.get_image(10)
    # delta = delta_between_images(ref_img_for_img18, orig_img18)
    # print(f"{ref_img_for_img18[7,194] = }")
    # print(f"{orig_img18[7,194] = }")
    # print(f"{tensor_img18[7,194] = }")
    # print(f"{delta[7,194] = }")
    # sp_mat = tensor_storage[1]
    # # print(f'{sp_mat[2316] = }, {len(sp_mat)}')
    # write_to_file(img_1_original, "img_1_original.txt")
    # write_to_file(img_1_tensor_storage, "img_1_tensor_storage.txt")
    # img11 = tensor_storage.get_image(11)
    # for i in range(num_images):
    #     original_img = original_dataset_[i]
    #     tensor_storage_img = tensor_storage.get_image(idx=i)
    #     assert np.array_equal(original_img, tensor_storage_img),\
    #         f"The original image({i}) and reconstructed img({i}) dont match!"


    print(f"total size in MB of tensor storage: {tensor_storage.get_size()}")
    print(f"total size in MB of original dataset: {original_dataset_.get_storage_size()}")

    # plot a decoded image

    image = tensor_storage.decompress_img(5,5)
    tensor_storage.plot_img(image)
    tensor_storage.plot_hist(1)


    #tensor_storage.plot_modified_image(2)
    #tensor_storage.plot_white_pixels_image(2) #plots non-negative delta values replaced with white pixel values
    # tensor_storage.printImageArray(1)

    tensor_storage.plot_encoded_data_sizes()