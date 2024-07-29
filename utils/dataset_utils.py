from matplotlib.ticker import PercentFormatter
from torchvision.datasets import VisionDataset
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import get_storage


def delta_between_images(ref_img: np.array, orig_img: np.array) -> np.array:
    """This function calculates the difference between two rgb images"""
    assert ref_img.shape == orig_img.shape, (f"ref({ref_img.shape}) and"
                                             f" orig({orig_img.shape}) images must have the same shape")
    # TODO, need to debug why the orig18 and recon18 are not matching when the dtypes are set to int8
    delta = np.zeros_like(ref_img, dtype=np.float32)
    ref_img = ref_img.astype(np.float32)
    orig_img = orig_img.astype(np.float32)
    for i in range(3):
        # print(f"{image1[:, :, i] = }")
        # print(f"{image2[:, :, i] = }")
        if orig_img[7, 194, 0] == 41:
            pass
        delta[:, :, i] = orig_img[:, :, i] - ref_img[:, :, i]
        # print(f"{delta[:, :, i] = }")
        # assert False
    return delta


def read_rgb_img(path) -> np.array:
    assert os.path.exists(path), "provided path does not exist!"
    bgr_img = cv2.imread(path, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return rgb_img


def plot_image(image_path, figsize=(5, 5)):
    img = read_rgb_img(image_path)
    title = str(image_path) + f"_{img.shape}"
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(img)
    plt.show()


def plot_image_array(image_array: np.array, figsize=(5, 5)):
    title = "image_array" + f"_{image_array.shape}"
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(abs(image_array))
    plt.show()


def plot_modified_image_array(image_array: np.array, pixel_count_percent, figsize=(5, 5)):
    total_pixels = image_array.shape[0] * image_array.shape[1]
    title = ("image_array" + f"_{image_array.shape}" +
             f"\nnum_white_pixels(non[0, 0, 0])_{round(pixel_count_percent, 2)} %")
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(abs(image_array))
    plt.show()


def plot_hist_array(hist_array: np.array, figsize=(5, 5)):
    title = "hist_array" + f"_{hist_array.shape}"
    data = np.ndarray.flatten(hist_array.flatten())
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.hist(data, weights=np.ones(len(data)) / len(data))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.show()


def print_image_array(image_array:np.array):
    print("image array:")
    print(image_array)


class OriginalDataset(VisionDataset):
    def __init__(self, data_path: str):
        super(OriginalDataset, self).__init__()
        self.data_path = data_path

    def __getitem__(self, idx: int) -> np.array:
        """
        this dataset returns the image corresponding to the index
        """

        if idx >= len(self) or idx < 0:
            # needed for iterator stop condition
            raise IndexError
        # the img_file_path existence check
        img_path = f'{self.data_path}/idx_{idx}.png'
        assert os.path.exists(img_path), f"Invalid img_path: {img_path} in {self.data_path}"
        img = read_rgb_img(img_path)
        return img

    def __len__(self) -> int:
        dirpath, dir_names, files = next(os.walk(self.data_path))
        return len([i for i in files if "resize" not in i])

    def __str__(self):
        return f"OriginalDataset({self.data_path})"

    def get_storage_size(self, num_images):
        "returns the total storage size of the dataset"
        total_storage = 0
        for data in [self[i] for i in range(num_images)]:
            total_storage += get_storage(data)
        return total_storage


if __name__ == '__main__':
    original_dataset = OriginalDataset('../datasets/droid_100_sample_pictures')
    len_ = (original_dataset.__len__())
    print(len_)
    storage_size = original_dataset.get_storage_size()
    print(f"{storage_size = }")
