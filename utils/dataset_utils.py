from matplotlib.ticker import PercentFormatter
from torchvision.datasets import VisionDataset
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import get_storage

def delta_between_images(ref_img: np.array, orig_img: np.array) -> np.array:
    """This function calculates the difference between two grayscale images."""
    # Check if the images have the same shape
    assert ref_img.shape == orig_img.shape, (
        f"ref({ref_img.shape}) and orig({orig_img.shape}) images must have the same shape"
    )
    
    # Convert images to float32 for precise calculation
    ref_img = ref_img.astype(np.float32)
    orig_img = orig_img.astype(np.float32)
    
    # Calculate the delta
    delta = orig_img - ref_img
    
    return delta

def read_grayscale_img(path) -> np.array:
    """Reads an image in grayscale mode."""
    assert os.path.exists(path), "Provided path does not exist!"
    gray_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return gray_img

def plot_image(image_path, figsize=(5, 5)):
    img = read_grayscale_img(image_path)
    title = str(image_path) + f"_{img.shape}"
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.show()

def plot_image_array(image_array: np.array, figsize=(5, 5)):
    title = "image_array" + f"_{image_array.shape}"
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(abs(image_array), cmap='gray')
    plt.show()

def plot_modified_image_array(image_array: np.array, pixel_count, figsize=(5, 5)):
    total_pixels = image_array.shape[0] * image_array.shape[1]
    title = ("image_array" + f"_{image_array.shape}" +
             f" num_white_pixels_{round((pixel_count/total_pixels)*100, 2)} %")
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(abs(image_array), cmap='gray')
    plt.show()

def plot_hist_array(hist_array: np.array, figsize=(5, 5)):
    title = "hist_array" + f"_{hist_array.shape}"
    data = hist_array.flatten()
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.hist(data, weights=np.ones(len(data)) / len(data))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.show()

def print_image_array(image_array: np.array):
    print("image array:")
    print(image_array)

class OriginalDataset(VisionDataset):
    def __init__(self, data_path: str):
        super(OriginalDataset, self).__init__()
        self.data_path = data_path

    def __getitem__(self, idx: int) -> np.array:
        """Returns the grayscale image corresponding to the index."""
        if idx >= len(self) or idx < 0:
            raise IndexError
        img_path = f'{self.data_path}/idx_{idx}.png'
        assert os.path.exists(img_path), f"Invalid img_path: {img_path} in {self.data_path}"
        img = read_grayscale_img(img_path)
        return img

    def __len__(self) -> int:
        _, _, files = next(os.walk(self.data_path))
        return len([i for i in files if "resize" not in i])

    def __str__(self):
        return f"OriginalDataset({self.data_path})"

    def get_storage_size(self):
        "returns the total storage size of the dataset"
        total_storage = 0
        for data in self:
            total_storage += get_storage(data)
        return total_storage

if __name__ == '__main__':
    original_dataset = OriginalDataset('../datasets/droid_100_sample_pictures_grayscale')
    len_ = len(original_dataset)
    print(len_)
    storage_size = original_dataset.get_storage_size()
    print(f"{storage_size = }")
