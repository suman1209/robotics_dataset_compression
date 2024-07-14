import numpy as np
from torchvision.datasets import VisionDataset
import os

import cv2
import numpy
import matplotlib.pyplot as plt
import numpy as np


def read_rgb_img(path) -> np.array:
    assert os.path.exists(path), "provided path does not exist!"
    bgr_img = cv2.imread(path, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return rgb_img


def plot_image(image_path, figsize=(10, 10)):
    img = read_rgb_img(image_path)
    title = str(image_path) + f"_{img.shape}"
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(img)
    plt.show()


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


if __name__ == '__main__':
    original_dataset = OriginalDataset('../datasets/droid_100_sample_pictures')
    len_ = (original_dataset.__len__())
    print(len_)
