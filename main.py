import math

import numpy as np
from tqdm import tqdm

from utils.dataset_utils import OriginalDataset
from utils.compare_compression_algo import generate_results
from utils.tensor_storage import TensorStorage
from utils.utils import write_to_file
import pickle
from utils.dataset_utils import delta_between_images

if __name__ == "__main__":
    # initial configurations
    checkpoint = 0
    enlarge_factor = 20
    num_images = 20
    original_dataset_path = "datasets/droid_100_sample_pictures"
    encoding_scheme = "delta"
    debug_mode = False

    # initialisation
    original_dataset_ = OriginalDataset(data_path=original_dataset_path)
    print("# profiling the deltas to calculate a suitable offset. #")
    min_delta = math.inf
    for i in range(1, num_images):
        ref_img = original_dataset_[i - 1].astype(np.int64)
        orig_img = original_dataset_[i].astype(np.int64)
        delta = delta_between_images(ref_img=ref_img, orig_img=orig_img, offset=0)
        delta = np.reshape(delta, (1, -1))
        if delta.min() < min_delta:
            min_delta = delta.min()
    print(f'{min_delta=}')
    tensor_storage = TensorStorage(checkpoint=checkpoint,
                                   original_dataset=original_dataset_,
                                   encoding_scheme=encoding_scheme,
                                   enlarge_factor=enlarge_factor,
                                   debug_mode=debug_mode,
                                   offset=min_delta)
    # num_images = len(original_dataset_)

    msg1 = f"# Compressing and storing {num_images} images #"
    for idx in tqdm(range(num_images), msg1):
        tensor_storage.add()
        if debug_mode:
            file_name = f"datasets/compressed/droid/tensor_storage{idx}.pkl"
            with open(file_name, 'wb') as f:
                pickle.dump(tensor_storage[idx], f)
    msg2 = f"# Verifying the reconstructed image {num_images} images #"
    for i in tqdm(range(num_images), msg2):
        original_img = original_dataset_[i]
        if i == 32:
            print(f"debug")
        tensor_storage_img = tensor_storage.get_image(idx=i)
        if i == 32:
            print(f"debug")
            write_to_file(original_img, f"img_{i}_original.txt")
            write_to_file(tensor_storage_img, f"img_{i}_tensor_storage.txt")

        assert np.array_equal(original_img, tensor_storage_img),\
            f"The original image({i}) and reconstructed img({i}) dont match!"


    # plot the deltas of a given image
    tensor_storage.plot_white_pixels_image(1)
    print(f"# Generating results #")
    generate_results(original_dataset_=original_dataset_,
                     comp_algo_list=['FOR', 'delta'],
                     checkpoint=checkpoint,
                     enlarge_factor=enlarge_factor,
                     num_images=num_images,
                     debug_mode=debug_mode)
