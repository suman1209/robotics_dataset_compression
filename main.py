import numpy as np
from utils.dataset_utils import OriginalDataset
from utils.compare_compression_algo import generate_results
from utils.tensor_storage import TensorStorage


if __name__ == "__main__":
    # initial configurations
    checkpoint = 10
    enlarge_factor = 20
    num_images = 3
    original_dataset_path = "datasets/droid_100_sample_pictures"
    encoding_scheme = "delta"

    # initialisation
    original_dataset_ = OriginalDataset(data_path=original_dataset_path)
    tensor_storage = TensorStorage(checkpoint=checkpoint,
                                   original_dataset=original_dataset_,
                                   encoding_scheme=encoding_scheme,
                                   enlarge_factor=enlarge_factor)
    # num_images = len(original_dataset_)

    print(f"# Compressing and storing {num_images} images #")
    for idx in range(num_images):
        tensor_storage.add()

    print(f"# Verifying the reconstructed image {num_images} images #")
    for i in range(num_images):
        original_img = original_dataset_[i]
        tensor_storage_img = tensor_storage.get_image(idx=i)
        assert np.array_equal(original_img, tensor_storage_img),\
            f"The original image({i}) and reconstructed img({i}) dont match!"


    # plot the deltas of a given image
    # tensor_storage.plot_white_pixels_image(2)
    print(f"# Generating results #")
    generate_results(original_dataset_=original_dataset_,
                     comp_algo_list=['FOR', 'delta'],
                     checkpoint=10,
                     enlarge_factor=10,
                     num_images=10,
                     )
