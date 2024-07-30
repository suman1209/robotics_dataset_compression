import numpy as np
from utils.dataset_utils import OriginalDataset
from utils.compare_compression_algo import generate_results
from utils.tensor_storage import TensorStorage
from tqdm import tqdm
from utils.utils import write_to_file

if __name__ == "__main__":
    # initial configurations
    checkpoint = 10
    enlarge_factor = 20
    num_images = 20
    # remember to change the color to True or false depending on dataset
    original_dataset_path = "datasets/droid_100_sample_pictures"
    # whether to read color or grayscale images
    color = True
    encoding_scheme = "delta"

    # this decompresses all the compressed images and ensures they are the same
    verify = True
    debug = False
    # initialisation
    original_dataset_ = OriginalDataset(data_path=original_dataset_path, color=color)
    tensor_storage = TensorStorage(checkpoint=checkpoint,
                                   original_dataset=original_dataset_,
                                   encoding_scheme=encoding_scheme,
                                   enlarge_factor=enlarge_factor,
                                   color=color)
    # num_images = len(original_dataset_)

    msg1 = f"# Compressing and storing {num_images} images #"
    for idx in tqdm(range(num_images), msg1):
        tensor_storage.add()
    if verify:
        msg2 = f"# Verifying the reconstructed image {num_images} images #"
        for i in tqdm(range(num_images), msg2):
            original_img = original_dataset_[i]
            tensor_storage_img = tensor_storage.get_image(idx=i)
            if debug:
                write_to_file(original_img, f"img_{i}_original.txt")
                write_to_file(tensor_storage_img, f"img_{i}_tensor_storage.txt")
            assert np.array_equal(original_img, tensor_storage_img),\
                f"The original image({i}) and reconstructed img({i}) dont match!"
    # plot the deltas of a given image
    delta_to_plot = 2
    assert delta_to_plot < num_images, (f"cannot plot delta {delta_to_plot} only {num_images} were added"
                                         "to the tensor storage")
    tensor_storage.plot_white_pixels_image(delta_to_plot)
    print(f"# Generating results #")
    generate_results(original_dataset_=original_dataset_,
                     comp_algo_list=['FOR', 'delta'],
                     checkpoint=checkpoint,
                     enlarge_factor=enlarge_factor,
                     num_images=num_images,
                     color=color)
