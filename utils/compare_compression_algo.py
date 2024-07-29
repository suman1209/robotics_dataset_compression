## this code compares our compression algos
from utils.tensor_storage import TensorStorage
import numpy as np
from matplotlib import pyplot as plt
import time
from utils.dataset_utils import OriginalDataset
from prettytable import PrettyTable


def generate_comparison_table(performance_data):
    table = PrettyTable()
    table.field_names = ["Compression Algo", "Compression Size (MB)", "Original Size (MB)", "Compression Time (s)", "Decompression Time (s)", "Compression Ratio"]

    for method, comp_size, orig_size, comp_time, decomp_time, comp_ratio in performance_data:
        comp_size = f"{comp_size:.3f}"
        orig_size = f"{orig_size:.3f}"
        comp_time = f"{comp_time:.3f}"
        decomp_time = f"{decomp_time:.3f}"
        comp_ratio = f"{comp_ratio:.3f}"
        table.add_row([method, comp_size, orig_size, comp_time, decomp_time, comp_ratio])

    return table


def generate_results(original_dataset_,
                     comp_algo_list,
                     checkpoint,
                     enlarge_factor,
                     num_images):
    performance_data = []

    for comp_algo in comp_algo_list:
        tensor_storage = TensorStorage(checkpoint=checkpoint,
                                       original_dataset=original_dataset_,
                                       encoding_scheme=comp_algo,
                                       enlarge_factor=enlarge_factor)
        print(f"#### Compressing  using {comp_algo} and storing {num_images} images #### ")
        comp_start_time = time.time()
        for idx in range(num_images):
            tensor_storage.add()
        comp_end_time = time.time()

        total_comp_time = (comp_end_time - comp_start_time) / num_images
        # print(f'Average Compression Time of {comp_algo}: {total_comp_time} ')

        decomp_start_time = time.time()
        for idx in range(num_images):
            tensor_storage.get_image(idx)
        decomp_end_time = time.time()

        total_decomp_time = (decomp_end_time - decomp_start_time) / num_images
        # print(f'Average Decompression Time of {comp_algo}: {total_decomp_time} ')

        comp_size = tensor_storage.get_size()
        orig_size = original_dataset_.get_storage_size(num_images)

        # print(f"total size in KB of tensor storage: {comp_size}")
        # print(f"total size in KB of original dataset: {orig_size}")

        comp_ratio = comp_size / orig_size

        performance_data.append([comp_algo, comp_size, orig_size, total_comp_time, total_decomp_time, comp_ratio])

    comp_table = generate_comparison_table(performance_data)
    print(comp_table)


if __name__ == "__main__":
    original_dataset_ = OriginalDataset(data_path="datasets/droid_100_sample_pictures")
    generate_results(original_dataset_=original_dataset_,
                     comp_algo_list=['FOR', 'delta'],
                     checkpoint=10,
                     enlarge_factor=10,
                     num_images=100,
                     )



