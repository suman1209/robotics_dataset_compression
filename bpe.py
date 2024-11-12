from collections import Counter

from utils.dataset_utils import OriginalDataset
import numpy as np
def find_most_freq_pair(arr: np.array, merges: dict):
    for i, j in zip(arr, arr[1:]):
        count = merges.get((i, j), 0)
        merges[(i, j)] = count + 1
    merges = {k: v for k, v in sorted(merges.items(), key=lambda x: x[1], reverse=True)}
    most_freq_pair = list(merges.items())[0][0]
    count = list(merges.items())[0][1]
    return most_freq_pair, count, merges

def merge(arr: np.array, pair: tuple, new_token):
    merged_arr = []
    j = 0
    while j < len(arr):
        if j >= len(arr) - 1:
            break
        if arr[j] == pair[0] and arr[j + 1] == pair[1]:
            merged_arr.append(new_token)
            j += 2
        else:
            merged_arr.append(r[i][j])
            j += 1
    print(len(r[0]), len(merged_arr))
    return merged_arr


original_dataset = OriginalDataset('datasets/droid_100_sample_pictures')
len_ = (original_dataset.__len__())
image1 = original_dataset[0]
image2 = original_dataset[1]
image1 = np.array(image1, dtype = np.int16)
image2 = np.array(image2, dtype = np.int16)
diff_img = image1 - image2
r = diff_img[:,:, 0]
r_flattened_diff = r.flatten()
r_freq = sorted(Counter(r_flattened_diff).items())
r_freq_dict = dict(Counter(r_flattened_diff))

# find the most frequently occuring pair
tokens = sorted(list(r_freq_dict.keys()))
merge_book = {}

num_iterations = 2
arr = [0, 0, 0, 0, 0, 0]
for i in range(num_iterations):
    merges = {}
    new_token = tokens[-1] + 1
    tokens.append(new_token)
    most_freq_pair, count, merges = find_most_freq_pair(arr, merges)
    merged_arr = merge(arr, most_freq_pair, new_token)
    print(f"iteration: {i},"
          f"most_freq_pair, freq = {most_freq_pair}: {count}"
          f" compression: {len(arr)} -> {len(merged_arr)}"
          f" {new_token = }")
    print(f"{merged_arr = }")
    arr = merged_arr
print(arr)