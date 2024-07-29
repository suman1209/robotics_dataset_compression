import numpy as np


def get_storage(array: np.array) -> float:
    """
    Returns the storage size in (mB) of an array
    Parameters
    ----------
    array : (np.array) the array whose storage size is to be returned

    Returns
    -------

    """
    size = {"int8": 8, "uint8": 8, "int64": 64, "float64": 64}
    dtype = str(array.dtype)
    if dtype not in size.keys():
        raise Exception(f"Unsupported storage type: {dtype}")
    return size[dtype] * len(array) / (1024*8)


def write_to_file(array, filename: str):
    flattened_array = np.ndarray.flatten(array)
    with open(filename, "w") as f:
        for item in flattened_array:
            f.write(str(item) + "\n")


if __name__ == "__main__":
    test_tensor = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int8)
    storage_size = get_storage(test_tensor)
    print(f"{storage_size = } mB")
