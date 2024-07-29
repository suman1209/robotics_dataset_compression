import math
from typing import Union

import numpy as np


def get_storage(array: Union[np.array, list]) -> float:
    """
    Returns the storage size in (mB) of an array
    Parameters
    ----------
    array : (np.array) the array whose storage size is to be returned

    Returns
    -------
    """
    if isinstance(array, type(np.array([1, 2, 3]))):
        size = {"int8": 8, "uint8": 8, "int64": 64, "float64": 64}
        dtype = str(array.dtype)
        total_elements = array.shape[0] * array.shape[1] * array.shape[2]
        if dtype not in size.keys():
            raise Exception(f"Unsupported storage type: {dtype}")
        return size[dtype] * total_elements / (1024 * 1024)

    elif isinstance(array, list):
        total_elements = 0
        min_element = math.inf
        for element in array:
            for subelement in element:
                if isinstance(subelement, (int, np.uint8)):
                    total_elements += 1
                    if min_element > subelement:
                        min_element = subelement
                elif isinstance(subelement, list):
                    total_elements += len(subelement)
                    for subelement in subelement:
                        if min_element > subelement:
                            min_element = subelement
                else:
                    raise Exception(f"Unsupported storage type: {type(subelement)}")
        assert min_element <= 255, f"value {min_element} does not fit within uint8"
        return total_elements * 8 / (1024 * 1024)

    else:
        raise Exception(f"Unexpected storage type: {type(array)}")


def write_to_file(array, filename: str):
    flattened_array = np.ndarray.flatten(array)
    with open(filename, "w") as f:
        for item in flattened_array:
            f.write(str(item) + "\n")


if __name__ == "__main__":
    test_tensor = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int8)
    storage_size = get_storage(test_tensor)
    print(f"{storage_size = } mB")
