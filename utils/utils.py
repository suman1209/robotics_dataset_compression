import math
import numpy as np
import cv2
import os

def get_storage(array: np.array) -> float:
    """
    Returns the storage size in (mB) of an array
    Parameters
    ----------
    array : (np.array) the array whose storage size is to be returned

    Returns
    -------

    """
    if isinstance(array, np.ndarray):
        size = {"int8": 8, "uint8": 8, "int64": 64, "float64": 64}
        dtype = str(array.dtype)
        assert array.ndim == 3 or array.ndim == 2, f"Received array of shape: {array.shape}"
        if array.ndim == 3:
            total_elements = array.shape[0] * array.shape[1] * array.shape[2]
        else:
            total_elements = array.shape[0] * array.shape[1]
        if dtype not in size.keys():
            raise Exception(f"Unsupported storage type: {dtype}")
        return size[dtype] * total_elements / (1024 * 1024)

    elif isinstance(array, list):
        total_elements = 0
        min_element = math.inf
        for element in array:
            for subelement in element:
                if isinstance(subelement, (int, np.uint8, np.float32)):
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


def convert_images_to_grayscale(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):          
            img = cv2.imread(file_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(f"{gray_img.shape = }")
            assert False
            output_file_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_file_path, gray_img)

            print(f"Converted {filename} to grayscale and saved to {output_file_path}")


def write_to_file(array, filename: str):
    flattened_array = np.ndarray.flatten(array)
    with open(filename, "w") as f:
        for item in flattened_array:
            f.write(str(item) + "\n")


if __name__ == "__main__":
    # test_tensor = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int8)
    # storage_size = get_storage(test_tensor)
    # print(f"{storage_size = } mB")
    input_folder = '../datasets/droid_100_sample_pictures'
    output_folder = '../datasets/grayscale_droid_100_sample_pictures_test'

    convert_images_to_grayscale(input_folder, output_folder)