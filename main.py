from utils.dataset_utils import OriginalDataset


class TensorStorage(dict):
    def __init__(self, checkpoint, original_dataset):
        """

        Parameters
        ----------
        checkpoint : (int) how often to store the reference frame
        original_dataset: (OriginalDataset) the original dataset
        """
        super().__init__()
        self.checkpoint = checkpoint
        self.original_dataset = original_dataset

    def add(self):
        """this function adds the data to the dictionary"""
        idx = len(self)
        if idx % self.checkpoint == 0:
            self[idx] = self.original_dataset[idx]
        else:
            """@todo here we need to perform some encoding and store the encoded tensor"""
            pass

    def get_image(self, idx):
        """Here we need to reconstruct the original image and also verify that it is correct by
        comparing to the original image in the original dataset"""
        assert idx < len(self), (f"Trying to access idx {idx} which is not available in the TensorStorage,"
                                 f"available indices are {self.keys()}")
        if idx % self.checkpoint == 0:
            return self[idx]

    def get_size(self):
        pass


if __name__ == "__main__":
    original_dataset_ = OriginalDataset(data_path="./datasets/droid_100_sample_pictures")
    img_0 = original_dataset_[0]
    tensor_storage = TensorStorage(checkpoint=10,
                                   original_dataset=original_dataset_)
    tensor_storage.add()
    print(tensor_storage)
    img_0 = tensor_storage.get_image(0)
    print(f'{img_0.shape = }')
