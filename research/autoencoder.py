import sys  
sys.path.insert(1, '../')
# from utils.dataset_utils import OriginalDataset
import torch
import math

import os
from utils.dataset_utils import read_img, get_storage
from torchvision.datasets import VisionDataset
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
    
    
def load_model(checkpoint=None, train=True):
    if checkpoint is None:
        model = CNNAutoencoder()
    else:
        assert os.path.exists(checkpoint), f"{checkpoint} does not exist!"
        model = CNNAutoencoder()
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
    if train:
        model.train()
    else:
        model.eval()
    return model


def norm_value(s):
    # return (s + 255.0) / 510.0
    return s / 255

def denorm_value(s):
    # return (s * 510.0) - 255.0
    # print(f"{s = }")
    return torch.tensor(s * 255, dtype=torch.int8)

class OriginalDataset(VisionDataset):
    def __init__(self, data_path: str, color: bool=True):
        super(OriginalDataset, self).__init__()
        self.data_path = data_path
        self.color = color

    def __getitem__(self, idx: int):
        """
        this dataset returns the image corresponding to the index
        """

        if idx >= len(self) or idx < 0:
            # needed for iterator stop condition
            raise IndexError
        # the img_file_path existence check
        img_path = f'{self.data_path}/idx_{idx}.png'
        img2_path = f'{self.data_path}/idx_{idx+1}.png'
        assert os.path.exists(img_path), f"Invalid img_path: {img_path} in {self.data_path}"
        img1 = read_img(img_path, self.color)
        # img2 = read_img(img2_path, self.color)
        img1 = np.array(img1, dtype = np.float32)
        # img2 = np.array(img2, dtype = np.float32)
        # return norm_value(img2 - img1)
        return norm_value(img1)

    def __len__(self) -> int:
        dirpath, dir_names, files = next(os.walk(self.data_path))
        # return len([i for i in files if "resize" not in i]) - 1
        return len([i for i in files if "resize" not in i])

    def __str__(self):
        return f"OriginalDataset({self.data_path})"

    def get_storage_size(self, num_images):
        "returns the total storage size of the dataset"
        total_storage = 0
        for data in [self[i] for i in range(num_images)]:
            total_storage += get_storage(data)
        return total_storage

class CNNAutoencoder(nn.Module):
    def __init__(self):
        super(CNNAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            # nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(True),
            # nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(True),
            # nn.Conv2d(16, 8, kernel_size=7),
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=7, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 3, kernel_size=7),
            # nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=1),
            # nn.ReLU(True),
            # nn.Conv2d(16, 32, kernel_size=7, stride=2, padding=1),
            # nn.ReLU(True),
            # nn.Conv2d(32, 32, kernel_size=11),
            # nn.ReLU(True),
            # nn.Conv2d(32, 16, kernel_size=7),
            # nn.ReLU(True),
            # nn.Conv2d(16, 8, kernel_size=5)
        )
        
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(8, 16, kernel_size=7),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(3, 32, kernel_size=7),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=7, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.ConvTranspose2d(8, 16, kernel_size=5),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(16, 32, kernel_size=7),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(32, 32, kernel_size=11),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(32, 16, kernel_size=7, stride=2, padding=1),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(16, 3, kernel_size=5, stride=2, padding=1),
            # nn.Sigmoid()
        )
        
    def forward(self, x):
        # x = x / 255.0
        x = self.encoder(x)
        x = self.decoder(x)
        # x = x * 255.0
        # x = x.clamp(-255, 255)
        return x
    
    def test(self, x):
        x_hid = self.encoder(x)
        print(f"{x_hid.size()=}")
        x = self.decoder(x_hid)
        # x = x * 255.0
        # x = x.clamp(-255, 255)
        return x, x_hid
    
    def fit(self, original_dataset, checkpoint=None):
        if checkpoint is None:
            model = self
        else:
            assert os.path.exists(checkpoint), f"{checkpoint} does not exist!"
            model = self
            model.load_state_dict(torch.load(checkpoint, weights_only=True))
        model.train()
        num_epochs = 50
        train_loader = torch.utils.data.DataLoader(original_dataset, batch_size=4, shuffle=True)
        model = self
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        lowest_epoch_loss = math.inf
        for epoch in range(num_epochs):
            loss_epoch = 0
            for data in train_loader:
                img = data.permute(0, 3, 1, 2)
                output = model(img)
                # print(output.size(), img.size())
                loss = criterion(output, img)
                loss_epoch += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch+1:>2}/{num_epochs}], Loss: {loss_epoch/len(train_loader):.6f}')
            if loss_epoch < lowest_epoch_loss:
                "we save the checkpoint with the lowest loss"
                torch.save(model.state_dict(), "../checkpoints/lowest_loss_ae.pt")

        print("Finish")
        
if __name__ == "__main__":
    original_dataset = OriginalDataset('./datasets/droid_100_sample_pictures')
    len_ = (original_dataset.__len__())
    print(len_)

    original_dataset[0].shape



    train_loader = torch.utils.data.DataLoader(original_dataset, batch_size=4, shuffle=True)

    model = load_model(checkpoint="./checkpoints/autoencoder_4000.pt")
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    num_epochs = 25000
    for epoch in range(num_epochs):
        PATH = f"./checkpoints/autoencoder_{epoch+5000}.pt"
        loss_epoch = 0
        for data in train_loader:
            img = data.permute(0, 3, 1, 2)
            output = model(img)
            # print(output.size(), img.size())
            loss = criterion(output, img)
            loss_epoch += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 1000 == 0 and epoch >= 1000:
            torch.save(model.state_dict(), PATH)
        
        print(f'Epoch [{epoch+1:>2}/{num_epochs}], Loss: {loss_epoch/len(train_loader):.6f}')

    print("Finish")

    import matplotlib.pyplot as plt
    test_image = next(iter(train_loader))[:1]
    with torch.no_grad():
        reconstructed = model.test(test_image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    print(denorm_value(test_image[0][0][0]))
    print(denorm_value(reconstructed[0][0][0]))
    fig, axes = plt.subplots(1, 2, figsize=(15, 3))
    axes[0].imshow(denorm_value(test_image.squeeze()))
    axes[0].axis('off')
    axes[1].imshow(denorm_value(reconstructed.squeeze()))
    axes[1].axis('off')