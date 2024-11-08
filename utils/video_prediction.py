import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class FrameDataset(Dataset):
    def __init__(self, frame, displacement=None):
        self.frame = frame.flatten(0, 1)
        self.dim0 = frame.size()[0]
        self.dim1 = frame.size()[1]
        if displacement is not None:
            self.displacement = displacement.flatten(0, 1)
        else:
            self.displacement = None

    def __len__(self):
        return self.dim0 * self.dim1

    def __getitem__(self, idx):
        label = self.frame[idx]
        coord = [idx // self.dim1, idx % self.dim1]
        if self.displacement is not None:
            ad = self.displacement[idx]
            return torch.tensor(coord, dtype=torch.float32), torch.tensor(ad, dtype=torch.float32), label
        else:
            return torch.tensor(coord, dtype=torch.float32), label


class SIREN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SIREN, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.sin(self.hidden(x))
        return self.output(x)


class video_predict_model(nn.Module):
    def __init__(self, i_model, flow_model, res_model):
        super(video_predict_model, self).__init__()
        self.i_model = i_model
        self.flow_model = flow_model
        self.res_model = res_model

    def forward(self, x, displacement):
        # print(x.size())
        # print(self.displacement.size())
        h_out = self.flow_model(x)
        f_out = self.i_model(x + h_out + displacement)
        r_out = self.res_model(x)
        return f_out + r_out


def quantize_weights(weights, scale, bit_width=8):
    max_val = torch.max(weights).item()
    min_val = torch.min(weights).item()
    quantized = torch.round((weights - min_val) / (max_val - min_val) * (2 ** bit_width - 1))
    quantized = quantized * scale
    return quantized


def train_frame_model(model, target_frame, num_epochs=1000, lr=0.0001, displace=None):
    if displace == None:
        mydataset = FrameDataset(target_frame)
    else:
        mydataset = FrameDataset(target_frame, displacement=displace)
    train_dataloader = DataLoader(mydataset, batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(num_epochs):
        for i, data in enumerate(train_dataloader):
            if displace == None:
                coordinates, target_pixels = data
                optimizer.zero_grad()
                output = model(coordinates)
            else:
                coordinates, a_displace, target_pixels = data
                optimizer.zero_grad()
                output = model(coordinates, a_displace)
            loss = loss_fn(output, target_pixels)
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    return model


def train_ipf(video_frames):
    scale = 0.1
    bit_width = 8

    for f_idx in range(0, len(video_frames)):
        if f_idx == 0:
            i_frame = video_frames[0]
            i_frame_model = SIREN(input_dim=2, hidden_dim=32, output_dim=3)

            print("Training I-frame model...")
            i_frame_model = train_frame_model(i_frame_model, i_frame, num_epochs=1000)

            print("Quantizing and Freezing I-frame model...")
            for param in i_frame_model.parameters():
                param.data = quantize_weights(param.data, scale, bit_width)
                param.requires_grad = False

            displacement = torch.empty(i_frame.size()[0], i_frame.size()[1], 2)
            print(f"{displacement.size() = }")
        else:
            break
            flow_model = SIREN(input_dim=2, hidden_dim=16, output_dim=2)
            residual_model = SIREN(input_dim=2, hidden_dim=16, output_dim=3)
            v_model = video_predict_model(i_frame_model, flow_model, residual_model)
            p_frame = video_frames[f_idx]
            v_model = train_frame_model(v_model, p_frame, num_epochs=500, displace=displacement)
            displacement += v_model.flow_model(torch.zeros_like(displacement))
    return i_frame_model
