import torch
from torch import nn
import torch.nn.functional as F


__all__ = ['Input_conv', 'Net']

class Input_conv(nn.Module):
    def __init__(self, in_channels, out_channels, ks_x=25, ks_y=3):
        super().__init__()
        pad_x = int((ks_x - 1) / 2)
        pad_y = int((ks_y - 1) / 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(ks_y,ks_x),padding=(pad_y, pad_x))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d((1, 100))
        self.flat = nn.Flatten(2, -1)

    def forward(self, x1, x2):
        out1 = self.conv1(x1)
        out1 = self.bn1(out1)
        out1 = self.pool(out1)
        out1 = self.flat(out1)
        out1 = torch.unsqueeze(out1, -1)

        out2 = self.conv1(x2)
        out2 = self.bn1(out2)
        out2 = self.pool(out2)
        out2 = self.flat(out2)
        out2 = torch.unsqueeze(out2, -2)

        out = torch.matmul(out1, out2)

        return out

class Net(nn.Module):
    def __init__(self, n_classes, ks=5):
        super().__init__()
        padding = int((ks - 1) / 2)
        self.in_conv = Input_conv(in_channels=1, out_channels=3)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=padding)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=padding)
        self.fc1 = nn.Linear(16 * 256 * 256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        out = self.in_conv(x[:,:,0:32,:], x[:,:,32:64,:])
        #print("Network: ", out.shape)
        x = self.pool(F.relu(self.conv1(out)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
