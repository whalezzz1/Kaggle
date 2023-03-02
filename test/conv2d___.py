import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv2d = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=(1, 1), padding=0)

    def forward(self, x):
        print(x.requires_grad)
        x = self.conv2d(x)
        return x


model = Net()
print(model.conv2d.weight.shape)
print(model.conv2d.bias.shape)
