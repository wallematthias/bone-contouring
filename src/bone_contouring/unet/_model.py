"""Nathan Neeteson's UNet, reduced to the published inference architecture.

Adapted from models/UNet.py in this GPL-3.0-only repository. Module/key
names and layer ordering are unchanged for strict state-dict compatibility.
"""
import torch
from torch import nn


class Layer(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(inplace=True),
            nn.GroupNorm(outputs // 16, outputs), nn.Dropout2d(.1),
            nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU(inplace=True),
            nn.GroupNorm(outputs // 16, outputs), nn.Dropout2d(.1),
        )

    def forward(self, x):
        return self.layer(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        filters = (32, 64, 128, 256)
        self.layer_down = nn.ModuleList([Layer(5, filters[0])])
        self.down = nn.ModuleList()
        self.layer_up = nn.ModuleList()
        self.up = nn.ModuleList()
        for previous, current in zip(filters, filters[1:]):
            self.down.append(nn.MaxPool2d(2))
            self.layer_down.append(Layer(previous, current))
            self.up.append(nn.ConvTranspose2d(current, previous, 2, stride=2))
            self.layer_up.append(Layer(2 * previous, previous))
        self.map_to_output = nn.Conv2d(filters[0], 2, 1)

    def forward(self, x):
        skip = [self.layer_down[0](x)]
        for layer, down in zip(self.layer_down[1:], self.down):
            skip.append(layer(down(skip[-1])))
        x = skip.pop()
        for layer, up in zip(reversed(self.layer_up), reversed(self.up)):
            x = layer(torch.cat([skip.pop(), up(x)], dim=1))
        return self.map_to_output(x)
