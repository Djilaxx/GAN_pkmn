# ResNet generator and discriminator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import utils
import numpy as np

from config import config

class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)

class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                utils.spectral_norm(self.conv1),
                nn.ReLU(),
                utils.spectral_norm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                utils.spectral_norm(self.conv1),
                nn.ReLU(),
                utils.spectral_norm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )
        self.bypass = nn.Sequential()
        if stride != 1:

            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                utils.spectral_norm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
            # if in_channels == out_channels:
            #     self.bypass = nn.AvgPool2d(2, stride=stride, padding=0)
            # else:
            #     self.bypass = nn.Sequential(
            #         SpectralNorm(nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)),
            #         nn.AvgPool2d(2, stride=stride, padding=0)
            #     )


    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            utils.spectral_norm(self.conv1),
            nn.ReLU(),
            utils.spectral_norm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            utils.spectral_norm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense = nn.Linear(config.MAIN.NZ, 4 * 4 * config.MODEL.SNGAN.NGF)
        self.final = nn.Conv2d(config.MODEL.SNGAN.NGF, config.MODEL.SNGAN.NC, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)

        self.model = nn.Sequential(
            ResBlockGenerator(config.MODEL.SNGAN.NGF, config.MODEL.SNGAN.NGF, stride=2),
            ResBlockGenerator(config.MODEL.SNGAN.NGF, config.MODEL.SNGAN.NGF, stride=2),
            ResBlockGenerator(config.MODEL.SNGAN.NGF, config.MODEL.SNGAN.NGF, stride=2),
            ResBlockGenerator(config.MODEL.SNGAN.NGF, config.MODEL.SNGAN.NGF, stride=2),
            nn.BatchNorm2d(config.MODEL.SNGAN.NGF),
            nn.ReLU(),
            self.final,
            nn.Tanh())

    def forward(self, input):
        input = torch.reshape(input, (-1, config.MAIN.NZ))
        return self.model(self.dense(input).view(-1, config.MODEL.SNGAN.NGF, 4, 4))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
                FirstResBlockDiscriminator(config.MODEL.SNGAN.NC, config.MODEL.SNGAN.NDF, stride=2),
                ResBlockDiscriminator(config.MODEL.SNGAN.NDF, config.MODEL.SNGAN.NDF, stride=2),
                ResBlockDiscriminator(config.MODEL.SNGAN.NDF, config.MODEL.SNGAN.NDF),
                ResBlockDiscriminator(config.MODEL.SNGAN.NDF, config.MODEL.SNGAN.NDF),
                nn.ReLU(),
                nn.AvgPool2d(8),
            )
            
        self.fc = nn.Linear(config.MODEL.SNGAN.NDF, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        self.fc = utils.spectral_norm(self.fc)

    def forward(self, input):
        return self.fc(self.model(input).view(-1,config.MODEL.SNGAN.NDF))