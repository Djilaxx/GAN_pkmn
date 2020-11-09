import torch
import torch.nn as nn
from torch.nn import init, utils
from config import config

#Generator
class Generator(nn.Module):
    '''
    Our Generator is subclassed with the nn.Module
    The base class for all NN in torch
    Adding Dropout to the Generator to create "noise" as recommanded in https://github.com/soumith/ganhacks
    '''
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(config.MAIN.NZ, config.MODEL.SNGAN.NGF * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(config.MODEL.SNGAN.NGF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(config.MODEL.SNGAN.NGF * 8, config.MODEL.SNGAN.NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.MODEL.SNGAN.NGF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(config.MODEL.SNGAN.NGF * 4, config.MODEL.SNGAN.NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.MODEL.SNGAN.NGF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(config.MODEL.SNGAN.NGF * 2, config.MODEL.SNGAN.NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.MODEL.SNGAN.NGF),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(config.MODEL.SNGAN.NGF, config.MODEL.SNGAN.NC, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

#Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            utils.spectral_norm(nn.Conv2d(config.MODEL.SNGAN.NC, config.MODEL.SNGAN.NDF, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            utils.spectral_norm(nn.Conv2d(config.MODEL.SNGAN.NDF, config.MODEL.SNGAN.NDF * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            utils.spectral_norm(nn.Conv2d(config.MODEL.SNGAN.NDF * 2, config.MODEL.SNGAN.NDF * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            utils.spectral_norm(nn.Conv2d(config.MODEL.SNGAN.NDF * 4, config.MODEL.SNGAN.NDF * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            utils.spectral_norm(nn.Conv2d(config.MODEL.SNGAN.NDF * 8, 1, 4, 1, 0, bias=False))
        )

    def forward(self, input):
        return self.main(input)