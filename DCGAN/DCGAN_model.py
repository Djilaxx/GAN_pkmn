import torch
import torch.nn as nn
from config import config

class DCGAN_Generator(nn.Module):
    '''
    Our Generator is subclassed with the nn.Module
    The base class for all NN in torch
    Adding Dropout to the Generator to create "noise" as recommanded in https://github.com/soumith/ganhacks
    '''
    def __init__(self):
        super(DCGAN_Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(config.main.NZ, config.DCGAN.NGF * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(config.DCGAN.NGF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(config.DCGAN.NGF * 8, config.DCGAN.NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.DCGAN.NGF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(config.DCGAN.NGF * 4, config.DCGAN.NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.DCGAN.NGF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(config.DCGAN.NGF * 2, config.DCGAN.NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.DCGAN.NGF),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(config.DCGAN.NGF, config.DCGAN.NC, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class DCGAN_Discriminator(nn.Module):
    def __init__(self):
        super(DCGAN_Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(config.DCGAN.NC, config.DCGAN.NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(config.DCGAN.NDF, config.DCGAN.NDF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.DCGAN.NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(config.DCGAN.NDF * 2, config.DCGAN.NDF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.DCGAN.NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(config.DCGAN.NDF * 4, config.DCGAN.NDF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.DCGAN.NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(config.DCGAN.NDF * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)