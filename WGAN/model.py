import torch
import torch.nn as nn
from config import config

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
            nn.ConvTranspose2d(config.main.NZ, config.WGAN.NGF * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(config.WGAN.NGF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(config.WGAN.NGF * 8, config.WGAN.NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.WGAN.NGF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(config.WGAN.NGF * 4, config.WGAN.NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.WGAN.NGF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(config.WGAN.NGF * 2, config.WGAN.NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.WGAN.NGF),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(config.WGAN.NGF, config.WGAN.NC, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    '''
    Discriminator without BatchNorm for WGAN-GP training 
    '''
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(config.WGAN.NC, config.WGAN.NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(config.WGAN.NDF, config.WGAN.NDF * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(config.WGAN.NDF * 2, config.WGAN.NDF * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(config.WGAN.NDF * 4, config.WGAN.NDF * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(config.WGAN.NDF * 8, 1, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )            #Removing the sigmoid for WGAN, the discriminator send a scalar value as signal not a probability

    def forward(self, input):
        x = self.main(input)
        x = torch.mean(x, dim=(2,3))
        activation = nn.LeakyReLU(0.2, inplace=True)
        output = activation(x)
        return output