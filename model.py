
import torch
import torch.nn as nn
from config import config

class Generator(nn.Module):
    '''
    Our Generator is subclassed with the nn.Module
    The base class for all NN in torch
    '''
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = config.DATA.ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(config.DATA.nz, config.MODEL.dcgan.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(config.MODEL.dcgan.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(config.MODEL.dcgan.ngf * 8, config.MODEL.dcgan.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.MODEL.dcgan.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(config.MODEL.dcgan.ngf * 4, config.MODEL.dcgan.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.MODEL.dcgan.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(config.MODEL.dcgan.ngf * 2, config.MODEL.dcgan.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.MODEL.dcgan.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(config.MODEL.dcgan.ngf, config.MODEL.dcgan.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
'''
    def load_G(self, checkpoint, eval = True):
        netG = Generator(config.MODEL.dcgan.ngpu).to(device)
        cp = torch.load(checkpoint)
        netG.load_state_dict(cp["model_state_dict"])
        if eval is True:
            netG.eval()
        return netG
'''

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = config.DATA.ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(config.MODEL.dcgan.nc, config.MODEL.dcgan.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(config.MODEL.dcgan.ndf, config.MODEL.dcgan.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.MODEL.dcgan.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(config.MODEL.dcgan.ndf * 2, config.MODEL.dcgan.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.MODEL.dcgan.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(config.MODEL.dcgan.ndf * 4, config.MODEL.dcgan.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.MODEL.dcgan.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(config.MODEL.dcgan.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
        