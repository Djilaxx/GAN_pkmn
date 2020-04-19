import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.DCGAN import Generator, Discriminator
from utils import weights_init, label_smoothing, noisy_labelling, get_dataloader
from config import config

class DCGAN_train(object):
    def __init__(self):
        print("Training DCGAN model.")
        self.device =  torch.device("cuda:0" if (torch.cuda.is_available() and config.MODEL.ngpu > 0) else "cpu")
        self.writer = SummaryWriter('runs/GAN_pokemon')
        self.fix_noise = torch.randn(64, config.MODEL.dcgan.nz, 1, 1, device = self.device)
        self.G = Generator(config.MODEL.ngpu).to(self.device)
        self.G.apply(weights_init)
        self.D = Discriminator(config.MODEL.ngpu).to(self.device)
        self.D.apply(weights_init)

        self.loss = nn.BCELoss()

        self.optimD = optim.Adam(self.D.parameters(), lr = config.TRAIN.dcgan.lr, betas=(config.TRAIN.dcgan.beta1,config.TRAIN.dcgan.beta2))
        self.optimG = optim.Adam(self.G.parameters(), lr = config.TRAIN.dcgan.lr, betas=(config.TRAIN.dcgan.beta1,config.TRAIN.dcgan.beta2))

    def train(self, dataloader):
        real_label = 1
        fake_label = 0
        for epoch in range(config.TRAIN.dcgan.num_epochs):
            for i, data in enumerate(dataloader, 0):

                ###  TRAIN DISCRIMINATOR ON REAL DATA ###
                self.D.zero_grad()
                real_data = data[0].to(self.device)
                label = torch.full((config.TRAIN.dcgan.batch_size,), real_label, device=self.device)
                label = label_smoothing(label, real = True)
                label = noisy_labelling(label, 0.05, (0,0.2), smooth_label=True)
                
                output = self.D(real_data).view(-1)
                errD_real = self.loss(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                ### TRAIN DISCRIMINATOR ON FAKE DATA ###
                noise = torch.randn(config.TRAIN.dcgan.batch_size, config.MODEL.dcgan.nz, 1, 1, device = self.device)
                
                fake = self.G(noise)
                label.fill_(fake_label)
                label = label_smoothing(label, real = False)
                label = noisy_labelling(label, 0.05, (0.7, 1.2), smooth_label=True)

                output = self.D(fake.detach()).view(-1)
                errD_fake = self.loss(output,label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake

                # TAKE AN OPTIMIZER STEP FOR DISCRIMINATOR
                self.optimD.step()

                ### TRAIN GENERATOR

                self.G.zero_grad()
                label.fill_(real_label)
                label = label_smoothing(label, real = True)
                label = noisy_labelling(label, 0.05, (0, 0.2), smooth_label= True)

                output = self.D(fake).view(-1)
                errG = self.loss(output,label)
                errG.backward()
                D_G_z2 = output.mean().item()
                
                self.optimG.step()

                if i % 25 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, config.TRAIN.dcgan.num_epochs, i, len(dataloader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                
                
