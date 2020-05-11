import os
import random
import glob
import re
import time as t
from pathlib import Path
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torch import autograd

from backbone.DCGAN import DCGAN_Generator, DCGAN_Discriminator
from utils.dataloader import get_dataloader
from utils.utils import weights_init
from config import config

class WGANGP_train(object):
    '''
    Trainer for the WGAN-GP model

    Attributes : 
        D - Discriminator is a Torch Discriminator model
        G - Generator is a Torch Generator model
        OptimD - Torch optimizer for D (Adam for this model following paper guideline)
        OptimG - Torch optimizer for G (Adam too)
        dataloader : Torch object to load data from a dataset  
    '''
    
    def __init__(self, Generator, Discriminator, run_note = ''):
        print("Training WGAN-GP model.")
        self.device =  torch.device("cuda:0" if (torch.cuda.is_available() and config.DATA.ngpu > 0) else "cpu")
        self.run_note = run_note.replace(" ", "_")
        self.writer = SummaryWriter(os.path.join('runs/', self.run_note))
        self.fix_noise = torch.randn(64, config.DATA.nz, 1, 1, device = self.device)
        self.dataloader = get_dataloader()
        self.critics_iter = 5

        self.G = Generator(config.DATA.ngpu).to(self.device)
        self.G.apply(weights_init)
        self.D = Discriminator(config.DATA.ngpu).to(self.device)
        self.D.apply(weights_init)

        self.optimD = optim.Adam(self.D.parameters(), lr = config.TRAIN.wgan_gp.lr, betas=(config.TRAIN.wgan_gp.beta1, config.TRAIN.wgan_gp.beta2))
        self.optimG = optim.Adam(self.G.parameters(), lr = config.TRAIN.wgan_gp.lr, betas=(config.TRAIN.wgan_gp.beta1, config.TRAIN.wgan_gp.beta2))



    def train(self, checkpoint = "last"):
        self.t_begin = t.time()
        iters = 0

        def save_cp():
            '''
            Save G & D in the checkpoint repository
            '''
            Path("checkpoint/").mkdir(parents=True, exist_ok=True)

            torch.save({
            'epoch': epoch,
            'model_state_dict': self.G.state_dict(),
            'optimizer_state_dict': self.optimG.state_dict(),
            'loss': errG
            }, "checkpoint/checkpointG-" + "WGAN-GP"+ "-" + str(epoch) + '-' + str(round(errG.item(),2)) + '.pt')
            
            torch.save({
            'epoch': epoch,
            'model_state_dict': self.D.state_dict(),
            'optimizer_state_dict': self.optimD.state_dict(),
            'loss': errD
            }, 'checkpoint/checkpointD-' + str(epoch) + '-' + str(round(errD.item(),2)) + '.pt')

        for epoch in range(config.DATA.num_epochs):
            t_epoch = t.time()

            for i, data in enumerate(self.dataloader, 0):

                #----------------------------------------
                # UPDATE D NETWORK
                #----------------------------------------

                # TRAIN ON REAL DATA
                self.D.zero_grad()

                real_data = data[0].to(self.device)
                b_size = real_data.size(0)
                output_real = self.D(real_data)

                noise = torch.randn(b_size, config.DATA.nz, 1, 1, device = self.device)
                fake = self.G(noise)
                output_fake = self.D(fake)

                errD = self.wasserstein_loss_dis(output_fake=output_fake,
                                                 output_real=output_real)
                
                errD_GP = self.compute_gradient_penalty_loss(real_images=real_data, fake_images=fake, gp_scale=config.TRAIN.wgan_gp.lambda_gp)

                errD_total = errD + errD_GP
                errD_total.backward()

                self.optimD.step()

                #-------------------
                # TRAIN GENERATOR
                #-------------------
                if i % self.critics_iter == 0:

                        self.G.zero_grad()
                        
                        output_gen = self.D(fake)
                        errG = self.wasserstein_loss_gen(output_fake=output_gen)
                        errG.backward()
                        
                        self.optimG.step()

                #-------------------------
                # SAVING AND METRICS
                #-------------------------

                # PRINT TRAINING INFO AND SAVE LOSS
                if (iters % 500 == 0) or ((epoch == config.DATA.num_epochs-1) and (i == len(self.dataloader)-1)):
                    with torch.no_grad():
                        fake_grid = self.G(self.fix_noise).detach().cpu()
                    self.writer.add_image("fake pokemons", fake_grid, global_step= epoch * len(self.dataloader) + i, dataformats="NCHW")
                
                # MODEL CHECKPOINT FOR FURTHER TRAINING OR EVALUATION
                last_epoch = (epoch == config.DATA.num_epochs-1)
                last_iter = (i == len(self.dataloader)-1)
                if checkpoint == "last" and last_epoch and last_iter:
                        save_cp()

                elif checkpoint == "few":
                    cp_array = np.array([0.25, 0.5, 0.75])
                    cp_epoch = cp_array*config.DATA.num_epochs
                    if (epoch in cp_epoch.astype(int) and last_iter) or (last_epoch and last_iter):
                        save_cp()

                elif checkpoint == "often":
                    cp_array = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                    cp_epoch = cp_array*config.DATA.num_epochs
                    if (epoch in cp_epoch.astype(int) and last_iter) or (last_epoch and last_iter):
                        save_cp()
                
                else:
                    pass

                iters += 1

            time_epoch = t.time() - t_epoch
            print('[%d/%d][%d/%d]\tLoss_D: %.4f \tTotal_Loss_D: %.4f\tLoss_G: %.4f\tTime: %.4f'
                % (epoch, config.DATA.num_epochs, i, len(self.dataloader),
                    errD.item(), errD_total.item(), errG.item(), time_epoch))

            self.writer.add_scalar("DCGAN Generator Loss", errG.item(), global_step= epoch * len(self.dataloader) + i)
            self.writer.add_scalar("DCGAN Discriminator Loss", errD.item(), global_step= epoch * len(self.dataloader) + i )

        self.t_end = t.time()
        print('Time of training:{}'.format((self.t_end - self.t_begin)))



    def compute_gradient_penalty_loss(self,
                                      real_images,
                                      fake_images,
                                      gp_scale=10.0):
        """
        Computes gradient penalty loss, as based on:
        https://github.com/jalola/improved-wgan-pytorch/blob/master/gan_train.py
        
        Args:
            real_images (Tensor): A batch of real images of shape (N, 3, H, W).
            fake_images (Tensor): A batch of fake images of shape (N, 3, H, W).
            gp_scale (float): Gradient penalty lamda parameter.
        Returns:
            Tensor: Scalar gradient penalty loss.
        """
        # Obtain parameters
        N, _, H, W = real_images.shape
        device = real_images.device

        # Randomly sample some alpha between 0 and 1 for interpolation
        # where alpha is of the same shape for elementwise multiplication.
        alpha = torch.rand(N, 1)
        alpha = alpha.expand(N, int(real_images.nelement() / N)).contiguous()
        alpha = alpha.view(N, 3, H, W)
        alpha = alpha.to(device)

        # Obtain interpolates on line between real/fake images.
        interpolates = alpha * real_images.detach() \
            + ((1 - alpha) * fake_images.detach())
        interpolates = interpolates.to(device)
        interpolates.requires_grad_(True)

        # Get gradients of interpolates
        disc_interpolates = self.D(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates,
                                  inputs=interpolates,
                                  grad_outputs=torch.ones(
                                  disc_interpolates.size()).to(device),
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        # Compute GP loss
        gradient_penalty = (
            (gradients.norm(2, dim=1) - 1)**2).mean() * gp_scale

        return gradient_penalty
        
    def wasserstein_loss_dis(self, output_real, output_fake):
        """
        Computes the wasserstein loss for the discriminator.
        Args:
            output_real (Tensor): Discriminator output logits for real images.
            output_fake (Tensor): Discriminator output logits for fake images.
        Returns:
            Tensor: A scalar tensor loss output.        
        """
        loss = -1.0 * output_real.mean() + output_fake.mean()

        return loss

    
    def wasserstein_loss_gen(self, output_fake):
        """
        Computes the wasserstein loss for generator.
        Args:
            output_fake (Tensor): Discriminator output logits for fake images.
        Returns:
            Tensor: A scalar tensor loss output.
        """
        loss = -output_fake.mean()

        return loss