import os, random, re
from pathlib import Path
import numpy as np 
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
from torch import autograd

from utils.average_meter import AverageMeter
from utils.weights_init import weights_init 
from utils.label_smoothing import label_smoothing 
from utils.noisy_labelling import noisy_labelling
from datasets import pkmn_ds
from config import config

class WGAN_trainer:
    def __init__(self, generator, discriminator, optiD, optiG, loss, device):
        self.G = generator
        self.D = discriminator
        self.optiD = optiD
        self.optiG = optiG
        self.loss = loss
        self.device = device

    def training_step(self, dataloader):
        self.D.train()
        self.G.train()
        D_losses = AverageMeter()
        G_losses = AverageMeter()
        
        tk0 = tqdm(dataloader, total=len(dataloader))
        for i, data in enumerate(tk0):
            
            
            # DISCRIMINATOR STEP ON REAL DATA
            real_images = data["images"]
            real_images = real_images.to(self.device)
            b_size = real_images.size(0)

            self.D.zero_grad()

            output_real = self.D(real_images).view(-1)

            noise = torch.randn(b_size, config.main.NZ, 1, 1, device = self.device)
            fake = self.G(noise)
            output_fake = self.D(fake.detach()).view(-1)

            errD = self.wasserstein_loss_dis(output_fake=output_fake,
                                            output_real=output_real)
            
            errD_GP = self.compute_gradient_penalty_loss(real_images=real_data, fake_images=fake, gp_scale=config.WGAN.LAMBDA_GP)
            errD_total = errD + errD_GP
            errD_total.backward()
            self.optiD.step()
            D_x, D_G_z1 = self.compute_probs(output_real=output_real,
                                            output_fake=output_fake)


            if i % config.WGAN.CRITICS_ITER == 0:
                self.G.zero_grad()
                output_gen = self.D(fake).view(-1)
                errG = self.wasserstein_loss_gen(output_fake=output_gen)
                errG.backward()
                D_G_z2 = torch.sigmoid(output_gen).mean().item()
                self.optiG.step()

            D_losses.update(errD_total.item(), b_size)
            G_losses.update(errG.item(), b_size)
            tk0.set_postfix(D_loss= D_losses.avg, G_loss = G_losses.avg)
        print(f"Discriminator Loss = {D_losses.avg}, Prediction on Real data = {D_x} and fake data = {D_G_z1}")
        print(f"Generator Loss = {G_losses.avg}, and fooling power = {D_G_z2}")
        return errD_total, D_x, D_G_z1, errG, D_G_z2

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
    
    def compute_probs(self, output_real, output_fake):
        """
        Computes probabilities from real/fake images logits.
        Args:
            output_real (Tensor): A batch of output logits of shape (N, 1) from real images.
            output_fake (Tensor): A batch of output logits of shape (N, 1) from fake images.
        Returns:
            tuple: Average probabilities of real/fake image considered as real for the batch.
        """
        D_x = torch.sigmoid(output_real).mean().item()
        D_Gz = torch.sigmoid(output_fake).mean().item()
        return D_x, D_Gz
        
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