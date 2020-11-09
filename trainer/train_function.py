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

class Trainer:
    def __init__(self, generator, discriminator, optiD, optiG, loss, device):
        self.G = generator
        self.D = discriminator
        self.optiD = optiD
        self.optiG = optiG
        self.loss = loss
        self.device = device

    def dis_step(self, data):
        # REAL DATA
        real_images = data["images"]
        real_images = real_images.to(self.device)
        # FAKE DATA
        noise = torch.randn(config.MAIN.BATCH_SIZE, config.MAIN.NZ, 1, 1, device = self.device)
        # LABELS
        label_real = torch.ones(config.MAIN.BATCH_SIZE, dtype=torch.float32, device=self.device)
        label_fake = torch.zeros(config.MAIN.BATCH_SIZE, dtype=torch.float32, device=self.device) 
        # GRADIENTS TO ZERO
        self.D.zero_grad()
        # D OUTPUT ON REAL DATA
        output_real = self.D(real_images).view(-1)
        # D OUTPUT ON FAKE DATA
        fake = self.G(noise)
        output_fake = self.D(fake.detach()).view(-1)
        # DIS LOSS
        errD = self.loss.dis_loss(output_fake=output_fake, output_real=output_real, label_real=label_real, label_fake = label_fake)
        # WE ADD GRADIENT PENALTY IF WGAN LOSS
        if self.loss.name == "WGAN_loss":
            errD_GP = self.loss.compute_gradient_penalty_loss(real_images=real_images, fake_images=fake, discriminator = self.D, gp_scale=config.LOSS.WGAN.LAMBDA_GP)
            errD = errD + errD_GP
        # GRADIENT BACKPROP & OPTIMIZER STEP
        errD.backward()
        self.optiD.step()
        # COMPUTING PROBS
        D_x, D_G_z1 = self.loss.compute_probs(output_real=output_real, output_fake=output_fake)
        return errD, D_x, D_G_z1, fake

    def gen_step(self, fake_images):
        # LABEL FOR GENERATOR
        label_gen = torch.ones(config.MAIN.BATCH_SIZE, dtype=torch.float32, device=self.device) 
        # ZERO GRAD
        self.G.zero_grad()
        # OUTPUT OF DISCRIMINATOR AFTER TRAINING STEP
        output_gen = self.D(fake_images).view(-1)
        # LOSS WITH LABEL AS REAL
        errG = self.loss.gen_loss(output_gen=output_gen, label_gen=label_gen)
        errG.backward()
        self.optiG.step()

        # COMPUTING PROBS 
        D_G_z2 = torch.sigmoid(output_gen).mean().item()
        return errG, D_G_z2

    def training_step(self, dataloader):
        self.D.train()
        self.G.train()
        D_losses = AverageMeter()
        G_losses = AverageMeter()
        
        tk0 = tqdm(dataloader, total=len(dataloader))
        for i, data in enumerate(tk0):
            
            ######################
            # DISCRIMINATOR STEP #
            ######################
            errD, D_x, D_G_z1, fake_images = self.dis_step(data=data)

            ##################
            # GENERATOR STEP #
            ##################
            if (self.loss.name == "WGAN_loss") & (i % config.LOSS.WGAN.CRITICS_ITER != 0):
                pass
            else:
                errG, D_G_z2 = self.gen_step(fake_images)

            D_losses.update(errD.item(), config.MAIN.BATCH_SIZE)
            G_losses.update(errG.item(), config.MAIN.BATCH_SIZE)
            tk0.set_postfix(D_loss= D_losses.avg, G_loss = G_losses.avg)
        print(f"Discriminator Loss = {D_losses.avg}, Prediction on Real data = {D_x} and fake data = {D_G_z1}")
        print(f"Generator Loss = {G_losses.avg}, and fooling power = {D_G_z2}")
        return errD, D_x, D_G_z1, errG, D_G_z2