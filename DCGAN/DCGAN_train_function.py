import os, random, re
from pathlib import Path
import numpy as np 
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.utils.tensorboard import SummaryWriter

from utils.average_meter import AverageMeter
from utils.weights_init import weights_init 
from utils.label_smoothing import label_smoothing 
from utils.noisy_labelling import noisy_labelling
from datasets import pkmn_ds
from config import config

class DCGAN_trainer:
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
        for _, data in enumerate(tk0):
            
            # DISCRIMINATOR STEP ON REAL DATA
            real_images = data["images"]
            real_images = real_images.to(self.device)
            #b_size = real_images.size(0)
            label = torch.ones(config.DCGAN.BATCH_SIZE, dtype=torch.float32, device=self.device)

            #label = torch.full((b_size,), 1, dtype=torch.float32, device=self.device)
            label = label_smoothing(label, device = self.device, real=True)
            label = noisy_labelling(label, 0.05, (0,0.2), smooth_label=True)
            self.D.zero_grad()

            output = self.D(real_images).view(-1)
            errD_real = self.loss(output, label)

            errD_real.backward()
            D_x = output.mean().item()
            
            # DISCRIMINATOR STEP ON FAKE DATA
            #noise = torch.randn(b_size, config.main.NZ, 1, 1, device = self.device)
            noise = torch.randn(config.DCGAN.BATCH_SIZE, config.main.NZ, 1, 1, device = self.device)

            fake_images = self.G(noise)
            label.fill_(0)
            label = label_smoothing(label, device = self.device, real = False)
            label = noisy_labelling(label, 0.05, (0.7, 1.2), smooth_label=True)

            output = self.D(fake_images.detach()).view(-1)
            errD_fake = self.loss(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            self.optiD.step()

            # GENERATOR STEP
            self.G.zero_grad()
            label.fill_(1)

            output = self.D(fake_images).view(-1)
            errG = self.loss(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()

            self.optiG.step()

            D_losses.update(errD.item(), config.DCGAN.BATCH_SIZE)
            G_losses.update(errG.item(), config.DCGAN.BATCH_SIZE)
            tk0.set_postfix(D_loss= D_losses.avg, G_loss = G_losses.avg)
        print(f"Discriminator Loss = {D_losses.avg}, Prediction on Real data = {D_x} and fake data = {D_G_z1}")
        print(f"Generator Loss = {G_losses.avg}, and fooling power = {D_G_z2}")
        return errD, D_x, D_G_z1, errG, D_G_z2