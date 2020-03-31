import os
import random
import glob
import re
from PIL import Image
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from model import Generator, Discriminator
from utils import weights_init, label_smoothing, noisy_labelling
from config import config

device = torch.device("cuda:0" if (torch.cuda.is_available() and config.MODEL.ngpu > 0) else "cpu")
####### DATA LOADING #######
def get_data():
    dataset = dset.ImageFolder(root=config.DATA.dataroot,
                            transform=transforms.Compose([
                            transforms.Resize(config.DATA.image_size),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomRotation((0,20)),
                            #transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))

    dataloader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=config.TRAIN.batch_size,
                                         shuffle=True, 
                                         num_workers=config.DATA.workers)
    return dataloader


##### LOADING MODELS ######

netG = Generator(config.MODEL.ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (config.MODEL.ngpu > 1):
    netG = nn.DataParallel(netG, list(range(config.MODEL.ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Create the Discriminator
netD = Discriminator(config.MODEL.ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (config.MODEL.ngpu > 1):
    netD = nn.DataParallel(netD, list(range(config.MODEL.ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)


def train():

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    #Loading the data
    dataloader = get_data()

    #Creating the generator and initializing it's weights
    netG = Generator(config.MODEL.ngpu).to(device)
    netG.apply(weights_init)
    #Creating the discriminator and initializing it's weights
    netD = Discriminator(config.MODEL.ngpu).to(device)
    netD.apply(weights_init)

    # Initialize BCELoss function
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, config.MODEL.nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=config.TRAIN.lr, betas=(config.TRAIN.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=config.TRAIN.lr, betas=(config.TRAIN.beta1, 0.999))

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(config.TRAIN.num_epochs):
    # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0) #Give the first element of the size list which is batch size
            label = torch.full((b_size,), real_label, device=device)
            # We put real label between 0.7 - 1.2 instead of just 1
            label = label_smoothing(label, real=True)
            # We also create 5% of 'fake' labels, values between 0 - 0.2 to fool the discriminator  
            label = noisy_labelling(label, 0.05, (0,0.2), smooth_label=True)

            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, config.MODEL.nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            label = label_smoothing(label,real=False)
            label = noisy_labelling(label, 0.05, (0.7,1.2), smooth_label=True)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            label = label_smoothing(label, real=True)
            label = noisy_labelling(label, 0.05, (0,0.2), smooth_label=True)
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

                # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, config.TRAIN.num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == config.TRAIN.num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            # We also save G and D every few epochs so that later we can test every version and choose the best one
                torch.save({
                'epoch': epoch,
                'model_state_dict': netG.state_dict(),
                'optimizer_state_dict': optimizerG.state_dict(),
                'loss': errG
                }, "checkpoint/checkpointG-" + str(epoch) + '-' + str(errG.item()) + '.pt')
            
                torch.save({
                'epoch': epoch,
                'model_state_dict': netD.state_dict(),
                'optimizer_state_dict': optimizerD.state_dict(),
                'loss': errD
                }, 'checkpoint/checkpointD-' + str(epoch) + '-' + str(errD.item()) + '.pt')
            iters += 1


def evaluate():
    for file in glob.glob("checkpoint/checkpointG*"):
        name = re.findall(r'[^\\/]+|[\\/]', file)[2]
        os.mkdir("results/" + name)
        netG = Generator(config.MODEL.ngpu).to(device)
        cp = torch.load(file)
        netG.load_state_dict(cp["model_state_dict"])
        netG.eval()
        noise = torch.randn(64, config.MODEL.nz, 1, 1, device = device)
        with torch.no_grad:
            fake = netG(noise).detach().cpu()
        for i in range(0, len(fake)):
            img = fake[i]
            vutils.save_image(img, "results/" + name + "/" + name + "_" + str(i) + ".png")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type = str, default = "train", help = "train, evaluate")

    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "evaluate":
        evaluate()
    else:
        raise Exception("Unknown --mode")