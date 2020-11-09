import os, inspect, importlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import gc
import glob

import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

from datasets.pkmn_ds import POKEMON_DS
from utils.weights_init import weights_init
from utils.checkpoint import save_checkpoint
from trainer import train_function
from config import config

def run(model, backbone, loss, run_note):
    #Loading Generator and Discriminator and initiating layer's weight
    arch = importlib.import_module(f"models.{model}.{backbone}")
    Generator = arch.Generator().to(config.MAIN.DEVICE)
    Discriminator = arch.Discriminator().to(config.MAIN.DEVICE)
    #Weight init in resnet models is coded in the RESNET.py
    if backbone == "CONVNET": 
        Generator.apply(weights_init)
        Discriminator.apply(weights_init)

    #Creating the tensorboard writer for future logging
    writer = SummaryWriter(os.path.join('runs/' , run_note))
    #Fixed noise to evaluate the Generator every few epoch
    fix_noise = torch.randn(64, config.MAIN.NZ, 1, 1, device = config.MAIN.DEVICE)
    #Creating the dataset
    image_path = []
    for filename in glob.glob(os.path.join(config.MAIN.DATA_PATH, "*.jpg")):
        image_path.append(filename)

    pkmn_dataset = POKEMON_DS(
        image_path = image_path,
        resize = None,
        transforms = config.MAIN.TRANSFORMS
    )

    pkmn_dataloader = torch.utils.data.DataLoader(
            pkmn_dataset, batch_size=config.MAIN.BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True
        )
    
    #Defining loss and optimizer
    loss_module = importlib.import_module(f"loss.{loss}")
    loss_fct = loss_module.Loss_fct()
    optimizerD = torch.optim.Adam(Discriminator.parameters(), lr = config.LOSS[loss].LR, betas=(config.LOSS[loss].BETA1, config.LOSS[loss].BETA2))
    optimizerG = torch.optim.Adam(Generator.parameters(), lr = config.LOSS[loss].LR, betas=(config.LOSS[loss].BETA1, config.LOSS[loss].BETA2))

    #Creating the trainer objects
    trainer = train_function.Trainer(generator=Generator,
                    discriminator=Discriminator,
                    optiD=optimizerD,
                    optiG=optimizerG,
                    loss=loss_fct,
                    device=config.MAIN.DEVICE)

    #Training the model
    print(f"Training start for {config.MAIN.EPOCHS} epochs")
    for epoch in range(config.MAIN.EPOCHS):
        print(f"Starting epoch number : {epoch}")
        errD, D_x, D_G_z1, errG, D_G_z2 = trainer.training_step(pkmn_dataloader)

        #Logging the results and saving the model weights
        if epoch % config.MAIN.SAVE_FREQ == 0 or epoch == config.MAIN.EPOCHS-1:
            save_checkpoint(generator = Generator,
                            discriminator = Discriminator,
                            optiG = optimizerG,
                            optiD = optimizerD,
                            epoch = epoch,
                            loss_fct = loss_fct,
                            errG = errG,
                            errD = errD,
                            run_note = run_note)

            with torch.no_grad():
                fake_grid = Generator(fix_noise).detach().cpu()
                writer.add_image("fake pokemons", fake_grid, global_step = epoch, dataformats="NCHW")
                writer.add_scalar("DCGAN Generator Loss", errG.item(), global_step = epoch)
                writer.add_scalar("DCGAN Discriminator Loss", errD.item(), global_step = epoch)
                writer.add_scalar("DCGAN Real images detection", D_x, global_step = epoch)
                writer.add_scalar("DCGAN Fake images detection", D_G_z1, global_step = epoch)
                writer.add_scalar("DCGAN Fooling power", D_G_z2, global_step = epoch)
        
parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, help = "DCGAN - SNGAN", default = "DCGAN")
parser.add_argument("--backbone", type = str, help = "CONVNET - RESNET", default = "CONVNET")
parser.add_argument("--loss", type = str, help = "BCE - LS-CE - WGAN", default = "BCE")
parser.add_argument("--run_note", type = str, help = "NOTE OF RUN", default = "test")     

args = parser.parse_args()

if __name__ == "__main__":
    run(
        model = args.model,
        backbone = args.backbone,
        loss = args.loss,
        run_note=args.run_note
        )



        

