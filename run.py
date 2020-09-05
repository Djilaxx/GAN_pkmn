import os, inspect, importlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import gc
import glob

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from datasets.pkmn_ds import POKEMON_DS
from DCGAN.DCGAN_model import DCGAN_Discriminator, DCGAN_Generator
from DCGAN.DCGAN_train_function import DCGAN_trainer
from WGAN.WGAN_model import WGAN_Discriminator, WGAN_Generator
from WGAN.WGAN_train_function import WGAN_trainer
from utils.weights_init import weights_init
from utils.checkpoint import save_checkpoint
from dispatcher import models
from config import config

def run(model = "DCGAN", run_note = ''):
    Generator = models[model]["Generator"]
    Generator.apply(weights_init)
    Discriminator = models[model]["Discriminator"]
    Discriminator.apply(weights_init)
    fix_noise = torch.randn(64, config.main.NZ, 1, 1, device = config.main.DEVICE)
    writer = SummaryWriter(os.path.join('runs/' , run_note))    
    
    image_path = []
    for filename in glob.glob(os.path.join(config.main.DATA_PATH, "*.jpg")):
        image_path.append(filename)

    pkmn_dataset = POKEMON_DS(
        image_path = image_path,
        resize = None,
        transforms = config.main.TRANSFORMS
    )

    pkmn_dataloader = torch.utils.data.DataLoader(
            pkmn_dataset, batch_size=config.main.BATCH_SIZE, shuffle=True, num_workers=6
        )
    
    loss_fct = nn.BCELoss()
    optimizerD = torch.optim.Adam(Discriminator.parameters(), lr = config.main.LR, betas=(config.main.BETA1, config.main.BETA2))
    optimizerG = torch.optim.Adam(Generator.parameters(), lr = config.main.LR, betas=(config.main.BETA1, config.main.BETA2))
    if model == "WGAN":
        trainer = WGAN_trainer(generator=Generator,
                        discriminator=Discriminator,
                        optiD=optimizerD,
                        optiG=optimizerG,
                        loss=loss_fct,
                        device=config.main.DEVICE)
    elif model == "DCGAN":
        trainer = DCGAN_trainer(generator=Generator,
                        discriminator=Discriminator,
                        optiD=optimizerD,
                        optiG=optimizerG,
                        loss=loss_fct,
                        device=config.main.DEVICE)

    for epoch in range(config.main.EPOCHS):
        print(f"Starting epoch number : {epoch}")
        errD, D_x, D_G_z1, errG, D_G_z2 = trainer.training_step(pkmn_dataloader)

        if epoch % config.main.SAVE_FREQ == 0 or epoch == config.dcgan.EPOCHS-1:
            save_checkpoint(generator = Generator,
                            discriminator = Discriminator,
                            optiG = optimizerG,
                            optiD = optimizerD,
                            epoch = epoch,
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
parser.add_argument("--model", type = str, help = "One of DCGAN, WGAN")
parser.add_argument("--run_note", type = str, help = "Add a note on your training run to specify it - Will add the note to checkpoint name and tensorboard folder name")     

args = parser.parse_args()

if __name__ == "__main__":
    run(
        model = args.model,
        run_note=args.run_note
        )



        

