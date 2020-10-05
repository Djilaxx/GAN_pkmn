import glob, os, re
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

from config import config
from dispatcher import models

class Evaluate(object):
    def __init__(self, model, path = ''):
        self.device = config.main.DEVICE
        self.G = models[model]["Generator"]
        self.writer = SummaryWriter('runs/evaluate')
        self.path = path
        self.filename = re.findall(r'[^\\/]+|[\\/]', self.path)[-1].rsplit('.', 1)[0]
        self.noise = torch.randn(64, config.main.NZ, 1, 1, device = self.device)

    def generate(self):
        if not os.path.exists(self.path):
            print("Path to checkpoint does not exists")
        #--------------------
        # LOAD CHECKPOINT 
        #--------------------
        cp = torch.load(self.path)
        self.G.load_state_dict(cp["model_state_dict"])
        self.G.eval()

        #--------------------
        # CREATE FAKE IMAGES
        #--------------------
        with torch.no_grad():
            fake = self.G(self.noise).detach()
        
        #--------------------
        # SAVE FAKE IMAGES
        #--------------------
        Path("results/").mkdir(parents=True, exist_ok=True)
        vutils.save_image(vutils.make_grid(fake, padding=2, normalize=True), "results/" + self.filename + "_grid" + ".png")
        self.writer.add_image("fake pokemons", fake, dataformats="NCHW")
        for i in range(0, len(fake)):
            img = fake[i]
            vutils.save_image(img, "results/" + self.filename + "_" + str(i) + ".png")

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, help = "One of DCGAN, WGAN")
parser.add_argument("--eval_path", type = str, help = "Folder path to you checkpoint")     

args = parser.parse_args()

if __name__ == "__main__":
    eval = Evaluate(
        args.model,
        args.eval_path
    )
    eval.generate()
