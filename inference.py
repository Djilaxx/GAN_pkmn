import glob, os, re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import importlib
from pathlib import Path
from datetime import datetime

import torch
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

from config import config

def evaluate(model, backbone, path, note):
    backbone = importlib.import_module(f"models.{model}.{backbone}")
    Generator = backbone.Generator().to(config.MAIN.DEVICE)

    writer = SummaryWriter(os.path.join('runs/evaluate/', note))
    filename = re.findall(r'[^\\/]+|[\\/]', path)[-1].rsplit('.', 1)[0]
    noise = torch.randn(64, config.MAIN.NZ, 1, 1, device = config.MAIN.DEVICE)

    if not os.path.exists(path):
        print("Path to checkpoint does not exists")

    print(f"Creating 64 fake pokemons with model : {filename}")
    #--------------------
    # LOAD CHECKPOINT 
    #--------------------
    cp = torch.load(path)
    Generator.load_state_dict(cp["model_state_dict"])
    Generator.eval()
    #--------------------
    # CREATE FAKE IMAGES
    #--------------------
    with torch.no_grad():
        fake = Generator(noise).detach()
    #--------------------
    # SAVE FAKE IMAGES
    #--------------------
    Path("results/").mkdir(parents=True, exist_ok=True)
    Path(os.path.join("results/", filename)).mkdir(parents=True, exist_ok=True)
    vutils.save_image(vutils.make_grid(fake, padding=2, normalize=True), f"results/{filename}/{grid}.png")
    writer.add_image("fake pokemons evaluation", fake, dataformats="NCHW")
    for i in range(0, len(fake)):
        img = fake[i]
        vutils.save_image(img, f"results/{filename}/{i}.png")


parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, default = "DCGAN", help = "DCGAN - SNGAN")
parser.add_argument("--backbone", type = str, default = "CONVNET", help = "CONVNET - RESNET")
parser.add_argument("--path", type = str, help = "PATH TO CHECKPOINT")
parser.add_argument("--note", type = str, default = "TEST", help = "NAME OF EVAL")

args = parser.parse_args()

if __name__ == "__main__":
    evaluate(
        model=args.model,
        structure=args.backbone,
        path=args.path,
        note=args.note
    )
