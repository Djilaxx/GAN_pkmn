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

def evaluate(model, path, note):
    backbone = importlib.import_module(".model", model)
    Generator = backbone.Generator().to(config.main.DEVICE)

    writer = SummaryWriter(os.path.join('runs/evaluate/' , note))
    filename = re.findall(r'[^\\/]+|[\\/]', path)[-1].rsplit('.', 1)[0]
    noise = torch.randn(64, config.main.NZ, 1, 1, device = config.main.DEVICE)


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
    vutils.save_image(vutils.make_grid(fake, padding=2, normalize=True), "results/" + filename + "/" + "grid" + ".png")
    writer.add_image("fake pokemons evaluation", fake, dataformats="NCHW")
    for i in range(0, len(fake)):
        img = fake[i]
        vutils.save_image(img, "results/" + filename + "/" + str(i) + ".png")


parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, default = "DCGAN", help = "One of DCGAN, WGAN")
parser.add_argument("--path", type = str, help = "Folder path to you checkpoint")
parser.add_argument("--note", type = str, default = "test")

args = parser.parse_args()

if __name__ == "__main__":
    evaluate(
        model=args.model,
        path=args.path,
        note=args.note
    )
