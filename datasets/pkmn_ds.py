import os, glob
import torch
import numpy as np 

from PIL import Image
from PIL import ImageFile

from config import config


class POKEMON_DS:
    def __init__(self, image_path, resize, transforms=None):
        self.image_path = image_path
        self.resize = resize
        self.transforms = transforms
    
    def __getitem__(self, item):
        image = Image.open(self.image_path[item])
        if self.resize is not None:
            image.resize((self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )

        if self.transforms:
            image = self.transforms(image)
        
        return {
            "images" : image
        }

    def __len__(self):
        return(len(self.image_path))

