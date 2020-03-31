#%%
from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#%%
class Image_Process():
    '''
    Processing all image in data directory to reshape them into correct size
    Put them all in the correct format (.jpg + RBG)
    And creating the DataGenerator process
    '''
    def __init__(self,width,height)
        self.width = width
        self.height = height
        self.img_size = (width,height)


    def to_jpg():

    def resize_img():

import Image 
im = Image.open("data/201-e.png")
im.save("data/test.jpg", "JPEG")



for filename in glob.glob('data/'):
    if ".png" not in filename:
        continue
    else:
        im = Image.open(filename)
        im.save()
    im = Image.open("Ba_b_do8mag_c6_big.png")
    gb_im = im.convert('RGB')
    gb_im.save('colors.jpg')


im = Image.open("Ba_b_do8mag_c6_big.png")
rgb_im = im.convert('RGB')
rgb_im.save('colors.jpg')
image_list = []
for filename in glob.glob('yourpath/*.'): #assuming gif
    im=Image.open(filename)
    image_list.append(im)

#%%
img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20)