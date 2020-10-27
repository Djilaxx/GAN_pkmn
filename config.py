import torch
import torchvision.transforms as transforms
from easydict import EasyDict as edict 

config = edict()

config.main = edict()
config.main.DATA_PATH = 'data/data_ready/'
config.main.NGPU = 1
config.main.NZ = 100
config.main.IMAGE_SIZE = (64,64)
config.main.SAVE_FREQ = 25 #Checkpoint frequency
config.main.DEVICE = torch.device("cuda:0" if (torch.cuda.is_available() and config.main.NGPU > 0) else "cpu")

config.DCGAN = edict()
config.DCGAN.BATCH_SIZE = 256
config.DCGAN.EPOCHS = 500
config.DCGAN.NC = 3
config.DCGAN.NGF = 64
config.DCGAN.NDF = 64
config.DCGAN.LR = 0.0001
config.DCGAN.BETA1 = 0
config.DCGAN.BETA2 = 0.9
config.DCGAN.TRANSFORMS = transforms.Compose([
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.Resize(config.main.IMAGE_SIZE),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ])

config.WGAN = edict()
config.WGAN.BATCH_SIZE = 16
config.WGAN.EPOCHS = 300
config.WGAN.CRITICS_ITER = 5
config.WGAN.NC = 3
config.WGAN.NGF = 64
config.WGAN.NDF = 64
config.WGAN.LR = 0.00005
config.WGAN.BETA1 = 0
config.WGAN.BETA2 = 0.9
config.WGAN.LAMBDA_GP = 10
config.WGAN.TRANSFORMS = transforms.Compose([
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.Resize(config.main.IMAGE_SIZE),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ])

