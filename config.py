import torch
import torchvision.transforms as transforms
from easydict import EasyDict as edict 

config = edict()

# MAIN CONFIG
config.MAIN = edict()
config.MAIN.DATA_PATH = 'data/data_ready/'
config.MAIN.NGPU = 1
config.MAIN.NZ = 100
config.MAIN.IMAGE_SIZE = (64,64)
config.MAIN.SAVE_FREQ = 25 #Checkpoint frequency
config.MAIN.DEVICE = torch.device("cuda:0" if (torch.cuda.is_available() and config.MAIN.NGPU > 0) else "cpu")
config.MAIN.BATCH_SIZE = 128
config.MAIN.EPOCHS = 1000
config.MAIN.TRANSFORMS = transforms.Compose([
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.Resize(config.MAIN.IMAGE_SIZE),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ])

# MODEL CONFIG
config.MODEL = edict()
config.MODEL.DCGAN = edict()
config.MODEL.DCGAN.NC = 3
config.MODEL.DCGAN.NGF = 64
config.MODEL.DCGAN.NDF = 64

config.MODEL.SNGAN = edict()
config.MODEL.SNGAN.NC = 3
config.MODEL.SNGAN.NGF = 64
config.MODEL.SNGAN.NDF = 64

# LOSS CONFIG
config.LOSS = edict()
config.LOSS.BCE = edict()
config.LOSS.BCE.LR = 0.0001
config.LOSS.BCE.BETA1 = 0
config.LOSS.BCE.BETA2 = 0.9

config.LOSS.LB_CE = edict()
config.LOSS.LB_CE.LR = 0.0001
config.LOSS.LB_CE.BETA1 = 0
config.LOSS.LB_CE.BETA2 = 0.9

config.LOSS.WGAN = edict()
config.LOSS.WGAN.LR = 0.00005
config.LOSS.WGAN.BETA1 = 0
config.LOSS.WGAN.BETA2 = 0.9
config.LOSS.WGAN.LAMBDA_GP = 10
config.LOSS.WGAN.CRITICS_ITER = 5
