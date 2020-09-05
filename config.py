import torch
import torchvision.transforms as transforms
from easydict import EasyDict as edict 

config = edict()

config.main = edict()
config.main.DATA_PATH = 'data/data_ready/'
config.main.NGPU = 1
config.main.NZ = 100
config.main.IMAGE_SIZE = (64,64)
config.main.SAVE_FREQ = 50 #Checkpoint frequency
config.main.DEVICE = torch.device("cuda:0" if (torch.cuda.is_available() and config.main.NGPU > 0) else "cpu")
config.main.BATCH_SIZE = 32
config.main.EPOCHS = 300
config.main.LR = 0.0001
config.main.BETA1 = 0
config.main.BETA2 = 0.9
config.main.TRANSFORMS = transforms.Compose([
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.Resize(config.main.IMAGE_SIZE),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ])

config.dcgan = edict()
config.dcgan.BATCH_SIZE = 32
config.dcgan.EPOCHS = 1000
config.dcgan.NC = 3
config.dcgan.NGF = 64
config.dcgan.NDF = 64
config.dcgan.LR = 0.0002
config.dcgan.BETA1 = 0.5
config.dcgan.BETA2 = 0.999
config.dcgan.TRANSFORMS = transforms.Compose([
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.Resize(config.main.IMAGE_SIZE),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ])

config.wgan = edict()
config.wgan.BATCH_SIZE = 32
config.wgan.EPOCHS = 1000
config.wgan.CRITICS_ITER = 5
config.wgan.NC = 3
config.wgan.NGF = 64
config.wgan.NDF = 64
config.wgan.LR = 0.0001
config.wgan.beta1 = 0
config.wgan.beta2 = 0.9
config.wgan.lambda_gp = 10
config.wgan.TRANSFORMS = transforms.Compose([
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.Resize(config.main.IMAGE_SIZE),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ])

