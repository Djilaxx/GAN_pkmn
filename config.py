from easydict import EasyDict as edict 

config = edict()

config.MODEL = edict()
config.MODEL.nc = 3
config.MODEL.nz = 100
config.MODEL.ngf = 64
config.MODEL.ndf = 64
config.MODEL.ngpu = 1

config.TRAIN = edict()
config.TRAIN.num_epochs = 10
config.TRAIN.lr = 0.0002
config.TRAIN.beta1 = 0.5
config.TRAIN.batch_size = 128

config.DATA = edict()
config.DATA.dataroot = 'D:/Documents/GitHub/DCGAN_pkmn/data'
config.DATA.workers = 0
config.DATA.image_size = (64,64)

