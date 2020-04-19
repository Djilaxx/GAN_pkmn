from easydict import EasyDict as edict 

config = edict()

config.MODEL = edict()
config.MODEL.dcgan = edict()
config.MODEL.dcgan.nc = 3
config.MODEL.dcgan.nz = 100
config.MODEL.dcgan.ngf = 64
config.MODEL.dcgan.ndf = 64
config.MODEL.dcgan.ngpu = 1

config.TRAIN = edict()
config.TRAIN.dcgan = edict()
config.TRAIN.dcgan.num_epochs = 10
config.TRAIN.dcgan.lr = 0.0002
config.TRAIN.dcgan.beta1 = 0.5
config.TRAIN.dcgan.beta2 = 0.999
config.TRAIN.dcgan.batch_size = 128

config.TRAIN.wgan_gp = edict()
config.TRAIN.wgan_gp.num_epochs = 10
config.TRAIN.wgan_gp.lambda_gp = 10
config.TRAIN.wgan_gp.lr = 0.0001
config.TRAIN.wgan_gp.beta1 = 0
config.TRAIN.wgan_gp.beta2 = 0.9
config.TRAIN.wgan_gp.batch_size = 128

config.DATA = edict()
config.DATA.dataroot = 'D:/Documents/GitHub/DCGAN_pkmn/data/'
config.DATA.workers = 0
config.DATA.image_size = (64,64)

