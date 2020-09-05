from DCGAN.DCGAN_model import DCGAN_Discriminator, DCGAN_Generator
from DCGAN.DCGAN_train_function import DCGAN_trainer
from WGAN.WGAN_model import WGAN_Discriminator, WGAN_Generator
from WGAN.WGAN_train_function import WGAN_trainer

from config import config

models = {
    "DCGAN" : {
        "Generator" : DCGAN_Generator().to(config.main.DEVICE),
        "Discriminator" : DCGAN_Discriminator().to(config.main.DEVICE)

    },
    "WGAN" : {
        "Generator" : WGAN_Generator().to(config.main.DEVICE),
        "Discriminator" : WGAN_Discriminator().to(config.main.DEVICE)
    }
}