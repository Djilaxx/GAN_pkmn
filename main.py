from training.DCGAN_training import DCGAN_train
from training.WGANGP_training import WGANGP_train
from backbone.DCGAN import DCGAN_Discriminator, DCGAN_Generator
import argparse
    
parser = argparse.ArgumentParser()
parser.add_argument('--backbone', type = str, default = "DCGAN", help = "Choose the backbone model for the GAN : DCGAN")
parser.add_argument('--model', type = str, help = "Choose training procedure : one of DCGAN, WGAN-GP")
parser.add_argument('--mode', type = str, default = "train", help = "train, evaluate")
parser.add_argument('--cp', type = str, default = "last", help = "none, last, few, often")
parser.add_argument('--eval_type', type = str, default = "batch", help = "one, batch, full -- one and batch create fake image on last cp, full create fake for every cp")
args = parser.parse_args()

def main(args):
    model = None
    if args.model == "WGAN-GP":
        if args.backbone == "DCGAN":
            model = WGANGP_train(DCGAN_Generator, DCGAN_Discriminator)
        else: 
            print("Backbone not supported")
    elif args.model == "DCGAN":
        if args.backbone == "DCGAN":
            model = DCGAN_train(DCGAN_Generator, DCGAN_Discriminator)
        else: 
            print("Backbone not supported")
    else: 
        print("Model type not supported")
    
    if args.mode == "train":
        model.train(checkpoint = args.cp)
    elif args.mode == "evaluate":
        model.evaluate(type = args.eval_type)
    else:
        raise Exception("Unknown --mode")


if __name__ == "__main__":
    main(args)