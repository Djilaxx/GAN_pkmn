from training.DCGAN_training import DCGAN_train
from training.WGANGP_training import WGANGP_train
from backbone.DCGAN import DCGAN_Discriminator, DCGAN_Generator
from backbone.WGAN import WGAN_Generator, WGAN_Discriminator
from utils.evaluate import Evaluate
import argparse
    
parser = argparse.ArgumentParser()
parser.add_argument('--backbone', type = str, default = "DCGAN", help = "Choose the backbone model for the GAN : DCGAN, WGAN")
parser.add_argument('--model', type = str, help = "Choose training procedure : one of DCGAN, WGAN-GP")
parser.add_argument('--run_note', type = str, default = '', help = "Add a note on your training run to specify it - Will add the note to checkpoint name and tensorboard folder name")
parser.add_argument('--mode', type = str, default = "train", help = "train, evaluate")
parser.add_argument('--cp', type = str, default = "last", help = "none, last, few, often")
parser.add_argument('--eval_path', type = str, default = '', help = " complete path to checkpoint for evaluation")

args = parser.parse_args()

def main(args):
    if args.mode == "train":
        model = None
        if args.model == "WGAN-GP":
            if args.backbone == "WGAN":
                model = WGANGP_train(WGAN_Generator, WGAN_Discriminator, args.run_note)
                model.train(checkpoint=args.cp)
            else:
                print("Backbone not supported for this model, must be WGAN")
        elif args.model == "DCGAN":
            if args.backbone == "DCGAN":
                model = DCGAN_train(DCGAN_Generator, DCGAN_Discriminator, args.run_note)
                model.train(checkpoint=args.cp)
            elif args.backbone == "WGAN":
                model = DCGAN_train(WGAN_Generator, WGAN_Discriminator, args.run_note)
                model.train(checkpoint=args.cp)
            else: 
                print("Backbone not supported, must be one of : DCGAN, WGAN")
        else: 
            print("Model type not supported")
    elif args.mode == "evaluate":
        if args.model == "DCGAN":
            Eval = Evaluate(DCGAN_Generator, path = args.eval_path)
            Eval.generate()

        elif args.model == "WGAN-GP":
            Eval = Evaluate(WGAN_Generator, path = args.eval_path)
            Eval.generate()
    else:
        print("mode not supported, must be one of : train, evaluate")

    
if __name__ == "__main__":
    main(args)