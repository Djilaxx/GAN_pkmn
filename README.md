# GAN_pokemon

I will use this project to teach myself to create GAN's and train them.
https://github.com/jacobgil/keras-dcgan/blob/master/dcgan.py
https://towardsdatascience.com/implementing-a-generative-adversarial-network-gan-dcgan-to-draw-human-faces-8291616904a
https://github.com/kvpratama/gan/tree/master/pokemon
https://github.com/moxiegushi/pokeGAN
Most ressources make use of tensorflow thoughout the modeling. I want to use Keras (pure tensorflow is still a possibility) so we we'll see about that.
## STEPS

### DATA GENERATOR
we will use the DataLoader function from the Pytorch library  which allow us to load images per batch, apply transforms and put them into tensor form for the model to use. 
First though, we need all images to be in JPEG format with a white background. 
Thus, first step in our data pre-processing as we have both .jpg images and .png. Is to make a function that take the .png images as input, add a white background to them and save them as JPEG. 
the goal is to have a directory a ready to use images at the end so it should also transfer the JPEG images in the good directory. 

### GERERATOR

### DISCRIMINATOR

### TRAINING PROCEDURE

### EVALUATION
Checkpoint every 8-9 epoch and manually evaluating by using the model on batch of noise. We observe a few cool results, the model doesn't produce really "better" pokemons after around epoch 100. Also, around epoch 350 the generator finds pattern that seems to fool the discriminator and repeat them. 

## **Code**
<hr />
