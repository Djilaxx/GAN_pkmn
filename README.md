# GAN Pokemon

I will use this project to learn how to train GAN's, how they work and apply it on Pokemon data to try and create new fun forms of pokemon.  
There are a few similar projects on Github that inspired me :  
- https://github.com/kvpratama/gan/tree/master/pokemon
- https://github.com/moxiegushi/pokeGAN  

## **DCGAN**
<hr />

![DCGAN](imgs/dcgan_model.png) 

Basically, the **generator** is a CNN that is trained to turn a tensor of random (gaussian) noise into an image, and the **discriminator** is also a CNN that is trained to differentiate between real images and fake created by the **generator**. (I put a link to the paper in the references down below)  
The particularity of GAN's such as this one is that in the case of creating new pokemons is that there isn't a real benchmark to tell you when the model is good or not, it's good when it's able to create images that you find good enough. 
## **Pokemon GAN**
<hr />

<a href="url"><img src="imgs/Pkmn_img19.jpg" align="center" height="96" width="96" ></a>
<a href="url"><img src="imgs/Pkmn_img99.jpg" align="center" height="96" width="96" ></a>
<a href="url"><img src="imgs/Pkmn_img453.jpg" align="center" height="96" width="96" ></a>  


I used a pytorch implementation for the dataloading, model and training. For that i used an already existing one that i found [here], and reworked it to fit my needs. 
I also used the tips and tricks from this [github]. The goal is to smooth the training of the model and improve it's stability. 


The model is able to create blobs of color with forms that could ressemble animal/pokemon forms, but no details (no eyes, nose etc...).  
Such as those : 

<a href="url"><img src="imgs/fake1.png" align="center" height="64" width="64" ></a>
<a href="url"><img src="imgs/fake2.png" align="center" height="64" width="64" ></a>
<a href="url"><img src="imgs/fake3.png" align="center" height="64" width="64" ></a>  

as you can see, these are not super cool new pokemons yet, but this project is still in it's infancy.  

[here]: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
[github]: https://github.com/soumith/ganhacks

## **Next steps**
<hr />
The next steps for this project include :  

* Adding the functional evaluation function
* Improving the training function to checkpoint the model more effectively
* Changing the model (image size, more performant model)
* Improving the dataset (Cleaning images that are too similar, adding new images)


## **Code**
<hr />

### **Prerequisites**
```
python -m pip install -r requirement.txt
```

### **Train**
I apply the model on the pokemon data in the data_ready folder, you can re-train using :
```
python train.py --mode train 
```

You can also train on your own data (this is basically a torch implementation of DCGAN applied on pokemon data) by changing the root in config.py

### **Evaluate**
Feature is not ready yet 
```
``` 

## **Reference**
[1] [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)  
[2] [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661v1)