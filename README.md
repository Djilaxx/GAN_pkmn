# GAN Pokemon

I will use this project to learn how to train GAN's, how they work and apply it on Pokemon data to try and create new fun forms of pokemon.  
There are a few similar projects on Github that inspired me :  
- https://github.com/kvpratama/gan/tree/master/pokemon
- https://github.com/moxiegushi/pokeGAN  

## **DCGAN**
<hr />

## **Pokemon GAN**
<hr />

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

```
``` 

## **Reference**
[1] [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)  
[2] [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661v1)