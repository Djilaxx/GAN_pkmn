import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from numpy.random import choice
from config import config

device = torch.device("cuda:0" if (torch.cuda.is_available() and config.MODEL.ngpu > 0) else "cpu")


def weights_init(m):
    '''
    The function check the layers name and init it's weights with samples from a N(0,0.02)
    Init BN with N(1,0.02) (why?) and bias at 0 
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def label_smoothing(y, real = True):
    '''
    Smoothing the label values
    1 will be between 0.7 and 1.2
    0 will be between 0 and 0.2
    '''
    if real:
        return y - 0.3 + (torch.rand(y.shape,device=device)*0.5)
    else: 
        return y + (torch.rand(y.shape,device=device)*0.2)


def noisy_labelling(y, p_flip, value_range, smooth_label=True):
    '''
    Return a tensor with noisy labels
    p_flip is the % of labels to which we add noise
    value range is for smooth labels we need to know to which range of values to transform (from [0.7 - 1.2] to [0 - 0.2] for ex)
    '''
    n_select = int(p_flip*y.shape[0])
    flip_x = choice([i for i in range (y.shape[0])],size = n_select)
    if smooth_label:
        y[flip_x] = (((value_range[1] - value_range[0])*(y[flip_x] - min(y)))/(max(y) - min(y))) + value_range[0]
        return y
    else: 
        y[flip_x] = 1 - y[flip_x]
        return y