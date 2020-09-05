import torch.nn as nn 

def weights_init(m):
    '''
    The function check the layers name and init it's weights with samples from a N(0,0.02)
    Init BN with N(1,0.02) (why?) and bias at 0 g
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)