import torch


def label_smoothing(y, device, real = True):
    '''
    Smoothing the label values
    1 will be between 0.7 and 1.2
    0 will be between 0 and 0.2
    '''
    if real:
        return y - 0.3 + (torch.rand(y.shape,device=device)*0.5)
    else: 
        return y + (torch.rand(y.shape,device=device)*0.2)