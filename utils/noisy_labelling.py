from numpy.random import choice

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