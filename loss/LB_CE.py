import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper functions from fastai
def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss

# Implementation from fastai https://github.com/fastai/fastai2/blob/master/fastai2/layers.py#L338
'''
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, ε:float=0.1, reduction='mean'):
        super().__init__()
        self.ε,self.reduction = ε,reduction
    
    def forward(self, output, target):
        # number of classes
        c = 2
        log_preds = nn.LogSoftmax(output)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = nn.NLLLoss(log_preds, target, reduction=self.reduction)
        # (1-ε)* H(q,p) + ε*H(u,p)
        return (1-self.ε)*nll + self.ε*(loss/c) 
'''

class LabelSmoothingCrossEntropy(nn.Module):
    y_int = True
    def __init__(self, eps:float=0.1, reduction='mean'): self.eps,self.reduction = eps,reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum': loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1) #We divide by that size at the return line so sum and not mean
            if self.reduction=='mean':  loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target.long(), reduction=self.reduction)

    def activation(self, out): return F.softmax(out, dim=-1)
    def decodes(self, out):    return out.argmax(dim=-1)

class Loss_fct:
    """
    Label Smoothing cross entropy loss class
    """
    def __init__(self):
        self.name = "LB_CE_loss"
        self.loss_fct = LabelSmoothingCrossEntropy()

    def dis_loss(self, output_real, label_real, output_fake, label_fake):
        errD_real = self.loss_fct(output_real, label_real)
        errD_fake = self.loss_fct(output_fake, label_fake)
        errD = errD_real + errD_fake
        return errD

    def gen_loss(self, output_gen, label_gen):
        errG = self.loss_fct(output_gen, label_gen)
        return errG

    def compute_probs(self, output_real, output_fake):
        """
        Computes probabilities from real/fake images logits.
        Args:
            output_real (Tensor): A batch of output logits of shape (N, 1) from real images.
            output_fake (Tensor): A batch of output logits of shape (N, 1) from fake images.
        Returns:
            tuple: Average probabilities of real/fake image considered as real for the batch.
        """
        D_x = output_real.mean().item()
        D_G_z1 = output_fake.mean().item()
        return D_x, D_G_z1
