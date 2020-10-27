import torch
import torch.nn as nn

# Helper functions from fastai
def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss

# Implementation from fastai https://github.com/fastai/fastai2/blob/master/fastai2/layers.py#L338
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, ε:float=0.1, reduction='mean'):
        super().__init__()
        self.ε,self.reduction = ε,reduction
    
    def forward(self, output, target):
        # number of classes
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        # (1-ε)* H(q,p) + ε*H(u,p)
        return (1-self.ε)*nll + self.ε*(loss/c) 

class LB_CE_loss:
    """
    Label Smoothing cross entropy loss class
    """
    def __init__(self):
        self.loss_fct = LabelSmoothingCrossEntropy()

    def dis_loss(self, output_real, label_real, output_fake, label_fake):
        errD_real = self.loss_fct(output_real, label_real)
        errD_fake = self.loss_fct(output_fake, label_fake)
        errD_total = errD_real + errD_fake
        return errD, errD_real, errD_fake

    def gen_loss(self, output_G, label_G):
        errG = self.loss_fct(output_G, label_G)
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
