import torch
import torch.nn as nn

class Loss_fct:
    def __init__(self):
        self.name = "BCE_loss"
        self.loss_fct = nn.BCEWithLogitsLoss()

    def dis_loss(self, output_real, output_fake, label_real ,label_fake):
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

