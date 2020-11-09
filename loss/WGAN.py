import torch
import torch.nn as nn
import torch.nn.parallel
from torch import autograd

class Loss_fct:
    def __init__(self):
        self.name = "WGAN_loss"

    def compute_gradient_penalty_loss(self, real_images, fake_images, discriminator, gp_scale=10.0):
        """
        Computes gradient penalty loss, as based on:
        https://github.com/jalola/improved-wgan-pytorch/blob/master/gan_train.py
        
        Args:
            real_images (Tensor): A batch of real images of shape (N, 3, H, W).
            fake_images (Tensor): A batch of fake images of shape (N, 3, H, W).
            gp_scale (float): Gradient penalty lamda parameter.
        Returns:
            Tensor: Scalar gradient penalty loss.
        """
        # Obtain parameters
        N, _, H, W = real_images.shape
        device = real_images.device

        # Randomly sample some alpha between 0 and 1 for interpolation
        # where alpha is of the same shape for elementwise multiplication.
        alpha = torch.rand(N, 1)
        alpha = alpha.expand(N, int(real_images.nelement() / N)).contiguous()
        alpha = alpha.view(N, 3, H, W)
        alpha = alpha.to(device)

        # Obtain interpolates on line between real/fake images.
        interpolates = alpha * real_images.detach() \
            + ((1 - alpha) * fake_images.detach())
        interpolates = interpolates.to(device)
        interpolates.requires_grad_(True)

        # Get gradients of interpolates
        disc_interpolates = discriminator(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates,
                                    inputs=interpolates,
                                    grad_outputs=torch.ones(
                                    disc_interpolates.size()).to(device),
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        # Compute GP loss
        gradient_penalty = (
            (gradients.norm(2, dim=1) - 1)**2).mean() * gp_scale

        return gradient_penalty
        
    def compute_probs(self, output_real, output_fake):
        """
        Computes probabilities from real/fake images logits.
        Args:
            output_real (Tensor): A batch of output logits of shape (N, 1) from real images.
            output_fake (Tensor): A batch of output logits of shape (N, 1) from fake images.
        Returns:
            tuple: Average probabilities of real/fake image considered as real for the batch.
        """
        D_x = torch.sigmoid(output_real).mean().item()
        D_G_z1 = torch.sigmoid(output_fake).mean().item()
        return D_x, D_G_z1
        
    def dis_loss(self, output_real, output_fake, label_real, label_fake):
        """
        Computes the wasserstein loss for the discriminator.
        Args:
            output_real (Tensor): Discriminator output logits for real images.
            output_fake (Tensor): Discriminator output logits for fake images.
        Returns:
            Tensor: A scalar tensor loss output.        
        """
        errD = -1.0 * output_real.mean() + output_fake.mean()

        return errD


    def gen_loss(self, output_gen, label_gen):
        """
        Computes the wasserstein loss for generator.
        Args:
            output_fake (Tensor): Discriminator output logits for fake images.
        Returns:
            Tensor: A scalar tensor loss output.
        """
        errG = -output_gen.mean()

        return errG