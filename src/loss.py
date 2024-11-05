import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Use for later
"""

class GANLoss(nn.Module):
    """Create a GAN Loss"""
    def __init__(self, mode='vanilla', lambda_gp=10):
        super(GANLoss, self).__init__()
        self.mode = mode
        if mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif mode == 'wgan-gp':
            self.loss = self._wgan_gp_loss
            self.lambda_gp = lambda_gp
        else:
            raise ValueError(f"Invalid GAN mode: {mode}")

    def _wgan_gp_loss(self, pred_real, pred_fake, gradient_penalty):
        # Wasserstein GAN with Gradient Penalty loss
        loss_D_real = -pred_real.mean()
        loss_D_fake = pred_fake.mean()
        # Combined loss
        loss_D = loss_D_real + loss_D_fake + self.lambda_gp * gradient_penalty
        return loss_D
    
    def _gradient_penalty(self, real_images, target_images, fake_images):
        epsilon =  torch.rand(target_images.size(0), 1, 1, 1, device=target_images.device)
        x_hat = (epsilon * target_images + (1 - epsilon) * fake_images).requires_grad_(True)
        pred = 0

    def forward(self, pred_real, pred_fake, is_disc=True):
        if self.mode == 'wgan-gp':
            return self._wgan_gp_loss(pred_real, pred_fake)
        else:
            loss_D_real = self.loss(pred_real, torch.ones_like(pred_real))
            loss_D_fake = self.loss(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            return loss_D
