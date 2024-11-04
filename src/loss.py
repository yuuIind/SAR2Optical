import torch
import torch.nn as nn
import torch.nn.functional as F

class GANLoss(nn.Module):
    """Create a GAN Loss"""
    def __init__(self, mode='vanilla'):
        super(GANLoss, self).__init__()


class WGANLoss(nn.Module):
    """Create a WGAN-GP Loss"""
    def __init__(self, weight):
        super(WGANLoss, self).__init__()
        self.weight = weight

    def forward():
        pass

