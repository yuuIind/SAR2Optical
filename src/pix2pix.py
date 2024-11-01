import torch
import torch.nn as nn
import torch.nn.functional as F

from .network import UnetGenerator, PatchGAN

class Pix2Pix(nn.Module):
    """Create a Pix2Pix class. It is a model for image to image translation tasks.
    By default, the model uses a Unet architecture for generator with transposed
    convolution. The discriminator is 70x70 PatchGAN discriminator, by default.
     """
    def __init__(self, c_in=3, c_out=3, is_train=True, netD='patch', 
                 use_upsampling=False, mode='nearest', c_hid=64, n_layers=3):
        super(Pix2Pix, self).__init__()
        self.gen = UnetGenerator(c_in=c_in, c_out=c_out, use_upsampling=use_upsampling, mode=mode)
        self.gen = self.gen.apply(self.weights_init)
        if is_train:
            self.disc = PatchGAN(c_in=c_in, c_hid=c_hid, mode=netD, n_layers=n_layers)
            self.disc = self.disc.apply(self.weights_init)
        
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)