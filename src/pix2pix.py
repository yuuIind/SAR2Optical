import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import UnetGenerator, PatchGAN

class Pix2Pix(nn.Module):
    """Create a Pix2Pix class. It is a model for image to image translation tasks.
    By default, the model uses a Unet architecture for generator with transposed
    convolution. The discriminator is 70x70 PatchGAN discriminator, by default.
     """
    def __init__(self, 
                 c_in: int = 3, 
                 c_out: int = 3, 
                 is_train: bool = True,
                 netD: str = 'patch', 
                 gan_mode: str = 'vanilla',
                 lambda_L1: float = 100.0,
                 lambda_gp: float = 10.0,
                 is_CGAN: bool = True,
                 use_upsampling: bool = False,
                 mode: str = 'nearest',
                 c_hid: int = 64,
                 n_layers: int = 3,
                 lr: float = 0.0002,
                 beta1: float = 0.5,
                 beta2: float = 0.999
                 ):
        """Constructs the Pix2Pix class.
        
        Args:
            c_in: Number of input channels
            c_out: Number of output channels
            is_train: Whether the model is in training mode
            netD: Type of discriminator ('patch' or 'pixel')
            gan_mode: Type of GAN loss ('vanilla', 'lsgan', or 'wgan-gp')
            lambda_L1: Weight for L1 loss
            lambda_gp: Weight for gradient penalty (WGAN-GP only)
            is_CGAN: If True, use conditional GAN architecture
            use_upsampling: If True, use upsampling in generator instead of transpose conv
            mode: Upsampling mode ('nearest', 'bilinear', 'bicubic')
            c_hid: Number of base filters in discriminator
            n_layers: Number of layers in discriminator
            lr: Learning rate
            beta1: Beta1 parameter for Adam optimizer
            beta2: Beta2 parameter for Adam optimizer
        """
        super(Pix2Pix, self).__init__()
        self.is_CGAN = is_CGAN
        self.lambda_L1 = lambda_L1


        self.gen = UnetGenerator(c_in=c_in, c_out=c_out, use_upsampling=use_upsampling, mode=mode)
        self.gen = self.gen.apply(self.weights_init)
        if is_train:
            # Conditional GANs need both input and output together, the total input channel is c_in+c_out
            disc_in = c_out if is_CGAN else c_in + c_out
            self.disc = PatchGAN(c_in=disc_in, c_hid=c_hid, mode=netD, n_layers=n_layers) 
            self.disc = self.disc.apply(self.weights_init)
        
        self.gen_optimizer = torch.optim.Adam(self.gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.disc_optimizer = torch.optim.Adam(self.disc.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.criterion = nn.BCEWithLogitsLoss()
        self.criterionL1 = torch.nn.L1Loss()
        
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.gen(x)
    
    def stepD(self, real_images, target_images, fake_images):
        if self.is_CGAN:
            # Conditional GANs need both input and output together, the total input channel is c_in+c_out
            real_AB = torch.cat(real_images, target_images, dim=1)
            fake_AB = torch.cat(real_images, fake_images.detach(), dim=1)
        else:
            real_AB = target_images
            fake_AB = fake_images.detach()
          
        # Forward pass through the discriminator
        pred_real = self.disc(real_AB)
        pred_fake = self.disc(fake_AB)

        # Compute the losses
        lossD_real = self.criterion(pred_real, torch.ones_like(pred_real))
        lossD_fake = self.criterion(pred_fake, torch.zeros_like(pred_fake))
        lossD = (lossD_real + lossD_fake) / 2
        return lossD
    
    def stepG(self, real_images, target_images, fake_images):
        if self.is_CGAN:
            # Conditional GANs need both input and output together, the total input channel is c_in+c_out
            fake_AB = torch.cat(real_images, fake_images, dim=1)
        else:
            fake_AB = fake_images
          
        # Forward pass through the discriminator
        pred_fake = self.disc(fake_AB)

        # Compute the losses
        lossG_GaN = self.criterion(pred_fake, torch.ones_like(pred_fake))
        lossG_L1 = self.criterionL1(fake_images, target_images)
        lossG = lossG_GaN + self.lambdaL1 * lossG_L1
        return lossG
    
    def train(self, real_images, target_images):
        # Forward pass through the generator
        fake_images = self.forward(real_images)
        
        # Update discriminator
        self.disc_optimizer.zero_grad() # Reset the gradients for D
        lossD = self.stepD(real_images, target_images, fake_images) # Compute the loss
        lossD.backward()
        self.disc_optimizer.step() # Update D

        # Update generator
        self.gen_optimizer.zero_grad() # Reset the gradients for D
        lossG = self.stepG(real_images, target_images, fake_images) # Compute the loss
        lossG.backward()
        self.gen_optimizer.step() # Update D

        return lossD.item(), lossG.item()