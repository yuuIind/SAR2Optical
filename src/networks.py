import torch
import torch.nn as nn

from .layers import DownsamplingBlock, UpsamplingBlock

class UnetEncoder(nn.Module):
    """Create the Unet Encoder Network.
    
    C64-C128-C256-C512-C512-C512-C512-C512
    """
    def __init__(self, c_in=3, c_out=512):
        """
        Constructs the Unet Encoder Network.

        Ck denote a Convolution-BatchNorm-ReLU layer with k filters.
            C64-C128-C256-C512-C512-C512-C512-C512
        Args:
            c_in (int, optional): Number of input channels.
            c_out (int, optional): Number of output channels. Default is 512.
        """
        super(UnetEncoder, self).__init__()
        self.enc1 = DownsamplingBlock(c_in, 64, use_norm=False) # C64
        self.enc2 = DownsamplingBlock(64, 128) # C128
        self.enc3 = DownsamplingBlock(128, 256) # C256
        self.enc4 = DownsamplingBlock(256, 512) # C512
        self.enc5 = DownsamplingBlock(512, 512) # C512
        self.enc6 = DownsamplingBlock(512, 512) # C512
        self.enc7 = DownsamplingBlock(512, 512) # C512
        self.enc8 = DownsamplingBlock(512, c_out) # C512

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc6(x5)
        x7 = self.enc7(x6)
        x8 = self.enc8(x7)
        out = [x8, x7, x6, x5, x4, x3, x2, x1] # latest activation is the first element
        return out
    

class UnetDecoder(nn.Module):
    """Creates the Unet Decoder Network.
    """
    def __init__(self, c_in=512, c_out=64, use_upsampling=False, mode='nearest'):
        """
        Constructs the Unet Decoder Network.

        Ck denote a Convolution-BatchNorm-ReLU layer with k filters.
        
        CDk denotes a Convolution-BatchNorm-Dropout-ReLU layer with a dropout rate of 50%.
            CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
        Args:
            c_in (int): Number of input channels.
            c_out (int, optional): Number of output channels. Default is 512.
            use_upsampling (bool, optional): Upsampling method for decoder. 
                If True, use upsampling layer followed regular convolution layer.
                If False, use transpose convolution. Default is False
            mode (str, optional): the upsampling algorithm: one of 'nearest', 
                'bilinear', 'bicubic'. Default: 'nearest'
        """
        super(UnetDecoder, self).__init__()
        self.dec1 = UpsamplingBlock(c_in, 512, use_dropout=True, use_upsampling=use_upsampling, mode=mode) # CD512
        self.dec2 = UpsamplingBlock(1024, 512, use_dropout=True, use_upsampling=use_upsampling, mode=mode) # CD1024
        self.dec3 = UpsamplingBlock(1024, 512, use_dropout=True, use_upsampling=use_upsampling, mode=mode) # CD1024
        self.dec4 = UpsamplingBlock(1024, 512, use_upsampling=use_upsampling, mode=mode) # C1024
        self.dec5 = UpsamplingBlock(1024, 256, use_upsampling=use_upsampling, mode=mode) # C1024
        self.dec6 = UpsamplingBlock(512, 128, use_upsampling=use_upsampling, mode=mode) # C512
        self.dec7 = UpsamplingBlock(256, 64, use_upsampling=use_upsampling, mode=mode) # C256
        self.dec8 = UpsamplingBlock(128, c_out, use_upsampling=use_upsampling, mode=mode) # C128
    

    def forward(self, x):
        x9 = torch.cat([x[1], self.dec1(x[0])], 1) # (N,1024,H,W)
        x10 = torch.cat([x[2], self.dec2(x9)], 1) # (N,1024,H,W)
        x11 = torch.cat([x[3], self.dec3(x10)], 1) # (N,1024,H,W)
        x12 = torch.cat([x[4], self.dec4(x11)], 1) # (N,1024,H,W)
        x13 = torch.cat([x[5], self.dec5(x12)], 1) # (N,512,H,W)
        x14 = torch.cat([x[6], self.dec6(x13)], 1) # (N,256,H,W)
        x15 = torch.cat([x[7], self.dec7(x14)], 1) # (N,128,H,W)
        out = self.dec8(x15) # (N,64,H,W)
        return out
    

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""
    def __init__(self, c_in=3, c_out=3, use_upsampling=False, mode='nearest'):
        """
        Constructs a Unet generator
        Args:
            c_in (int): The number of input channels.
            c_out (int): The number of output channels.
            use_upsampling (bool, optional): Upsampling method for decoder. 
                If True, use upsampling layer followed regular convolution layer.
                If False, use transpose convolution. Default is False
            mode (str, optional): the upsampling algorithm: one of 'nearest', 
                'bilinear', 'bicubic'. Default: 'nearest'
        """
        super(UnetGenerator, self).__init__()
        self.encoder = UnetEncoder(c_in=c_in)
        self.decoder = UnetDecoder(use_upsampling=use_upsampling, mode=mode)
        # In the paper, the authors state:
        #   """
        #       After the last layer in the decoder, a convolution is applied
        #       to map to the number of output channels (3 in general, except
        #       in colorization, where it is 2), followed by a Tanh function.
        #   """
        # However, in the official Lua implementation, only a Tanh layer is applied.
        # Therefore, I took the liberty of adding a convolutional layer with a 
        # kernel size of 3.
        # For more information please check the paper and official github repo:
        # https://github.com/phillipi/pix2pix
        # https://arxiv.org/abs/1611.07004
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=c_out,
                      kernel_size=3, stride=1, padding=1,
                      bias=True
                      ), 
            nn.Tanh()
            )
    
    def forward(self, x):
        outE = self.encoder(x)
        outD = self.decoder(outE)
        out = self.head(outD)
        return out


class PatchDiscriminator(nn.Module):
    """Create a PatchGAN discriminator"""
    def __init__(self, c_in=3, c_hid=64, n_layers=3):
        """Constructs a PatchGAN discriminator

        Args:
            c_in (int, optional): The number of input channels. Defaults to 3.
            c_hid (int, optional): The number of channels after first conv layer.
                Defaults to 64.
            n_layers (int, optional): the number of convolution blocks in the 
                discriminator. Defaults to 3.
        """
        super(PatchDiscriminator, self).__init__()
        model = [DownsamplingBlock(c_in, c_hid, use_norm=False)]

        n_p = 1  # multiplier for previous channel
        n_c = 1  # multiplier for current channel
        # last block is with stride of 1, therefore iterate (n_layers-1) times
        for n in range(1, n_layers): 
            n_p = n_c
            n_c = min(2**n, 8)  # The number of channels is 512 at most

            model += [DownsamplingBlock(c_hid*n_p, c_hid*n_c)]
        
        n_p = n_c
        n_c = min(2**n_layers, 8)
        model += [DownsamplingBlock(c_hid*n_p, c_hid*n_c, stride=1)] # last block is with stride of 1

        # last layer is a convolution followed by a Sigmoid function.
        model += [nn.Conv2d(in_channels=c_hid*n_c, out_channels=1, 
                            kernel_size=4, stride=1, padding=1, bias=True
                            )] 
        # Normally, there should be a sigmoid layer at the end of discriminator.
        # However, nn.BCEWithLogitsLoss combines the sigmoid layer with BCE loss, 
        # providing greater numerical stability. Therefore, the discriminator outputs
        # logits to take advantage of this stability.
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    

class PixelDiscriminator(nn.Module):
    """Create a PixelGAN discriminator (1x1 PatchGAN discriminator)"""
    def __init__(self, c_in=3, c_hid=64):
        """Constructs a PixelGAN discriminator, a special form of PatchGAN Discriminator.
        All convolutions are 1x1 spatial filters

        Args:
            c_in (int, optional): The number of input channels. Defaults to 3.
            c_hid (int, optional): The number of channels after first conv layer.
                Defaults to 64.
        """
        super(PixelDiscriminator, self).__init__()
        self.model = nn.Sequential(
            DownsamplingBlock(c_in, c_hid, kernel_size=1, stride=1, padding=0, use_norm=False),
            DownsamplingBlock(c_hid, c_hid*2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=c_hid*2, out_channels=1, kernel_size=1)
            )
        # Similar to PatchDiscriminator, there should be a sigmoid layer at the end of discriminator.
        # However, nn.BCEWithLogitsLoss combines the sigmoid layer with BCE loss, 
        # providing greater numerical stability. Therefore, the discriminator outputs
        # logits to take advantage of this stability.

    def forward(self, x):
        return self.model(x)
    

class PatchGAN(nn.Module):
    """Create a PatchGAN discriminator"""
    def __init__(self, c_in=3, c_hid=64, mode='patch', n_layers=3):
        """Constructs a PatchGAN discriminator.

        Args:
            c_in (int, optional): The number of input channels. Defaults to 3.
            c_hid (int, optional): The number of channels after first 
                convolutional layer. Defaults to 64.
            mode (str, optional): PatchGAN type. Use 'pixel' for PixelGAN, and 
                'patch' for other types. Defaults to 'patch'.
            n_layers (int, optional): PatchGAN number of layers. Defaults to 3.
                - 16x16 PatchGAN if n=1
                - 34x34 PatchGAN if n=2
                - 70x70 PatchGAN if n=3
                - 142x142 PatchGAN if n=4
                - 286x286 PatchGAN if n=5
                - 574x574 PatchGAN if n=6
        """
        super(PatchGAN, self).__init__()
        if mode == 'pixel':
            self.model = PixelDiscriminator(c_in, c_hid)
        else:
            self.model = PatchDiscriminator(c_in, c_hid, n_layers)
    
    def forward(self, x):
        return self.model(x)