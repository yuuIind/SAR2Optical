import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DownsamplingBlock, UpsamplingBlock

class UnetEncoder(nn.Module):
    """Create the Unet Encoder Network.
    
    C64-C128-C256-C512-C512-C512-C512-C512
    """
    def __init__(self, in_channels=3, out_channels=512):
        """
        Constructs the Unet Encoder Network.

        Ck denote a Convolution-BatchNorm-ReLU layer with k filters.
            C64-C128-C256-C512-C512-C512-C512-C512
        Args:
            in_channels (int, optional): Number of input channels.
            out_channels (int, optional): Number of output channels. Default is 512.
        """
        super(UnetEncoder, self).__init__()
        self.enc1 = DownsamplingBlock(in_channels, 64, use_norm=False) # C64
        self.enc2 = DownsamplingBlock(64, 128) # C128
        self.enc3 = DownsamplingBlock(128, 256) # C256
        self.enc4 = DownsamplingBlock(256, 512) # C512
        self.enc5 = DownsamplingBlock(512, 512) # C512
        self.enc6 = DownsamplingBlock(512, 512) # C512
        self.enc7 = DownsamplingBlock(512, 512) # C512
        self.enc8 = DownsamplingBlock(512, out_channels) # C512

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
    def __init__(self, in_channels=512, out_channels=64):
        """
        Constructs the Unet Decoder Network.

        Ck denote a Convolution-BatchNorm-ReLU layer with k filters.
        
        CDk denotes a Convolution-BatchNorm-Dropout-ReLU layer with a dropout rate of 50%.
            CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
        Args:
            in_channels (int): Number of input channels.
            out_channels (int, optional): Number of output channels. Default is 512.
        """
        super(UnetDecoder, self).__init__()
        self.conv9 = UpsamplingBlock(in_channels, 512, use_dropout=True, upsample=False) # CD512
        self.conv10 = UpsamplingBlock(1024, 512, use_dropout=True, upsample=False) # CD1024
        self.conv11 = UpsamplingBlock(1024, 512, use_dropout=True, upsample=False) # CD1024
        self.conv12 = UpsamplingBlock(1024, 512, upsample=False) # C1024
        self.conv13 = UpsamplingBlock(1024, 256, upsample=False) # C1024
        self.conv14 = UpsamplingBlock(512, 128, upsample=False) # C512
        self.conv15 = UpsamplingBlock(256, 64, upsample=False) # C256
        self.conv16 = UpsamplingBlock(128, out_channels, upsample=False) # C128
    

    def forward(self, x):
        x9 = torch.cat([x[1], self.conv9(x[0])], 1) # (N,1024,H,W)
        x10 = torch.cat([x[2], self.conv10(x9)], 1) # (N,1024,H,W)
        x11 = torch.cat([x[3], self.conv11(x10)], 1) # (N,1024,H,W)
        x12 = torch.cat([x[4], self.conv12(x11)], 1) # (N,1024,H,W)
        x13 = torch.cat([x[5], self.conv13(x12)], 1) # (N,512,H,W)
        x14 = torch.cat([x[6], self.conv14(x13)], 1) # (N,256,H,W)
        x15 = torch.cat([x[7], self.conv15(x14)], 1) # (N,128,H,W)
        out = self.conv16(x15) # (N,64,H,W)
        return out
    

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""
    def __init__(self, in_channels=3, out_channels=3):
        """
        Constructs a Unet generator
        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
        """
        super(UnetGenerator, self).__init__()
        self.encoder = UnetEncoder(in_channels=in_channels)
        self.decoder = UnetDecoder()
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
            nn.Conv2d(in_channels=64, out_channels=out_channels,
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