import torch.nn as nn


class DownsamplingBlock(nn.Module):
    """Defines the Unet downsampling block. 
    
    Consists of Convolution-BatchNorm-ReLU layer with k filters.
    """
    def __init__(self, c_in, c_out, kernel_size=4, stride=2, 
                 padding=1, negative_slope=0.2, use_norm=True):
        """
        Initializes the UnetDownsamplingBlock.
        
        Args:
            c_in (int): The number of input channels.
            c_out (int): The number of output channels.
            kernel_size (int, optional): The size of the convolving kernel. Default is 4.
            stride (int, optional): Stride of the convolution. Default is 2.
            padding (int, optional): Zero-padding added to both sides of the input. Default is 0.
            negative_slope (float, optional): Negative slope for the LeakyReLU activation function. Default is 0.2.
            use_norm (bool, optinal): If use norm layer. If True add a BatchNorm layer after Conv. Default is True.
        """
        super(DownsamplingBlock, self).__init__()
        block = []
        block += [nn.Conv2d(in_channels=c_in, out_channels=c_out,
                          kernel_size=kernel_size, stride=stride, padding=padding,
                          bias=(not use_norm) # No need to use a bias if there is a batchnorm layer after conv
                          )]
        if use_norm:
            block += [nn.BatchNorm2d(num_features=c_out)]
        
        block += [nn.LeakyReLU(negative_slope=negative_slope)]

        self.conv_block = nn.Sequential(*block)
        
    def forward(self, x):
        return self.conv_block(x)
    

class UpsamplingBlock(nn.Module):
    """Defines the Unet upsampling block.
    """
    def __init__(self, c_in, c_out, kernel_size=4, stride=2, 
                 padding=1, use_dropout=False, use_upsampling=False, mode='nearest'):
        
        """
        Initializes the Unet Upsampling Block.
        
        Args:
            c_in (int): The number of input channels.
            c_out (int): The number of output channels.
            kernel_size (int, optional): Size of the convolving kernel. Default is 4.
            stride (int, optional): Stride of the convolution. Default is 2.
            padding (int, optional): Zero-padding added to both sides of the input. Default is 0.
            use_dropout (bool, optional): if use dropout layers. Default is False.
            upsample (bool, optinal): if use upsampling rather than transpose convolution. Default is False.
            mode (str, optional): the upsampling algorithm: one of 'nearest', 
                'bilinear', 'bicubic'. Default: 'nearest'
        """
        super(UpsamplingBlock, self).__init__()
        block = []
        if use_upsampling:
            # Transpose convolution causes checkerboard artifacts. Upsampling
            # followed by a regular convolutions produces better results appearantly
            # Please check for further reading: https://distill.pub/2016/deconv-checkerboard/
            # Odena, et al., "Deconvolution and Checkerboard Artifacts", Distill, 2016. http://doi.org/10.23915/distill.00003
            
            mode = mode if mode in ('nearest', 'bilinear', 'bicubic') else 'nearest'
            
            block += [nn.Sequential(
                nn.Upsample(scale_factor=2, mode=mode),
                nn.Conv2d(in_channels=c_in, out_channels=c_out,
                          kernel_size=3, stride=1, padding=padding,
                          bias=False
                          )
                )]
        else:
            block += [nn.ConvTranspose2d(in_channels=c_in, 
                                         out_channels=c_out,
                                         kernel_size=kernel_size, 
                                         stride=stride,
                                         padding=padding, bias=False
                                         )
                     ]
        
        block += [nn.BatchNorm2d(num_features=c_out)]

        if use_dropout:
            block += [nn.Dropout(0.5)]
            
        block += [nn.ReLU()]

        self.conv_block = nn.Sequential(*block)

    def forward(self, x):
        return self.conv_block(x)