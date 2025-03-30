import torch
import torch.nn as nn

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
                 lambda_L1: float = 100.0,
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
            lambda_L1: Weight for L1 loss
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
        self.is_train = is_train

        self.gen = UnetGenerator(c_in=c_in, c_out=c_out, use_upsampling=use_upsampling, mode=mode)
        self.gen = self.gen.apply(self.weights_init)
        
        if self.is_train:
            # Conditional GANs need both input and output together, the total input channel is c_in+c_out
            disc_in = c_in + c_out if is_CGAN else c_out
            self.disc = PatchGAN(c_in=disc_in, c_hid=c_hid, mode=netD, n_layers=n_layers) 
            self.disc = self.disc.apply(self.weights_init)

            # Initialize optimizers
            self.gen_optimizer = torch.optim.Adam(
                self.gen.parameters(), lr=lr, betas=(beta1, beta2))
            self.disc_optimizer = torch.optim.Adam(
                self.disc.parameters(), lr=lr, betas=(beta1, beta2))

            # Initialize loss functions
            self.criterion = nn.BCEWithLogitsLoss()
            self.criterion_L1 = nn.L1Loss()
    
    def forward(self, x: torch.Tensor):
        return self.gen(x)
    
    @staticmethod    
    def weights_init(m):
        """Initialize network weights.
        
        Args:
            m: network module
        """
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

    def _get_disc_inputs(self, 
                         real_images: torch.Tensor,
                         target_images: torch.Tensor, 
                         fake_images: torch.Tensor
                         ):
        """Prepare discriminator inputs based on conditional/unconditional setup."""
        if self.is_CGAN:
            # Conditional GANs need both input and output together, 
            # Therefore, the total input channel is c_in+c_out
            real_AB = torch.cat([real_images, target_images], dim=1)
            fake_AB = torch.cat([real_images, 
                               fake_images.detach()], 
                               dim=1)
        else:
            real_AB = target_images
            fake_AB = fake_images.detach()
        return real_AB, fake_AB
    
    def _get_gen_inputs(self, 
                        real_images: torch.Tensor, 
                        fake_images: torch.Tensor
                        ):
        """Prepare discriminator inputs based on conditional/unconditional setup."""
        if self.is_CGAN:
            # Conditional GANs need both input and output together, 
            # Therefore, the total input channel is c_in+c_out
            fake_AB = torch.cat([real_images, 
                               fake_images], 
                               dim=1)
        else:
            fake_AB = fake_images
        return fake_AB
    
    
    def step_discriminator(self, 
                           real_images: torch.Tensor, 
                           target_images: torch.Tensor, 
                           fake_images: torch.Tensor
                           ):
        """Discriminator forward/backward pass.
        
        Args:
            real_images: Input images
            target_images: Ground truth images
            fake_images: Generated images
            
        Returns:
            Discriminator loss value
        """
        # Prepare inputs
        real_AB, fake_AB = self._get_disc_inputs(real_images, target_images, 
                                                fake_images)
          
        # Forward pass through the discriminator
        pred_real = self.disc(real_AB) # D(x, y)
        pred_fake = self.disc(fake_AB) # D(x, G(x))

        # Compute the losses
        lossD_real = self.criterion(pred_real, torch.ones_like(pred_real)) # (D(x, y), 1)
        lossD_fake = self.criterion(pred_fake, torch.zeros_like(pred_fake)) # (D(x, y), 0)
        lossD = (lossD_real + lossD_fake) * 0.5 # Combined Loss
        return lossD
    
    def step_generator(self, 
                       real_images: torch.Tensor, 
                       target_images: torch.Tensor, 
                       fake_images: torch.Tensor
                       ):
        """Discriminator forward/backward pass.
        
        Args:
            real_images: Input images
            target_images: Ground truth images
            fake_images: Generated images
            
        Returns:
            Discriminator loss value
        """
        # Prepare input
        fake_AB = self._get_gen_inputs(real_images, fake_images)
          
        # Forward pass through the discriminator
        pred_fake = self.disc(fake_AB)

        # Compute the losses
        lossG_GaN = self.criterion(pred_fake, torch.ones_like(pred_fake)) # GAN Loss
        lossG_L1 = self.criterion_L1(fake_images, target_images)           # L1 Loss
        lossG = lossG_GaN + self.lambda_L1 * lossG_L1                      # Combined Loss
        # Return total loss and individual components
        return lossG, {
            'loss_G': lossG.item(),
            'loss_G_GAN': lossG_GaN.item(),
            'loss_G_L1': lossG_L1.item()
        }
    
    def train_step(self, 
                   real_images: torch.Tensor, 
                   target_images: torch.Tensor
                   ):
        """Performs a single training step.
        
        Args:
            real_images: Input images
            target_images: Ground truth images
            
        Returns:
            Dictionary containing all loss values from this step
        """
        # Forward pass through the generator
        fake_images = self.forward(real_images)
        
        # Update discriminator
        self.disc_optimizer.zero_grad() # Reset the gradients for D
        lossD = self.step_discriminator(real_images, target_images, fake_images) # Compute the loss
        lossD.backward()
        self.disc_optimizer.step() # Update D

        # Update generator
        self.gen_optimizer.zero_grad() # Reset the gradients for D
        lossG, G_losses = self.step_generator(real_images, target_images, fake_images) # Compute the loss
        lossG.backward()
        self.gen_optimizer.step() # Update D

        # Return all losses
        return {
            'loss_D': lossD.item(),
            **G_losses
        }
    
    def validation_step(self, 
                   real_images: torch.Tensor, 
                   target_images: torch.Tensor
                   ):
        """Performs a single validation step.
        
        Args:
            real_images: Input images
            target_images: Ground truth images
            
        Returns:
            Dictionary containing all loss values from this step
        """
        with torch.no_grad():
            # Forward pass through the generator
            fake_images = self.forward(real_images)

            # Compute the loss for D
            lossD = self.step_discriminator(real_images, target_images, fake_images)
            
            # Compute the loss for G
            _, G_losses = self.step_generator(real_images, target_images, fake_images)

        # Return all losses
        return {
            'loss_D': lossD.item(),
            **G_losses
        }
    
    def generate(self, 
                 real_images: torch.Tensor, 
                 is_scaled: bool = False, 
                 to_uint8: bool = False
                 ):
        if not is_scaled:
            real_images = real_images.to(dtype=torch.float32) # Make sure it's a float tensor
            real_images = real_images / 255.0 # Normalize to [0, 1]
            real_images = (real_images - 0.5) / 0.5 # Scale to [-1, 1]
        # Normalization is applied unconditionally if the line below is uncommented.
        # Logically, it should be applied only when is_scaled is False. 
        # However, I might be stupid and it could be a design oversight. 
        # So, If someone complains later how the training does not work, 
        # then we know why :)
        # THE LINE IN QUESTION IS COMMENTED OUT BELOW
        # real_images = (real_images - 0.5) / 0.5 # Scale to [-1, 1]
        with torch.no_grad(): # generate image
            generated_images = self.forward(real_images)

        generated_images = (generated_images + 1) / 2  # Rescale to [0, 1]
        if to_uint8:
            generated_images = (generated_images* 255).to(dtype=torch.uint8)  # Scale to [0, 255] and convert to uint8
        
        return generated_images

            
    def save_model(self, gen_path: str, disc_path: str = None):
        """
        Saves the generator model's state dictionary to the specified path.
        If in training mode and a discriminator path is provided, saves the
        discriminator model's state dictionary as well.

        Args:
            gen_path (str): The file path where the generator model's state dictionary will be saved.
            disc_path (str, optional): The file path where the discriminator model's state dictionary will be saved. Defaults to None.
        """
        torch.save(self.gen.state_dict(), gen_path)
        if self.is_train and disc_path is not None:
            torch.save(self.disc.state_dict(), disc_path)
    
    def load_model(self, gen_path: str, disc_path: str = None, device: str = None):
        """
        Loads the generator and optionally the discriminator model from the specified file paths.

        Args:
            gen_path (str): Path to the generator model file.
            disc_path (str, optional): Path to the discriminator model file. Defaults to None.
            device (torch.device, optional): The device on which to load the models. If None, the device of the model's parameters will be used. Defaults to None.

        Returns:
            None
        """
        device = device if device else next(self.gen.parameters()).device
        self.gen.load_state_dict(torch.load(gen_path, map_location=device, weights_only=True), strict=False)
        if disc_path is not None and self.is_train:
            device = device if device else next(self.disc.parameters()).device
            self.disc.load_state_dict(torch.load(gen_path, map_location=device, weights_only=True), strict=False)
    
    def save_optimizer(self, gen_opt_path: str, disc_opt_path: str = None):
        """
        Save the state of the optimizers to the specified file paths.
        Args:
            gen_opt_path (str): The file path to save the generator optimizer state.
            disc_opt_path (str, optional): The file path to save the discriminator optimizer state. Defaults to None.
        Notes:
            This method saves the state of the generator optimizer to the specified `gen_opt_path`.
            If `disc_opt_path` is provided, it also saves the state of the discriminator optimizer to the specified path.
            This method only works if the model is in training mode (`is_train` is True). If the model is not in training mode,
            it will print a message indicating that the model is not initialized in train mode.
        """
        if self.is_train:
            torch.save(self.gen_optimizer.state_dict(), gen_opt_path)
            if disc_opt_path is not None:
                torch.save(self.disc_optimizer.state_dict(), disc_opt_path)
        else:
            print('Model is initialized in train mode. See `is_train` for more.')
    
    def load_optimizer(self, gen_opt_path: str, disc_opt_path: str = None):
        """
        Loads the optimizer states for the generator and discriminator from the specified file paths.
        Args:
            gen_opt_path (str): Path to the file containing the generator optimizer state.
            disc_opt_path (str, optional): Path to the file containing the discriminator optimizer state. Defaults to None.
        Notes:
            This method saves the state of the generator optimizer to the specified `gen_opt_path`.
            If `disc_opt_path` is provided, it also saves the state of the discriminator optimizer to the specified path.
            This method only works if the model is in training mode (`is_train` is True). If the model is not in training mode,
            it will print a message indicating that the model is not initialized in train mode.
        """
        if self.is_train:
            self.gen_optimizer.load_state_dict(torch.load(gen_opt_path, weights_only=True))
            if disc_opt_path is not None:
                self.disc_optimizer.load_state_dict(torch.load(disc_opt_path, weights_only=True))
        else:
            print('Model is initialized in train mode. See `is_train` for more.')
    
    def get_current_visuals(self, 
                            real_images: torch.Tensor, 
                            target_images: torch.Tensor
                            ):
        """Return visualization images.
        
        Args:
            real_images: Input images
            target_images: Ground truth images
            
        Returns:
            Dictionary containing input, target and generated images
        """
        with torch.no_grad():
            fake_images = self.gen(real_images)
        return {
            'real': real_images,
            'fake': fake_images,
            'target': target_images
        }