# train.py
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from tqdm import tqdm

from utils.config import Config
from utils.utils import setup_logging, init_comet, log_metrics
from src.dataset import Sentinel
from src.pix2pix import Pix2Pix

def save_checkpoint(
        model: Pix2Pix, 
        epoch: int, 
        config: Config,
        ):
    """Save model checkpoint"""
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save generator
    gen_filename = f"generator_epoch_{epoch}.pth"
    gen_path = checkpoint_dir / gen_filename
    
    # Save discriminator
    disc_filename = f"discriminator_epoch_{epoch}.pth"
    disc_path = checkpoint_dir / disc_filename
    
    model.save_model(str(gen_path), str(disc_path))
    
    # Save config with model files
    config.save(checkpoint_dir / "config.yaml")

def load_checkpoint(model: Pix2Pix, config: Config):
    """Load model checkpoint"""
    gen_checkpoint = Path(config['training']['gen_checkpoint'])
    disc_checkpoint = Path(config['training']['disc_checkpoint'])

    if not gen_checkpoint.exists():
        raise FileNotFoundError(f"Generator checkpoint file not found: {gen_checkpoint}\nPlease check config.yaml")
    if not disc_checkpoint.exists():
        raise FileNotFoundError(f"Generator checkpoint file not found: {disc_checkpoint}\nPlease check config.yaml")
    
    model.load_model(gen_path=gen_checkpoint, disc_path=disc_checkpoint)

def create_dataloader(config, split_type: str, transform):
    """Create dataset and dataloader based on split type"""
    dataset = Sentinel(
        root_dir=config['dataset']['root_dir'],
        split_type=split_type,
        transform=transform,
        split_mode=config['dataset']['split_mode'],
        split_ratio=config['dataset']['split_ratio'],
        split_file=config['dataset']['split_file'],
        seed=config['dataset']['seed']
    )
    return DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=config['dataset']['shuffle'],
        num_workers=config['training']['num_workers']
    )

def train_epoch(model, train_loader, device, epoch, experiment):
    """Train for one epoch"""
    model.train()
    total_lossD, total_lossG = 0.0, 0.0
    total_lossG_GAN, total_lossG_L1 = 0.0, 0.0

    with tqdm(train_loader, desc=f"Epoch {epoch}") as pbar:
        for real_images, target_images in pbar:
            real_images, target_images = real_images.to(device), target_images.to(device)
            losses = model.train_step(real_images, target_images)
            total_lossD += losses['loss_D']
            total_lossG += losses['loss_G']
            total_lossG_GAN += losses['loss_G_GAN']
            total_lossG_L1 += losses['loss_G_L1']
            pbar.set_postfix({"loss_D": losses['loss_D'], "loss_G": losses['loss_G']})

    num_steps = len(train_loader)
    loss_D = total_lossD / num_steps
    loss_G = total_lossG / num_steps
    loss_G_GAN = total_lossG_GAN / num_steps
    loss_G_L1 = total_lossG_L1 / num_steps

    losses = {
        'loss_D' : loss_D,
        'loss_G' : loss_G,
        'loss_G_GAN' : loss_G_GAN, 
        'loss_G_L1' : loss_G_L1,
    }

    # Log metrics for the epoch
    log_metrics(experiment, losses, epoch)

def validate(model: Pix2Pix, val_loader: DataLoader, device: torch.device, epoch, experiment):
    """Validate the model"""
    model.eval()
    total_lossD, total_lossG = 0.0, 0.0
    total_lossG_GAN, total_lossG_L1 = 0.0, 0.0
        
    for real_images, target_images in val_loader:
        real_images, target_images = real_images.to(device), target_images.to(device)
        losses = model.validation_step(real_images, target_images)
        total_lossD += losses['loss_D']
        total_lossG += losses['loss_G']
        total_lossG_GAN += losses['loss_G_GAN']
        total_lossG_L1 += losses['loss_G_L1']
    
    num_steps = len(val_loader)
    loss_D = total_lossD / num_steps
    loss_G = total_lossG / num_steps
    loss_G_GAN = total_lossG_GAN / num_steps
    loss_G_L1 = total_lossG_L1 / num_steps

    losses = {
        'Val loss_D' : loss_D,
        'Val loss_G' : loss_G,
        'Val loss_G_GAN' : loss_G_GAN, 
        'Val loss_G_L1' : loss_G_L1,
    }
    # Log metrics for the epoch
    log_metrics(experiment, losses, epoch)

def main():
    # Load configuration
    config = Config('config.yaml')
    use_validation = config['training']['use_validation']
    
    # Setup logging
    setup_logging(config)
    experiment = init_comet(config)
    if experiment:
        experiment.log_parameters(config['model'])
        experiment.log_parameters(config['training'])
        experiment.log_parameters(config['dataset'])
    
    # Set device
    device = torch.device(config['training']['device'])

    # Create transforms
    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5]),
    ])
    
    # Create dataloaders
    train_loader = create_dataloader(config, "train", train_transforms)

    # use validation
    if use_validation:
        val_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5], std=[0.5]),
        ])
        val_loader = create_dataloader(config, "val", val_transforms)
    
    # Create model
    model = Pix2Pix(
        c_in=config['model']['c_in'],
        c_out=config['model']['c_out'],
        netD=config['model']['netD'],
        lambda_L1=config['model']['lambda_L1'],
        is_CGAN=config['model']['is_CGAN'],
        use_upsampling=config['model']['use_upsampling'],
        mode=config['model']['mode'],
        c_hid=config['model']['c_hid'],
        n_layers=config['model']['n_layers'],
        lr=config['training']['lr'],
        beta1=config['training']['beta1'],
        beta2=config['training']['beta2']
    ).to(device)

    # Load checkpoint for resuming training
    start_epoch: int = 1
    end_epoch: int = config['training']['num_epochs'] + 1
    if config['training']['resume']:
        load_checkpoint(model, config)
        start_epoch = config['training'].get('resume_epoch', 1)
    
    model = torch.compile(model) # compile model for possible performance boost

    # Training loop
    for epoch in range(start_epoch, end_epoch):
        # Train
        train_epoch(model, train_loader, device, epoch, experiment)
        
        # Validate
        if use_validation:
            validate(model, val_loader, device, epoch, experiment)
        
        # Regular checkpoint saving
        if epoch % config['training']['save_freq'] == 0:
            save_checkpoint(model, epoch, config)
    
    # Save final model
    save_checkpoint(model, config['training']['num_epochs'], config)
    
    if config['logging']['comet']['enabled']:
        experiment.finish()

if __name__ == '__main__':
    main()