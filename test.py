"""
Evaluation Script
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision import models

import numpy as np

from utils.config import Config
from src.dataset import Sentinel
from src.pix2pix import Pix2Pix
from src.metric import extract_features, calculate_fid

def main():
    # Load configuration
    config = Config('config.yaml')
    # Set device
    device = torch.device(config['training']['device'])

    inception = models.inception_v3(weights='DEFAULT', transform_input=False).eval().to(device)

    # transforms from inception_v3 documentation
    transform = v2.Compose([
        v2.Resize(342),
        v2.CenterCrop(299),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = Sentinel(
        root_dir=config['dataset']['root_dir'],
        split_type="test",
        split_mode=config['dataset']['split_mode'],
        split_ratio=config['dataset']['split_ratio'],
        split_file=config['dataset']['split_file'],
        seed=config['dataset']['seed']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=config['dataset']['shuffle'],
        num_workers=config['training']['num_workers']
    )

    # Create model
    model = Pix2Pix(
        c_in=config['model']['c_in'],
        c_out=config['model']['c_out'],
        is_train=False,
        use_upsampling=config['model']['use_upsampling'],
        mode=config['model']['mode'],
    ).to(device).eval()

    gen_checkpoint = Path(config['training']['gen_checkpoint'])

    if not gen_checkpoint.exists():
        raise FileNotFoundError(f"Generator checkpoint file not found: {gen_checkpoint}\nPlease check config.yaml")
    
    model.load_model(gen_path=gen_checkpoint)

    target_features = []
    fake_features = []

    for real_images, target_images in dataloader:
        real_images, target_images = real_images.to(device), target_images.to(device)
        
        # Pix2Pix.generate() gets a scaled tensor ([0,1]) returns a uint8 tensor ([0,255])
        fake_images = model.generate(real_images, is_scaled=True, to_uint8=True) 

        # Get target features
        target_images = (target_images * 255).to(dtype=torch.uint8)
        target_images = transform(target_images)
        target_feats = extract_features(target_images, inception)
        target_features.append(target_feats.cpu().numpy())

        # Get fake features
        fake_images = transform(fake_images)
        fake_feats = extract_features(fake_images, inception)
        fake_features.append(fake_feats.cpu().numpy())

    # Convert lists to numpy arrays
    real_features = np.concatenate(target_features, axis=0)
    generated_features = np.concatenate(fake_features, axis=0)

    # Compute FID score
    fid_score = calculate_fid(real_features, generated_features)
    print(f"FID Score: {fid_score}")


    