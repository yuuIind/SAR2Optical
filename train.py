"""
Train Script
"""
import comet_ml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from .src.pix2pix import Pix2Pix
from .src.dataset import Sentinel
from .src.utils.arg_parser import get_args

if __name__ == '__main__':
    args = get_args()

    comet_ml.login(api_key=api_key)
    # Create experiment object
    experiment = comet_ml.start(project_name="SAR-2-Optical")

    params = {
        'netD' : args.netD,
        'lambda_L1' : args.lambda_L1,
        'use_upsampling' : args.use_upsampling,
        'mode' : args.mode,
        'c_hid' : args.c_hid,
        'n_layers' : args.n_layers,
        'lr' : args.lr,
        'beta1' : args.beta1,
        'beta2' : args.beta2,
        'batch_size' : args.batch_size,
        'epochs' : args.epochs,
        'seed' : args.seed
    }

    experiment.log_parameters(params)

    SEED = args.seed
    DEVICE = args.device
    torch.manual_seed(SEED)

    # Initialize the Pix2Pix model
    model = Pix2Pix(
        c_in=args.c_in,
        c_out=args.c_out,
        is_train=args.is_train,
        netD=args.netD,
        lambda_L1=args.lambda_L1,
        is_CGAN=args.is_CGAN,
        use_upsampling=args.use_upsampling,
        mode=args.mode,
        c_hid=args.c_hid,
        n_layers=args.n_layers,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2
        )

    # Load the custom dataset
    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=0.5, std=0.5),
    ])

    dataset = Sentinel(
        root_dir=args.root_dir, 
        split_type='train', 
        transform=train_transforms, 
        split_mode=args.split_mode,
        split_ratio=args.split_mode, 
        split_file=args.split_file,
        seed=SEED
        )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=args.shuffle, 
        num_workers=args.num_workers
        )
    
    # Train the model
    num_epochs = 100
    for epoch in range(num_epochs):
        for real_images, target_images in dataloader:
            losses = model.train_step(real_images, target_images)
            # Log the losses
            print(f"Epoch [{epoch}/{num_epochs}] - Loss_D: {losses['loss_D']:.4f}, Loss_G: {losses['loss_G']:.4f}")
            experiment.log_metrics(
                {
                    'epoch': epoch, 
                    'lossD': losses['loss_D'],
                    'lossG' : losses['loss_G'],
                    'lossG_GAN' : losses['loss_G_GAN'],
                    'lossG_L1' : losses['loss_G_L1']
                }, 
                step=epoch
                )

        
    
