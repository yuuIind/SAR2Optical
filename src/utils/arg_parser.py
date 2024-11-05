import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description="Pix2Pix training, evaluation, and inference Parameters."
        )
    
    # Basic parameters
    basic_group = parser.add_argument_group('Basic Parameters')
    basic_group.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for training')
    basic_group.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # Model parameters
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument('--c_in', type=int, default=3, help='Number of input channels')
    model_group.add_argument('--c_out', type=int, default=3, help='Number of output channels')
    model_group.add_argument('--is_train', type=bool, default=True, help='Flag for training mode')
    model_group.add_argument('--netD', type=str, default='patch', choices=['patch', 'pixel'], help='Type of discriminator ("patch" or "pixel")')
    model_group.add_argument('--lambda_L1', type=float, default=100.0, help='Weight for L1 loss')
    model_group.add_argument('--is_CGAN', type=bool, default=True, help='Use conditional GAN')
    model_group.add_argument('--use_upsampling', type=bool, default=False, help='Flag to use upsampling in generator instead of transpose conv')
    model_group.add_argument('--mode', type=str, default='nearest', choices=['nearest', 'bilinear', 'bicubic'], help='Upsampling mode ("nearest", "bilinear", "bicubic")')
    model_group.add_argument('--c_hid', type=int, default=64, help='Number of base filters in discriminator')
    model_group.add_argument('--n_layers', type=int, default=3, help='Number of layers in discriminator (1,...,6)')

    # Dataset parameters
    dataset_group = parser.add_argument_group('Dataset Parameters')
    dataset_group.add_argument('--root_dir', type=str, required=True, help='Root directory of the dataset')
    dataset_group.add_argument('--split_mode', type=str, default='random', choices=['random', 'split'], help='Mode for splitting the dataset')
    dataset_group.add_argument('--split_ratio', type=float, nargs=3, default=(0.7, 0.15, 0.15), help='Ratio for train/validation/test split')
    dataset_group.add_argument('--split_file', type=str, default=None, help='Path to file specifying the dataset split')
    dataset_group.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    dataset_group.add_argument('--num_workers', type=int, default=2, help='Number of data loader workers')
    dataset_group.add_argument('--shuffle', type=bool, default=True, help='Shuffle the dataset')

    # Training parameters
    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    training_group.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    training_group.add_argument('--beta1', type=float, default=0.5, help='Beta1 hyperparameter for Adam optimizer')
    training_group.add_argument('--beta2', type=float, default=0.999, help='Beta2 hyperparameter for Adam optimizer')
    training_group.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for regularization')

    # Saving and loading parameters
    save_load_group = parser.add_argument_group('Saving and Loading Parameters')
    save_load_group.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    save_load_group.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
    save_load_group.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    save_load_group.add_argument('--load_checkpoint', type=str, help='Path to load a checkpoint for resuming training')

    return parser.parse_args()
    
