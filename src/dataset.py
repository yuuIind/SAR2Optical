import json
import random
from pathlib import Path
from enum import Enum
from typing import Tuple, List, Optional, Callable, Union, Literal

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


class SplitType(Enum):
    """Enumeration for dataset split types"""
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class Sentinel(Dataset):
    """
    A PyTorch Dataset for handling Sentinel-1&2 Image Pairs.
    
    This dataset assumes a directory structure of:
    root_dir/
        category1/
            s1/
                image1.png
                image2.png
            s2/
                image1.png
                image2.png
        category2/
            ...
    
    This class has support for train/val/test splits. When `split_type` is `None`, 
    uses the complete dataset. When `split_type` is specified 
    (``'train'``, ``'val'``, ``'test'``), the dataset can be split using:

    1. A split that defines which images belong to which split
    2. Random splitting with a specified ratio

    Args:
        root_dir (str | Path): Root directory containing the dataset
        split_type (str | None): Which split to use ('train', 'val', 'test') or None for full dataset
        transform (callable, optional): Transform to apply to both SAR and optical images
        split_mode (str, optional): How to split the dataset ('random', 'split')
        split_ratio (Tuple[float, float, float], optional): Ratio for train/val/test splits
        split_file (str | Path, optional): predefined the splits
        seed (int, optional): Random seed for reproducible splitting
        
    Attributes:
        root_dir (Path): Path to the dataset root directory
        transform (callable): Transform pipeline for the images
        image_pairs (List[Tuple[Path, Path]]): List of paired image paths (SAR, optical)
    """
    def __init__(self,
                 root_dir: Union[str, Path],
                 split_type: Optional[str] = None,
                 input_transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 split_mode: Literal['random', 'split'] = 'random',
                 split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                 split_file: Optional[Union[str, Path]] = None,
                 seed: int = 42):
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root directory not found: {self.root_dir}")
        
        # Convert string split_type to enum if provided
        self.split_type = SplitType(split_type) if split_type else None

        # Default transform pipeline
        self.input_transform = input_transform if input_transform else v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        self.target_transform = target_transform if target_transform else self.input_transform

        # Collect image pairs
        self.all_image_pairs = self._collect_images()

        # Apply split if specified
        if split_type:
            if split_mode == 'split' and split_file:
                self.image_pairs = self._apply_predefined_split(split_file)
            elif split_mode == 'random':
                self.image_pairs = self._apply_random_split(split_ratio, seed)
            else:
                raise ValueError("Invalid split configuration. Use either 'split' with a split_file or 'random' with split_ratio")
        else:
            # If no split type specified, use all images
            self.image_pairs = self.all_image_pairs

        print(f'Total image pairs found: {len(self)}')

    def _collect_images(self) -> List[Tuple[Path, Path]]:
        """
        Collects paired SAR (s1) and optical (s2) image paths from the dataset directory.
            
        Returns:
            List[Tuple[Path, Path]]: List of (SAR image path, optical image path) pairs
        """
        image_pairs = []
        
        # Iterate through category subdirectories
        for category in self.root_dir.iterdir():
            # Check if it's a directory
            if not category.is_dir():
                continue

            s1_path = category / 's1'
            s2_path = category / 's2'
            
            if not (s1_path.is_dir() and s2_path.is_dir()):
                # print(f"Missing s1 or s2 subdirectory in category: {category.name}")
                continue

            # Collect pairs
            for s1_file in s1_path.glob('*.png'):
                # Convert SAR filename to optical filename
                # e.g. 'ROIs1970_fall_s1_13_p265.png' -> 'ROIs1970_fall_s2_13_p265.png'
                s2_filename = list(s1_file.name.split('_'))
                s2_filename[2] = 's2'
                s2_file = s2_path / '_'.join(s2_filename)

                if not s2_file.exists():
                    # print(f"Missing optical image for SAR image: {s1_file.name} - {s2_file.name}")
                    continue

                image_pairs.append((s1_file, s2_file))
        
        return image_pairs
    
    def _apply_predefined_split(self, split_file: Union[str, Path]) -> List[Tuple[Path, Path]]:
        """
        Applies a predefined split from a JSON file.
        
        Args:
            split_file: Path to JSON file containing split definitions
            
        Returns:
            List[Tuple[Path, Path]]: Image pairs for the specified split
        """
        try:
            with open(split_file, 'r') as f: # get the split content
                splits = json.load(f)
                
            if self.split_type.value not in splits['data']: # check if it helds
                raise ValueError(f"Split type {self.split_type.value} not found in split file")
            
            split_filenames = set(splits['data'][self.split_type.value]) # data['split']
            return [pair for pair in self.all_image_pairs # collect and return split
                if any(str(p.relative_to(self.root_dir)) in split_filenames for p in pair[:2])]
        except Exception as e:
            print(f'Could not open split file\n\t{e}')
            raise

    def _apply_random_split(
        self, 
        split_ratio: Tuple[float, float, float],
        seed: int
    ) -> List[Tuple[Path, Path]]:
        """
        Randomly splits the dataset according to the given ratios.
        
        Args:
            split_ratio: Tuple of (train, val, test) ratios
            seed: Random seed for reproducibility
            
        Returns:
            List[Tuple[Path, Path]]: Image pairs for the specified split
        """
        if sum(split_ratio) != 1:
            raise ValueError("Split ratios must sum to 1")
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Shuffle indices
        indices = list(range(len(self.all_image_pairs)))
        random.shuffle(indices)
        
        # Calculate split points
        train_end = int(len(indices) * split_ratio[0])
        val_end = train_end + int(len(indices) * split_ratio[1])
        
        # Select appropriate slice based on split type
        if self.split_type == SplitType.TRAIN:
            split_indices = indices[:train_end]
        elif self.split_type == SplitType.VAL:
            split_indices = indices[train_end:val_end]
        else:  # TEST
            split_indices = indices[val_end:]
            
        return [self.all_image_pairs[i] for i in split_indices]
    
    def save_split(self, output_file: Union[str, Path], is_exists: bool = False):
        """
        Saves the current split configuration to a JSON file.
        
        Args:
            output_file: Path to save the split configuration
            is_exists: If file exist, add new split data
        """
        if self.split_type:
            split = self.split_type.value
            split_info = {
                'data' : {
                    split: [str(p[0].relative_to(self.root_dir)) for p in self.image_pairs]
                }
            }
            # Check if the file exists
            if is_exists and Path(output_file).exists():
                # Read the existing content
                with open(output_file, 'r') as f:
                    existing_data = json.load(f)
                # Check if 'data' is already in the existing content, if not, create it
                if 'data' not in existing_data:
                    existing_data['data'] = {}
                
                # Add or update the split information
                existing_data['data'][split] = split_info['data'][split]
                split_info = existing_data
            
            with open(output_file, 'w') as f:
                json.dump(split_info, f, indent=2)
    
    def __len__(self):
        """Returns the total number of image pairs in the dataset."""
        return len(self.image_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the image pair at the given index.
        
        Args:
            idx (int): Index of the image pair to retrieve
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Processed (SAR image, optical image) pair
        """
        # Get paths for SAR and optical images
        s1_path, s2_path = self.image_pairs[idx]
        
        # Load images
        s1_image = Image.open(s1_path).convert('RGB')
        s2_image = Image.open(s2_path).convert('RGB')
        
        # Apply transforms
        s1_image = self.input_transform(s1_image)
        s2_image = self.target_transform(s2_image)
        
        return s1_image, s2_image