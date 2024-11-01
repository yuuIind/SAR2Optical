from typing import Tuple, List, Optional, Callable
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


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

    Args:
        root_dir (str | Path): Root directory containing the dataset
        transform (callable, optional): Transform to apply to both SAR and optical images
        
    Attributes:
        root_dir (Path): Path to the dataset root directory
        transform (callable): Transform pipeline for the images
        image_pairs (List[Tuple[Path, Path]]): List of paired image paths (SAR, optical)
    """
    def __init__(self, root_dir: str | Path, transform: Optional[Callable] = None):
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root directory not found: {self.root_dir}")

        # Default transform pipeline
        self.transform = transform if transform else v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

        # Collect image pairs
        self.image_pairs = self._collect_images()

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
        s1_image = self.transform(s1_image)
        s2_image = self.transform(s2_image)
        
        return s1_image, s2_image


if __name__ == '__main__':
    path = './data/'

    sent = Sentinel(root_dir=path)