from pathlib import Path
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

def download_and_organize_dataset(dataset_name: str, local_data_path: Path):
    """
    Download Kaggle dataset and organize it in the specified local data folder.
    
    Args:
        dataset_name (str): Kaggle dataset name in format 'owner/dataset'
        local_data_path (Path): Local path where dataset will be downloaded and organized
    """
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Create temporary download directory
    temp_download_dir = local_data_path / 'temp_download'
    temp_download_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    try:
        api.dataset_download_files(dataset_name, path=temp_download_dir, unzip=True)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return
    
    # Move and organize files
    v2_source = temp_download_dir / 'v_2'
    if v2_source.exists():
      shutil.move(str(v2_source), str(local_data_path))
    
    # Clean up temporary download directory
    shutil.rmtree(temp_download_dir)
    
    print(f"Dataset downloaded and organized in {local_data_path / 'v_2'}")

def main():
    # Configuration
    DATASET_NAME = 'requiemonk/sentinel12-image-pairs-segregated-by-terrain'
    LOCAL_DATA_PATH = Path('data') # /SAR2OPTICAL/data
    
    download_and_organize_dataset(DATASET_NAME, LOCAL_DATA_PATH)

if __name__ == '__main__':
    main()