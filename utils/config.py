# utils/config.py
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

class Config:
    """Configuration class to handle YAML config files"""
    
    def __init__(self, config_path: str, overrides: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration from YAML file with optional overrides
        
        Args:
            config_path: Path to YAML config file
            overrides: Optional dictionary of values to override config
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        # Load config file
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        # Apply any overrides
        if overrides:
            self._override_config(overrides)
            
        # Set up paths
        self._setup_paths()
        
    def _override_config(self, overrides: Dict[str, Any]):
        """Recursively override configuration values"""
        def _update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = _update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
            
        _update(self.config, overrides)
        
    def _setup_paths(self):
        """Setup output directories and experiment naming"""
        # Create unique experiment name if not specified
        if not self.config['logging']['comet']['name']:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.config['logging']['comet']['name'] = f"pix2pix_{timestamp}"
            
        # Setup directories
        for dir_name in ['checkpoint_dir', 'results_dir']:
            path = Path(self.config['training'][dir_name])
            path = path / self.config['logging']['comet']['name']
            path.mkdir(parents=True, exist_ok=True)
            self.config['training'][dir_name] = str(path)
            
    def __getitem__(self, key):
        return self.config[key]
    
    def get(self, key, default=None):
        """Get config value with optional default"""
        try:
            return self[key]
        except KeyError:
            return default
            
    def save(self, save_path: str):
        """Save current config to file"""
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)