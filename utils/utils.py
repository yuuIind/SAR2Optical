import logging
from pathlib import Path

from .config import Config

def setup_logging(config: Config):
    """Setup logging configuration"""
    results_dir = Path(config['training']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(results_dir / "training.log"),
            logging.StreamHandler()
        ]
    )

def init_comet(config: Config):
    """Initialize Comet ML logging"""
    if config['logging']['comet']['enabled']:
        import comet_ml

        comet_ml.login()
        # Create experiment object
        project = config['logging']['comet']['project']
        # You can further customizate logging behavior by updating 
        # `config.yaml` file. Check comet_ml documents for details about 
        # experiment config : 
        # https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/ExperimentConfig/#comet_ml.ExperimentConfig
        experiment_config = comet_ml.ExperimentConfig(
            name=config['logging']['comet']['name']
            )
        
        experiment = comet_ml.start(
            project_name=project,
            experiment_config=experiment_config
            )
        return experiment
    return None

def log_metrics(experiment, metrics: dict, step: int, last: int = None):
    """Log metrics to Comet if enabled, otherwise use local logging."""
    if experiment:
        experiment.log_metrics(metrics, epoch=step)
    else:
        if last:
            logging.info(f"Step {step}/{last} metrics: {metrics}")
        else:
            logging.info(f"Step {step} metrics: {metrics}")