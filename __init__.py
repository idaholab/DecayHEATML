"""
Decay Heat Prediction Package

A comprehensive toolkit for modeling and visualizing decay heat phenomena in nuclear materials.
"""

__version__ = "0.1.0"
__all__ = [
    'DataLoader',
    'ModelTrainer',
    'Visualizer',
    'PolynomialFitter',
    'Config'
]

# Core components
from DecayHeatML.data_loader import DataLoader
from DecayHeatML.model_trainer import ModelTrainer
from DecayHeatML.visualizer import Visualizer
from DecayHeatML.polynomial_fitter import PolynomialFitter
from DecayHeatML.config import Config

# Initialize package-level logging
import logging

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)
logger.info(f"Decay Heat Package v{__version__} initialized")

def get_version():
    """Return the current package version"""
    return __version__

def list_available_models():
    """List all available regression models in the package"""
    from .model_trainer import ModelTrainer
    return list(ModelTrainer(pd.DataFrame()).models.keys())

# Add type hinting for better IDE support
__init__: 'decay_heat'
DataLoader: 'DataLoader'
ModelTrainer: 'ModelTrainer'
Visualizer: 'Visualizer'
PolynomialFitter: 'PolynomialFitter'
Config: 'Config'
