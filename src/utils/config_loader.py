"""
Configuration loader for daily Bitcoin classifier.
"""

import yaml
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_feature_names(config: Dict[str, Any]) -> list:
    """Extract all feature names from config."""
    features = []
    
    if 'external' in config['features']:
        features.extend(config['features']['external'])
    
    if 'bitcoin' in config['features']:
        features.extend(config['features']['bitcoin'])
    
    if 'technical' in config['features']:
        features.extend(config['features']['technical'])
    
    if 'lagged' in config['features']:
        features.extend(config['features']['lagged'])
    
    if 'blockchain' in config['features']:
        features.extend(config['features']['blockchain'])
    
    return features
