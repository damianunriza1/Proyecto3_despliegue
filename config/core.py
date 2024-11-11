from pathlib import Path
from typing import Dict, List, Optional, Union, Sequence

from pydantic import BaseModel
from strictyaml import YAML, load

import model

# Project Directories
PACKAGE_ROOT = Path(model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained"

# Base model configuration (common for all models)
class AppConfig(BaseModel):
    """
    Application-level config.
    """
    package_name: str
    train_data_file: str
    test_data_file: str
    pipeline_save_file: str

# Configurations specific to RandomForest or other models
class RandomForestModelConfig(BaseModel):
    """
    Configuration for RandomForest model.
    """
    target: str
    features: List[str]
    test_size: float
    random_state: int
    n_estimators: int
    max_depth: int
    temp_features: List[str]
    qual_vars: List[str]
    categorical_vars: Sequence[str]
    qual_mappings: Dict[str, int]

class XGBoostModelConfig(BaseModel):
    """
    Configuration for XGBoost model.
    """
    target: str
    features: List[str]
    test_size: float
    random_state: int
    learning_rate: float
    n_estimators: int
    max_depth: int
    gamma: Optional[float] = None

# Main ModelConfig that will dynamically adjust based on the model type
class ModelConfig(BaseModel):
    """
    Main config that can handle multiple model types.
    """
    model_type: str  # Could be 'RandomForest', 'XGBoost', etc.
    config: Union[RandomForestModelConfig, XGBoostModelConfig, Dict]

# Master Config
class Config(BaseModel):
    """Master config object combining application and model config."""
    app_config: AppConfig
    model_config: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    """Parse YAML containing the package configuration."""
    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # Here we dynamically create the Config object based on the model_type in the YAML
    model_type = parsed_config.data.get('model_type', 'RandomForest')
    
    # Dynamically load the specific model config based on model_type
    if model_type == 'RandomForest':
        model_config = RandomForestModelConfig(**parsed_config.data['config'])
    elif model_type == 'XGBoost':
        model_config = XGBoostModelConfig(**parsed_config.data['config'])
    else:
        model_config = parsed_config.data['config']  # for generic model config

    _config = Config(
        app_config=AppConfig(**parsed_config.data['app_config']),
        model_config=ModelConfig(model_type=model_type, config=model_config)
    )

    return _config


# Load configuration
config = create_and_validate_config()
