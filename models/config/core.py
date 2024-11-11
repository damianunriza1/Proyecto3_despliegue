from pathlib import Path
from typing import Dict, List, Optional, Union, Sequence

from pydantic import BaseModel
from strictyaml import YAML, load

import models

# Project Directories
PACKAGE_ROOT = Path(models.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained"
DEFAULT_CONFIG_FILE = PACKAGE_ROOT / "config.yml"

# Base models configuration (common for all models)
class AppConfig(BaseModel):
    """
    Application-level config.
    """
    package_name: str
    train_data_file: str
    test_data_file: str
    pipeline_save_file: str

# Configurations specific to RamdomForestClassifier or other models
class RandomForestModelConfig(BaseModel):
    """
    Configuration for RamdomForestClassifier models.
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
    Configuration for XGBoost models.
    """
    target: str
    features: List[str]
    test_size: float
    random_state: int
    learning_rate: float
    n_estimators: int
    max_depth: int
    gamma: Optional[float] = None

# Main ModelConfig that will dynamically adjust based on the models type
class ModelConfig(BaseModel):
    """
    Main config that can handle multiple models types.
    """
    model_type: str  # Could be 'RamdomForestClassifier', 'XGBoost', etc.
    config: Union[RandomForestModelConfig, XGBoostModelConfig, Dict]

# Master Config
class Config(BaseModel):
    """Master config object combining application and models config."""
    app_config: AppConfig
    model_config: ModelConfig


def find_config_file(model_type: Optional[str] = 'RandomForestClassifier') -> Path:
    """Locate the configuration file based on the model type."""
    
    # Puedes usar el tipo de modelo para definir la ruta o el nombre del archivo de configuración
    config_filename = f"{model_type.lower()}_config.yml"
    model_config_path = PACKAGE_ROOT / "configs" / config_filename  # Ruta a un subdirectorio "configs"
    
    # Primero, intentamos encontrar el archivo específico del modelo
    if model_config_path.is_file():
        return model_config_path
    
    # Si no se encuentra el archivo específico, busca el archivo por defecto
    if DEFAULT_CONFIG_FILE.is_file():
        return DEFAULT_CONFIG_FILE
    
    # Si no se encuentra ningún archivo, lanzamos un error
    raise FileNotFoundError(f"Config file for {model_type} not found. Tried: {model_config_path} and {DEFAULT_CONFIG_FILE}")


def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    """Parse YAML containing the package configuration."""
    if not cfg_path:
        cfg_path = find_config_file()

    try:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    except FileNotFoundError:
        raise OSError(f"Did not find config file at path: {cfg_path}")
    except Exception as e:
        raise OSError(f"Error while reading config file: {e}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # Ensure the model_type is provided and valid
    model_type = parsed_config.data.get('model_type', 'RamdomForestClassifier')
    if model_type not in ['RamdomForestClassifier', 'XGBoost']:
        raise ValueError(f"Invalid model_type '{model_type}' in config. Supported models are 'RamdomForestClassifier' and 'XGBoost'.")

    # Dynamically load the specific model config based on model_type
    if model_type == 'RamdomForestClassifier':
        model_config = RandomForestModelConfig(**parsed_config.data['config'])
    elif model_type == 'XGBoost':
        model_config = XGBoostModelConfig(**parsed_config.data['config'])
    else:
        model_config = parsed_config.data['config']  # for generic models config

    _config = Config(
        app_config=AppConfig(**parsed_config.data['app_config']),
        model_config=ModelConfig(model_type=model_type, config=model_config)
    )

    return _config


# Load configuration
config = create_and_validate_config()
