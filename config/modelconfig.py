from typing import List, Dict, Optional, Union, Sequence
from pydantic import BaseModel

class BaseModelConfig(BaseModel):
    """
    Base configuration class for all models.
    Contains parameters common to most models.
    """
    target: str
    features: List[str]
    test_size: float
    random_state: int
    temp_features: List[str]
    qual_vars: List[str]
    categorical_vars: Sequence[str]
    qual_mappings: Dict[str, int]

class RandomForestModelConfig(BaseModel):
    """
    Configuration for RandomForest model, extending BaseModelConfig.
    """
    n_estimators: int
    max_depth: int

class XGBoostModelConfig(BaseModel):
    """
    Configuration for XGBoost model, extending BaseModelConfig.
    XGBoost may have different hyperparameters.
    """
    learning_rate: float
    n_estimators: int
    max_depth: int
    gamma: Optional[float] = None

class ModelConfig(BaseModel):
    """
    Model configuration, can be used generically with a specific model config.
    """
    model_type: str  # Could be 'RandomForest', 'XGBoost', etc.
    config: Union[RandomForestModelConfig, XGBoostModelConfig, Dict]

# Example of how you could structure the config for a RandomForest model
random_forest_config = {
    "model_type": "RandomForest",
    "config": {
        "target": "target_column",
        "features": ["feature1", "feature2", "feature3"],
        "test_size": 0.2,
        "random_state": 42,
        "temp_features": ["temp1", "temp2"],
        "qual_vars": ["qual1"],
        "categorical_vars": ["cat1"],
        "qual_mappings": {"qual1": 1},
        "n_estimators": 100,
        "max_depth": 10
    }
}

rf_config = ModelConfig(**random_forest_config)
print(rf_config)
