import joblib
from pathlib import Path
from typing import List
from sklearn.pipeline import Pipeline
from model.config.core import config, TRAINED_MODEL_DIR

def remove_old_pipelines(*, files_to_keep: List[str]) -> None:
    """
    Remove old model pipelines.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()

def save_pipeline(*, pipeline_to_persist: Pipeline, model_name:str) -> None:
    """
    Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """
    # Prepare versioned save file name
    model_version = config.model_configs[model_name].version
    save_file_name = f"{config.app_config.pipeline_save_file}-{model_name}-{model_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)

def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""
    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(file_path)
    return trained_model