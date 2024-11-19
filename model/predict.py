import typing as t
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from model.config.core import config, TRAINED_MODEL_DIR
from model.processing.data_manager import load_pipeline, load_label_encoders
from model.processing.validation import validate_inputs
from app.config import settings

_version = config.model_configs[settings.MODEL_NAME].version
pipeline_file_name = f"{config.app_config.pipeline_save_file}-{settings.MODEL_NAME}-{_version}.pkl"
_abandono_pipe = load_pipeline(file_name=pipeline_file_name)
_label_encoders, _label_encoder_target = load_label_encoders(model_name=settings.MODEL_NAME)

def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:
    """Make a prediction using a saved model pipeline."""
    
    # Convertir los datos de entrada a un DataFrame de Pandas
    data = pd.DataFrame(input_data)
    
    # Validar los datos de entrada
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "version": _version, "errors": errors}
    
    # Codificar las variables categóricas
    for col, le in _label_encoders.items():
        if col in validated_data.columns:
            validated_data[col] = le.transform(validated_data[col])
    
    # Seleccionar las características relevantes
    features = config.model_configs[settings.MODEL_NAME].features
    print("***************** sin errores1")
    
    if not errors:
        print("***************** sin errores2")
        predictions = _abandono_pipe.predict(validated_data[features])
        print("***************** sin errores3", predictions)
        results = {
            "predictions": [pred for pred in predictions], 
            "version": _version,
            "errors": errors,
        }

    return results