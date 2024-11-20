import typing as t
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from model.config.core import config, TRAINED_MODEL_DIR
from model.processing.data_manager import load_pipeline, load_label_encoders
from model.processing.validation import validate_inputs
from app.config import settings
import numpy as np

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
    validated_data = data
    errors = None
    results = {"predictions": None, "version": _version, "errors": errors}
    print("***************** sin errores uno", validated_data)
    
    # Codificar las variables categóricas
    print("***************** sin errores uno???", _label_encoders.items())
    for col, le in _label_encoders.items():
        if col in validated_data.columns:
            try:
                validated_data[col] = le.transform(validated_data[col])
            except ValueError as e:
                # Manejar valores no vistos durante el entrenamiento
                unseen_labels = set(validated_data[col]) - set(le.classes_)
                for unseen_label in unseen_labels:
                    validated_data[col] = validated_data[col].replace(unseen_label, -1)
                le.classes_ = np.append(le.classes_, -1)
                validated_data[col] = le.transform(validated_data[col])
    
    print("***************** sin errores dos**")
    print("***************** sin errores dos", validated_data)
    
    # Seleccionar las características relevantes
    features = config.model_configs[settings.MODEL_NAME].features
    print("***************** sin errores tres")
    predictions2 = _abandono_pipe.predict(validated_data)
    print("***************** sin errores cuatro", predictions2)
    
    if not errors:
        print("***************** sin errores cinco", validated_data)
        print("***************** sin errores seis", validated_data[features])
        predictions = _abandono_pipe.predict(validated_data[features])
        print("***************** sin errores siete", predictions)
        results = {
            "predictions": [pred for pred in predictions], 
            "version": _version,
            "errors": errors,
        }

    return results