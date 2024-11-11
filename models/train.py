import os
import sys
import importlib
import yaml
from sklearn.model_selection import train_test_split
from config.core import config
from processing.data_manager import load_dataset, save_pipeline

def load_pipeline(model_name):
    """
    Carga el pipeline del modelo especificado.
    """
    model_path = os.path.join("models", model_name)
    
    # Cargar la configuración del modelo
    with open(os.path.join(model_path, "config.yml"), "r") as file:
        model_config = yaml.safe_load(file)
    
    # Importar dinámicamente el pipeline
    pipeline_module = importlib.import_module(f"models.{model_name}.pipeline")
    
    return pipeline_module.abandono_pipe, model_config

def run_training(model_name):
    """
    Ejecuta el entrenamiento del modelo especificado.
    """
    abandono_pipe, model_config = load_pipeline(model_name)
    
    # Cargar los datos de entrenamiento
    data = load_dataset(file_name=config.app_config.train_data_file)
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        data[model_config["features"]],
        data[model_config["target"]],
        test_size=model_config["test_size"],
        random_state=model_config["random_state"],
    )
    
    # Mapear valores categóricos en y_train si es necesario
    y_train = y_train.map(model_config["qual_mappings"])
    
    # Entrenar el modelo
    abandono_pipe.fit(X_train, y_train)
    
    # Guardar el pipeline entrenado
    save_pipeline(pipeline_to_persist=abandono_pipe, name=f"{model_name}_pipeline")

if __name__ == "__main__":
    # El nombre del modelo se pasa como argumento de línea de comandos
    model_name = sys.argv[1] if len(sys.argv) > 1 else "RandomForestClassifier"
    run_training(model_name)
