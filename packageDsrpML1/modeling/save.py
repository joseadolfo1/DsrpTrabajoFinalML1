from packageDsrpML1.config import MODELS_DIR
import joblib

def save_ml_model(ml_object, name):
    """
    Guarda modelos de ML
    """
    path = f"{MODELS_DIR}/{name}.joblib"
    joblib.dump(ml_object,path)
    print(f"Modelo guardado exitosamente en: {path}")