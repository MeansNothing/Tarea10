import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def train_and_save_model():
    # Cargar dataset de cáncer de seno
    data = load_breast_cancer()

    # Seleccionar solo las 5 características que usaremos
    selected_features = [
        'mean radius',
        'mean texture',
        'mean perimeter',
        'mean area',
        'mean smoothness'
    ]

    # Obtener índices de las características seleccionadas
    feature_indices = [i for i, name in enumerate(data.feature_names)
                       if name in selected_features]

    X = pd.DataFrame(data.data[:, feature_indices], columns=selected_features)
    y = data.target

    # Resto del código sigue igual...
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Guardar modelo
    joblib.dump(model, 'breast_cancer_model.joblib')
    joblib.dump(data.feature_names, 'feature_names.joblib')


def load_model():
    model = joblib.load('breast_cancer_model.joblib')
    feature_names = joblib.load('feature_names.joblib')
    return model, feature_names

# Ejecutar esto una vez para entrenar y guardar el modelo
# train_and_save_model()