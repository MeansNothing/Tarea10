import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

# Definir las 5 características que usaremos
SELECTED_FEATURES = [
    'mean radius',
    'mean texture',
    'mean perimeter',
    'mean area',
    'mean smoothness'
]


def train_and_save_model():
    data = load_breast_cancer()

    # Crear DataFrame y seleccionar solo nuestras características
    df = pd.DataFrame(data.data, columns=data.feature_names)
    X = df[SELECTED_FEATURES].values
    y = data.target

    # Entrenar modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Guardar modelo y características
    joblib.dump(model, 'breast_cancer_model.joblib')
    joblib.dump(SELECTED_FEATURES, 'feature_names.joblib')


def load_model():
    model = joblib.load('breast_cancer_model.joblib')
    feature_names = joblib.load('feature_names.joblib')
    return model, feature_names