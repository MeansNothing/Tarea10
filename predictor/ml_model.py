import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def train_and_save_model():
    # Cargar dataset de cáncer de seno
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Guardar modelo
    joblib.dump(model, 'breast_cancer_model.joblib')
    joblib.dump(data.feature_names, 'feature_names.joblib')


def load_model():
    model = joblib.load('breast_cancer_model.joblib')
    feature_names = joblib.load('feature_names.joblib')
    return model, feature_names

# Ejecutar esto una vez para entrenar y guardar el modelo
# train_and_save_model()