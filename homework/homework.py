# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "../files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
import pandas as pd

dataset_test = pd.read_csv(
    "..files/input/test_data.csv.zip",
    index_col=False,
    compression="zip",
)
dataset_train = pd.read_csv(
    "..files/input/train_data.csv.zip",
    index_col=False,
    compression="zip",
)
# - Renombre la columna "default payment next month" a "default".
dataset_test.rename(columns={"default payment next month": "default"}, inplace=True)
dataset_train.rename(columns={"default payment next month": "default"}, inplace=True)
# - Remueva la columna "ID".
dataset_test.pop("ID")
dataset_train.pop("ID")
# - Elimine los registros con informacion no disponible.
dataset_test = dataset_test.dropna()
dataset = dataset_train.dropna()
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
dataset_test['EDUCATION'] = dataset_test['EDUCATION'].apply(lambda x: "others" if x > 4 else x)
dataset_train['EDUCATION'] = dataset_train['EDUCATION'].apply(lambda x: "others" if x > 4 else x)

# - Renombre la columna "default payment next month" a "default" (Repetido)
# - Remueva la columna "ID". (Repetido)
#
#



# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
y_train=dataset_train.copy()
y_train=y_train.pop("default")

x_train = dataset_train.copy()
x_train.pop("default")

y_test=dataset_test.copy()
y_test=y_test.pop("default")

x_test = dataset_test.copy()
x_test.pop("default")
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (random forest).
#
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

# Bloque 1: Crear un transformador personalizado para castear a entero
class CastToInteger(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].astype(str)  # Convertir a entero
        return X

# Bloque 2: Identificar las columnas categóricas
categorical_columns = ['SEX', 'EDUCATION', 'MARRIAGE']

# Bloque 3: Preprocesamiento de variables categóricas
# Usamos un transformador personalizado para castear y luego aplicar One-Hot Encoding
# categorical_transformer = Pipeline(steps=[
#     ('cast_to_int', CastToInteger(categorical_columns)),  # Castear columnas categóricas a entero
#     ('imputer', SimpleImputer(strategy='most_frequent')),  # Imputar valores faltantes si existieran
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))    # Aplicar One-Hot Encoding
# ])

# Bloque 4: Crear el ColumnTransformer para aplicar transformaciones específicas
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder, categorical_columns)  # Transformar solo las categóricas
    ],
    remainder='passthrough'  # Mantener las columnas numéricas sin cambios
)

# Bloque 5: Modelo de Bosques Aleatorios
random_forest_model = RandomForestClassifier(
    n_estimators=100,        # Número de árboles
    max_depth=None,          # Sin límite para la profundidad de los árboles
    random_state=42          # Asegurar reproducibilidad
)

# Bloque 6: Crear el Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Preprocesar los datos
    ('classifier', random_forest_model)  # Modelo de Bosques Aleatorios
])

# Bloque 7: Entrenar el Pipeline con el conjunto de entrenamiento
pipeline.fit(x_train, y_train)
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
from sklearn.model_selection import GridSearchCV

# Paso 4: Optimización de hiperparámetros con validación cruzada

# Definir los hiperparámetros a optimizar
param_grid = {
    'classifier__n_estimators': [50, 100, 200],  # Número de árboles
    'classifier__max_depth': [None, 10, 20, 30],  # Profundidad máxima
    'classifier__min_samples_split': [2, 5, 10],  # Número mínimo de muestras para dividir un nodo
    'classifier__min_samples_leaf': [1, 2, 4],   # Número mínimo de muestras por hoja
}

# Configurar GridSearchCV con validación cruzada (10 splits) y precisión balanceada
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='balanced_accuracy',  # Precisión balanceada
    cv=10,                        # 10 particiones para validación cruzada
    n_jobs=-1,                    # Usar todos los núcleos disponibles
    verbose=2                     # Mostrar progreso
)

# Ejecutar la búsqueda de hiperparámetros
grid_search.fit(x_train, y_train)

# Mostrar los mejores parámetros encontrados
print("\nMejores parámetros encontrados:")
print(grid_search.best_params_)

# Actualizar el pipeline con los mejores parámetros
best_pipeline = grid_search.best_estimator_
#
# Paso 5.
# Guarde el modelo como "../files/models/model.pkl".
#
import os
import joblib

# Paso 5: Guardar el modelo optimizado, asegurándose de que la ruta exista

# Definir la ruta del archivo de salida
output_dir = "../files/models"
output_path = os.path.join(output_dir, "model.pkl")

# Crear las carpetas necesarias si no existen
os.makedirs(output_dir, exist_ok=True)

# Guardar el modelo en el archivo
joblib.dump(best_pipeline, output_path)
print(f"\nModelo guardado exitosamente en '{output_path}'")
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo"../files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
import os
import json
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score

# Paso 6: Calcular métricas y guardar en un archivo JSON, asegurándose de que la ruta exista

# Definir la función para calcular métricas
def calculate_metrics(y_true, y_pred, dataset_name):
    metrics = {
        'dataset': dataset_name,
        'precision': precision_score(y_true, y_pred, average='binary'),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1_score': f1_score(y_true, y_pred, average='binary'),
    }
    return metrics

# Generar predicciones para los conjuntos de entrenamiento y prueba
y_train_pred = best_pipeline.predict(x_train)
y_test_pred = best_pipeline.predict(x_test)

# Calcular métricas para cada conjunto
train_metrics = calculate_metrics(y_train, y_train_pred, 'train')
test_metrics = calculate_metrics(y_test, y_test_pred, 'test')

# Definir la ruta del archivo de salida
output_dir = "../files/output"
output_path = os.path.join(output_dir, "metrics.json")

# Crear las carpetas necesarias si no existen
os.makedirs(output_dir, exist_ok=True)

# Guardar las métricas en un archivo JSON
with open(output_path, 'w') as f:
    json.dump([train_metrics, test_metrics], f, indent=4)

print(f"\nMétricas guardadas exitosamente en '{output_path}'")
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo"../files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import os
import json
from sklearn.metrics import confusion_matrix

# Paso 7: Calcular matrices de confusión y añadirlas al archivo JSON

# Función para generar la matriz de confusión en el formato requerido
def format_confusion_matrix(cm, dataset_name):
    return {
        'type': 'cm_matrix',
        'dataset': dataset_name,
        'true_0': {
            'predicted_0': int(cm[0, 0]),
            'predicted_1': int(cm[0, 1])
        },
        'true_1': {
            'predicted_0': int(cm[1, 0]),
            'predicted_1': int(cm[1, 1])
        }
    }

# Calcular las matrices de confusión
cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

# Formatear las matrices de confusión
train_cm_metrics = format_confusion_matrix(cm_train, 'train')
test_cm_metrics = format_confusion_matrix(cm_test, 'test')

# Definir la ruta del archivo para las matrices de confusión
confusion_matrix_dir = "../files/output"
confusion_matrix_path = os.path.join(confusion_matrix_dir, "metrics.json")

# Crear las carpetas necesarias si no existen
os.makedirs(confusion_matrix_dir, exist_ok=True)

# Cargar métricas existentes o inicializar una lista vacía si el archivo está vacío o no existe
metrics = []
if os.path.exists(confusion_matrix_path):
    try:
        with open(confusion_matrix_path, 'r') as f:
            content = f.read().strip()  # Leer contenido del archivo
            if content:  # Validar que no esté vacío
                metrics = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        print(f"Advertencia: El archivo '{confusion_matrix_path}' tiene un formato inválido. Será sobrescrito.")

# Añadir las matrices de confusión al archivo
metrics.extend([train_cm_metrics, test_cm_metrics])

# Guardar el archivo actualizado
with open(confusion_matrix_path, 'w') as f:
    json.dump(metrics, f, indent=4)

print(f"\nMatrices de confusión guardadas exitosamente en '{confusion_matrix_path}'")
