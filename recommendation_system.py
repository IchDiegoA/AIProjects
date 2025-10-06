# recommendation_system.py

# Importar las librerias necesarias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Cargar los datos (ejemplo: dataset de productos)
# Este dataset debe estar previamente descargado o puede ser generado
data = {
    'ProductID': [1, 2, 3, 4, 5],
    'Feature1': [10, 20, 30, 40, 50],
    'Feature2': [5, 15, 25, 35, 45],
    'Feature3': [2, 4, 6, 8, 10]
}
df = pd.DataFrame(data)

# Preprocesamiento de datos
# Seleccionar las caracteristicas y la etiqueta
features = df[['Feature1', 'Feature2', 'Feature3']]
product_ids = df['ProductID']

# Escalar las caracteristicas
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(scaled_features, product_ids, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = NearestNeighbors(n_neighbors=3, algorithm='auto')
model.fit(X_train)

# Funcion de recomendacion
def recommend(product_id, n_recommendations=3):
    product_index = product_ids[product_ids == product_id].index[0]
    product_features = scaled_features[product_index].reshape(1, -1)
    distances, indices = model.kneighbors(product_features, n_neighbors=n_recommendations)
    recommendations = product_ids.iloc[indices.flatten()].tolist()
    recommendations.remove(product_id)  # Excluir el producto actual de las recomendaciones
    return recommendations

# Ejemplo de uso de la funcion de recomendacion
example_product_id = 1
recommendations = recommend(example_product_id)
print(f"Recomendaciones para el producto {example_product_id}: {recommendations}")